"""SQL Writer agent: the planner prompt (schema, rules, examples, hints), plan parsing,
repair hints (Query Debugger) and the parallel extra queries.

Moved from gmr_agents/llm_pipeline.py unchanged.
"""

from __future__ import annotations
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any
import pandas as pd
from shared import config
from gmr_agents.catalog import business_rules, schema_dictionary
from gmr_agents.retrieval import knowledge
from gmr_agents.catalog.schema_brief import build as build_schema
from gmr_agents.presentation import stats as _stats
from gmr_agents.catalog.table_metadata import SCHEMA_RE
from gmr_agents.execution import executor
from gmr_agents.ops import context_manager
from gmr_agents.guards import injection
from gmr_agents.resolvers import query_encoder
from gmr_agents.guards import sql_guard
import logging
from gmr_agents.agents.common import _ANSI_RE, _LAST_CONTEXT, _SCHEMA_TOKEN_BUDGET, _step, _today

log = logging.getLogger("gmr_agents.llm_pipeline")  # same logger name as before the split


# --------------------------------------------------------------------------- #
# 2. Plan + repair
# --------------------------------------------------------------------------- #

_PLAN_SYSTEM = """You write the read-only SQL SELECT that answers a business question, and decide how to present the result.
- If the question itself gives every number the answer needs (e.g. "what is 12% of 4,214,678?", "1,277,180 installed and 1,106,912 passed SAT -- what percent passed?"), do not query the data for numbers the user already gave: reply {"tool": "calculator", "expression": "<numbers with + - * / ** ( ) and round/abs/min/max/sqrt>", "reasoning": "..."}.
- Use only tables/columns in SCHEMA, referenced exactly as in their ### header. One query = one engine (never mix postgres and athena tables).
- postgres tables: PostgreSQL syntax. athena tables: Trino/Presto syntax.
- Follow GLOBAL RULES and TOPIC rules exactly: they define how every KPI is computed so numbers match the dashboard. Use exactly the filters the rules give for a KPI -- never add extra boolean/flag filters (e.g. has_sat_number) the rules do not list, and never drop one they list.
- Apply ONLY the filters the question states (or the previous turn, for a follow-up). Never copy a place, period or category filter from an example. In top-N lists do not add a ROLLUP total row; when you do use ROLLUP, label the GROUPING row 'Total', never 'Unmapped'. 'Which <project/discom/zone/category> has the highest / lowest X' (no N given) returns EVERY one of them ranked by X (no LIMIT 1), so the answer can say who is next and by how much.
- Give every selected expression a snake_case alias. No SELECT *. Lists: ORDER BY + LIMIT (200 unless asked).
- 'why', 'should we', 'is it rising', 'what changed', summaries and what-ifs: return the numbers that answer it (comparisons, shares, trends) -- the answer writer reasons from them. 'Why did X rise / fall / change' (no breakdown named), when X's table has a date / period column: return X for the period asked about AND the one before it (the latest complete period and the previous one when none is named), split by the most useful place / category (zone or discom; OEM / agency when the question is about them) -- one row per (period, group), the period column first; no ROLLUP total row. When X comes from a snapshot with no history (TOPIC rules say so, e.g. SAT, invoice): do not look for dates in another engine -- return X's current breakdown; the answer says the month-by-month history is not stored.
- When the answer needs data from another engine or a differently-shaped result (e.g. a batch-date table plus a meter-count table, a trend plus a total), put up to 3 more queries in extra_queries (they run in parallel). Keep them small.
- Summaries, status briefs, CXO notes, 'risks and actions', drafts: never unanswerable -- query the headline KPIs (dashboard stats, SAT / non-SAT, stock, defects, collections) and the answer writer derives risks and actions from those numbers.
- If the data genuinely cannot answer it, set answer_kind "unanswerable" and explain in reason -- reason is shown to the user as is: plain business words, never table or column names. A 'why' / 'how come' / 'explain the difference' question is NOT unanswerable when the figures behind it can be shown: return those figures side by side.
Reply with JSON only:
{"reasoning": "ONE short sentence: table, filters, formula",
 "answer_kind": "single_value" | "breakdown" | "trend" | "list" | "record" | "unanswerable",
 "sql": "the main query (its result drives the cards/chart/table)",
 "extra_queries": [{"purpose": "what it adds", "sql": "..."}] or [],
 "labels": {"alias": "Readable Header", ...},
 "display": {"cards": [aliases of 0-6 headline numbers shown as cards. Cards read the FIRST row only, so use them only when the first row IS the overall answer (a single-row result, or a 'Total' row placed first); otherwise []],
             "chart": {"type": one type from CHART GUIDE -- the BEST match for this question and result shape, "x": "alias", "y": ["alias", ...], "y2": ["line alias"] (combo only), "group": "alias" (long rows -> one series per value), "unit": "...", "unit2": "..."},
             "table": true if the rows should be shown as a table},
 "title": "short title for the result",
 "reason": "only when unanswerable"}"""


def _plan_prompt(triage: dict, catalog, previous: dict | None) -> tuple[str, list[str]]:
    by_name = {t.table_name: t for t in catalog.list_tables()}
    question = " ".join([triage["question"], *triage.get("parts", [])])
    names = business_rules.tables_for(triage["topics"]) + triage["tables"]
    if config.SCHEMA_SCOPE == "lean" and triage.get("tables") and triage.get("source") != "router":
        # The understanding step named the tables this question needs: send those, not every
        # table linked to its topics (topic tables only fill in when it named none).
        # (Adding the best verified example's tables was A/B-tested on 20 live questions:
        # +12% tokens, tables matching the verified SQL 16 -> 15 -- not adopted.)
        names = list(triage["tables"])
    if triage.get("source") == "router":
        # Routed by its verified wording: the example's own SQL names exactly the tables needed --
        # send only those (smaller prompt, same tables the verified pattern uses).
        examples = business_rules.examples()
        ex_tables: list[str] = []
        for e in triage.get("examples") or []:
            i = str(e).lstrip("e")
            if i.isdigit() and int(i) < len(examples):
                ex = examples[int(i)]
                for sql in [ex.get("sql") or ""] + [x.get("sql") or "" for x in ex.get("extra") or []]:
                    ex_tables += re.findall(rf"\b{SCHEMA_RE}\.\"?([A-Za-z0-9_]+)", sql)
        if ex_tables:
            names = list(dict.fromkeys(t.lower() for t in ex_tables))
    if not names:  # no topic/table chosen: the schema index suggests tables
        names = knowledge.tables_for(question, 3)
    wanted = []
    for name in names:
        if name in by_name and by_name[name] not in wanted:
            wanted.append(by_name[name])
    schema_text, used = build_schema(wanted, [], token_budget=_SCHEMA_TOKEN_BUDGET)
    # Schema dictionary headers: purpose, grain, real date range, relationships, key filters.
    # Table guide: what a table holds and how to filter it -- each relationship / key filter
    # shown once per prompt (not again under the next table, nor when the rules already say it).
    seen: set[str] = set()
    rules_known = business_rules.rules_text(triage["topics"][:2])
    explained = [h for h in (schema_dictionary.header(t.table_name, seen, rules_known) for t in wanted) if h]
    if explained:
        schema_text = "TABLE GUIDE:" + chr(10) + chr(10).join(explained) + chr(10) + chr(10) + schema_text
    # Sections in prompt order, each with a priority for the token budget
    # (gmr_agents/context_manager.py): rules first, the question last; past answered
    # questions and the chart guide are cut first, the question / rules / schema never.
    S = context_manager.Section
    rules = business_rules.rules_text(triage["topics"][:2], with_examples="rule_examples" not in config.PROMPT_COMPACT)
    sections = [S("rules", 0, [f"Today: {_today()}", rules], required=True),
                S("schema", 2, [f"SCHEMA:\n{schema_text}"], required=True)]
    hints = knowledge.schema_hints(question, [t.table_name for t in wanted])
    if hints and "hint_values" in config.PROMPT_COMPACT:
        hints = _dedupe_hint_values(hints, schema_text)
    if hints:
        sections.append(S("column_hints", 3, injection.filter_items(hints.splitlines(), "column_hints"),
                          joiner="\n", required=True,
                          header="RELEVANT COLUMNS (retrieved for this question, with their REAL stored values -- use "
                                 "these exact spellings in filters):"))
    if triage.get("cross_domain"):
        from gmr_agents.agents.sql_writer.cross_domain import PLAN_HINT as _XD_HINT
        sections.append(S("cross_domain", 1, [_XD_HINT], required=True))
    if _stats.wanted(question):
        # Statistics are computed by code from the rows (presentation/stats.py): ask for a series.
        sections.append(S("analysis_rows", 1, [_stats.PLAN_HINT], required=True))
    sections.append(S("chart_guide", 6, business_rules.chart_guide(question).splitlines(), joiner="\n", min_items=12,
                      header="CHART GUIDE (pick the chart whose 'when' best matches the question and the rows your SQL "
                             "returns; shape the SQL so the chart can be drawn -- e.g. one column per stacked part, or a "
                             "group column; listed most relevant first):"))
    # Adaptive k: a question routed by a near-exact wording match (top-1 correct 24/24 on the
    # labelled retrieval set) gets ROUTED_EXAMPLES -- its own verified example -- not the full set.
    k = config.ROUTED_EXAMPLES if triage.get("source") == "router" and config.ROUTED_EXAMPLES else config.EXAMPLES_PER_QUESTION
    similar = business_rules.examples_blocks(question, k=k, chosen=triage.get("examples"))
    if similar:
        sections.append(S("examples", 4, similar, min_items=min(2, len(similar)),
                          header="SIMILAR VERIFIED EXAMPLES (correct patterns from past questions -- use your judgement: "
                                 "adapt them to THIS question's places, periods, categories and intent; ignore one that "
                                 "does not fit):"))
    # A question the router matched to a verified example is answered by that example: an unverified
    # past answer to the same wording only competes with it. Seen live: 'how many closed tickets in
    # August 2026' copied a logged one-query SQL and skipped the example's work-completed extra query.
    past = ([] if triage.get("source") == "router" else
            knowledge.past_questions_for(question, config.PAST_QUESTIONS_TOP_K, [t.table_name for t in wanted]))
    if past:
        # Whole SQL, never cut: a 900-char cut sent broken SQL (29 of 560 logged); skipping long
        # ones instead lost a past answer to the very same question (A/B, 2026-10-04). The
        # section is still the first one the token budget trims.
        sections.append(S("past_questions", 6, [injection.wrap(t, "past_question") for t in injection.filter_items(
                              [f"Q: {p['question']}\n{str(p['sql']).strip()}" for p in past], "past_questions")],
                          header="SIMILAR PAST ANSWERED QUESTIONS (not hand-verified -- a hint only; the rules and verified "
                                 "examples win):"))
    if previous and (previous.get("context") or {}).get("sql") and triage.get("source") != "router":
        # (A question routed by its verified wording is complete on its own: no previous-turn SQL,
        # so a filter from the last answer -- e.g. 'kashi' -- can never leak into it.)
        # Memory: the previous turn's SQL is always offered. A confirmed follow-up must adapt it;
        # otherwise the query writer decides whether this question builds on it (deep-dive chats).
        label = ("PREVIOUS ANSWER (this question is a follow-up -- adapt its SQL):" if triage.get("followup") else
                 "PREVIOUS TURN (continue its filters / grouping only if this question builds on it):")
        sections.append(S("previous_turn", 3, [f"{label}\n" + injection.wrap(
                                               f"Q: {previous.get('resolved_question') or previous.get('question')}\n"
                                               # whole SQL: a cut at 1,200 chars broke 5% of logged
                                               # queries (max 2,489 chars, ~620 tokens)
                                               f"SQL: {str(previous['context']['sql']).strip()}", "previous_turn")],
                          required=bool(triage.get("followup"))))
    # Encoder slots read by code from the (resolved) question: exact values, N and direction.
    encoded = query_encoder.encode(triage["question"])
    question_items = [f"QUESTION: {triage['question']}"] + ([query_encoder.prompt_line(encoded)] if encoded else [])
    if len(triage.get("parts") or []) > 1:
        question_items.append("PARTS -- answer EVERY one (main query for the first, extra_queries for the rest, or one "
                              "query that covers several):\n"
                              + "\n".join(f"{i}. {p}" for i, p in enumerate(triage["parts"], 1)))
    sections.append(S("question", 1, question_items, required=True))
    if config.PROMPT_ORDER == "edges":
        sections = _edges_order(sections)
    built = context_manager.build(sections, budget=config.CONTEXT_BUDGET_SQL_TOKENS, system=_PLAN_SYSTEM)
    if not context_manager.fits_hard_limit(built, config.MAX_CONTEXT_TOKENS):
        # The soft per-prompt budget (CONTEXT_BUDGET_SQL_TOKENS) already trimmed everything
        # it could; this is the absolute ceiling check right before the prompt is handed to
        # the LLM gateway (which would otherwise refuse the call as CONTEXT_LIMIT_EXCEEDED --
        # this just makes that failure diagnosable instead of a silent "no model available").
        log.error("context: planner prompt ~%d tokens exceeds the hard limit %d even after trimming",
                  built.tokens, config.MAX_CONTEXT_TOKENS)
    _LAST_CONTEXT[triage["question"]] = {**built.metrics(), "encoded": encoded or None}
    if "lean_repair" in config.PROMPT_COMPACT:
        # A repair fixes a column / syntax / filter error: rules, schema, column values, the
        # question and the error are what it needs. Chart guide, examples and past questions
        # (~2,200 tokens, measured) are left out -- the failed SQL already shows the pattern.
        lean = context_manager.build([s for s in sections if s.name not in ("chart_guide", "examples", "past_questions")],
                                     budget=config.CONTEXT_BUDGET_SQL_TOKENS, system=_PLAN_SYSTEM)
        _REPAIR_PROMPT[triage["question"]] = lean.prompt
    return built.prompt, used


_REPAIR_PROMPT: dict[str, str] = {}  # question -> the lean planner prompt for repair calls (PROMPT_COMPACT lean_repair)


def repair_base(question: str, base_prompt: str) -> str:
    """The prompt a repair call starts from: the lean one when built, else the full prompt."""
    return _REPAIR_PROMPT.pop(question, None) or base_prompt


def _dedupe_hint_values(hints: str, schema_text: str) -> str:
    """Drop a column hint's '-> values' when SCHEMA already lists that column's values
    (seen: project / zone / defect-reason lists sent twice or three times per prompt)."""
    blocks = {}
    for block in schema_text.split("\n### "):
        m = re.match(rf"{SCHEMA_RE}\.([A-Za-z0-9_]+)\s+\(", block)
        if m:
            blocks[m.group(1).lower()] = block
    out = []
    for line in hints.splitlines():
        m = re.match(r"- ([A-Za-z0-9_]+)\.([A-Za-z0-9_]+) -> ", line)
        if m and re.search(rf"^- {re.escape(m.group(2))} \S+ = ", blocks.get(m.group(1).lower(), ""), re.M):
            line = f"- {m.group(1)}.{m.group(2)} (values listed in SCHEMA)"
        out.append(line)
    return "\n".join(out)


_SQL_START_RE = re.compile(r"^\s*(SELECT|WITH)\b", re.IGNORECASE)


_EDGE_ORDER = ["rules", "schema", "chart_guide", "past_questions", "column_hints", "examples", "previous_turn",
               "reminder", "question"]

_REMINDER = ("KEY REMINDERS: use only tables / columns in SCHEMA; follow the closest verified example (the last one "
             "above) adapted to this question; filter only on what the QUESTION states, with the exact stored values "
             "from RELEVANT COLUMNS; one read-only SELECT.")


def _edges_order(sections: list) -> list:
    """Lost-in-the-middle: models use the start and the end of a long prompt best. Rules stay
    first; low-stakes reference (chart guide, past questions) moves to the middle; the
    decision-critical evidence -- real column values, the verified examples with the BEST
    one last, the previous turn -- sits right before the question, after a short reminder
    of the key rules. Trimming still removes the least relevant items first."""
    by_name = {s.name: s for s in sections}
    ex = by_name.get("examples")
    if ex is not None:
        ex.items = list(reversed(ex.items))   # best example nearest the question
        ex.trim_from = "start"                # ...so trimming drops the least relevant one
    by_name["reminder"] = context_manager.Section("reminder", 1, [_REMINDER], required=True)
    ordered = [by_name[n] for n in _EDGE_ORDER if n in by_name]
    return ordered + [s for s in sections if s.name not in _EDGE_ORDER]


def _clean_sql(raw: Any) -> str:
    text = _ANSI_RE.sub("", str(raw or ""))
    text = "".join(ch for ch in text if ch in "\t\n\r" or ch >= " ").strip()
    text = re.sub(r"^```(?:sql)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    # Models sometimes drop the leading keyword ("CASE WHEN ... AS project, SUM(x) ... FROM t").
    # Restoring it is safe: the SQL guard still validates the whole statement.
    if text and not _SQL_START_RE.match(text) and re.search(r"\bFROM\b", text, re.IGNORECASE):
        text = "SELECT " + text
    # The Technical default month written as MAX over a GROUP BY (one row per qualifying month, right today only
    # because one month has 10+ days): the same subquery wrapped, as the rules give it. 10 Nov retest: the entity
    # check sent this back 14 times in one run (a full SQL repair round each, 5 never fixed).
    text = _FLAT_DEFAULT_MONTH_RE.sub(
        lambda m: (f"(SELECT MAX(month_date){m.group('alias') or ''} FROM (SELECT month_date FROM hes_gold.dt_load_profile "
                   f"WHERE month_date >= DATE '{m.group('since') or '2026-06-01'}' GROUP BY month_date "
                   "HAVING COUNT(DISTINCT data_date) >= 10) c)"), text)
    return text.rstrip(";").strip()


_FLAT_DEFAULT_MONTH_RE = re.compile(
    r"\(\s*SELECT\s+MAX\s*\(\s*month_date\s*\)(?P<alias>\s+AS\s+\w+)?\s+FROM\s+hes_gold\.dt_load_profile\s+"
    r"(?:WHERE\s+month_date\s*>=\s*DATE\s*'(?P<since>\d{4}-\d{2}-\d{2})'\s+)?GROUP\s+BY\s+month_date\s+"
    r"HAVING\s+COUNT\s*\(\s*DISTINCT\s+data_date\s*\)\s*>=\s*10\s*\)", re.IGNORECASE)


def _to_plan(raw: dict, guard: dict, catalog, model: str) -> dict:
    """The planner's JSON -> the plan dict the executor consumes. Nothing here
    changes the LLM's choices; it only carries them."""
    sql = guard["normalized_sql"]
    tables = sorted({t.lower() for t in re.findall(rf"\b{SCHEMA_RE}\.\"?([A-Za-z0-9_]+)", sql)})
    first = catalog.get_table(tables[0]) if tables else None
    display = raw.get("display") if isinstance(raw.get("display"), dict) else {}
    labels = raw.get("labels") if isinstance(raw.get("labels"), dict) else {}
    # Up to six: the home-page headline strip has six tiles (supplied .. PMPM collected).
    cards = [str(a) for a in display.get("cards") or []][:6]
    return {
        "engine": "sql", "data_engine": guard["data_engine"], "answer_kind": str(raw.get("answer_kind") or ""),
        "sql": sql, "database": first.database if first and first.engine == "athena" else None,
        "tables": tables, "labels": {str(k): str(v) for k, v in labels.items()}, "display": display,
        # The executor reads metrics/headline_alias; the LLM's cards are those.
        "metrics": [{"alias": a, "label": labels.get(a)} for a in cards], "group_by": [],
        "headline_alias": cards[0] if cards else None, "filters": [], "warnings": [],
        "planner": model, "reasoning": str(raw.get("reasoning") or "")[:600], "table_title": raw.get("title"),
        "skip_scope_count": True,
    }


_TABLE_REF_RE = re.compile(rf"\b{SCHEMA_RE}\.\"?([A-Za-z0-9_]+)")


def _repair_hint(sql: str, problem: str, catalog=None) -> str:
    hint = (f"\n\nYOUR PREVIOUS SQL FAILED -- fix it:\n{sql[:1500]}\n"
            f"PROBLEM: {problem[:700]}")
    if re.search(r"extract\(unknown, integer\)", problem, re.IGNORECASE):
        # Seen live ('mi pace vs stock category wise'): EXTRACT(EPOCH FROM date - date) failed,
        # and the repair kept the same expression because the error names no column.
        hint += ("\nIn Postgres, date - date is already an integer number of days: use (d1 - d2) directly "
                 "(e.g. CURRENT_DATE - DATE '2026-01-01' + 1 AS days), never EXTRACT(EPOCH/DAY FROM d1 - d2).")
    if re.search(r"has \d+ columns available but \d+ columns specified", problem, re.IGNORECASE):
        # Seen live ('receivables summary project wise'): a column name was added to the
        # VALUES alias list, e.g. AS v(project, ord, stream, ...), but not to each VALUES row.
        hint += ("\nAn alias column list (AS v(a, b, ...)) must name exactly as many columns as each VALUES row / "
                 "subquery has. Either add the value (e.g. l.project) to EVERY row or drop that name from the list; "
                 "for a per-project split, GROUP BY project instead of unpivoting.")
    if re.search(r"keyword\(s\) not allowed", problem, re.IGNORECASE):
        # The read-only check refuses command words anywhere, aliases included. 11 Nov: 'why did installations drop
        # last month' aliased a column AS drop; both repairs kept it and the question failed.
        hint += ("\nA column alias or name is a SQL command word (drop, update, delete, create, merge, copy ...), which "
                 "the read-only check refuses: rename it (e.g. drop -> fall_in_installs, update -> last_updated).")
    if re.search(r"column reference \"?\w+\"? is ambiguous|ambiguous column", problem, re.IGNORECASE):
        # 11 Nov: 'compare Agra and Kashi on installs, SAT % and defect rate' joined three per-project subqueries
        # and kept ORDER BY project -- the repair sent the same SQL twice and the question failed.
        hint += ("\nA column name exists in more than one joined table / subquery: qualify it EVERYWHERE it is used "
                 "(SELECT, WHERE, GROUP BY, ORDER BY) with its alias, e.g. i.project, or ORDER BY the output alias / "
                 "position (ORDER BY 1).")
    if catalog is not None and re.search(r"could not be resolved|Unknown column|does not exist", problem, re.IGNORECASE):
        # An unknown column usually means the SQL used a table whose schema was not in the
        # prompt: give that table's REAL columns, else the repair guesses again (seen live:
        # two repairs both rejected for the same invented 'oem' column).
        lines = []
        for name in dict.fromkeys(t.lower() for t in _TABLE_REF_RE.findall(sql)):
            t = catalog.get_table(name)
            if t is not None:
                # every column: a 900-char cut hid the tail of 5 of 40 tables (widest ~1,450 chars)
                lines.append(f"{name}: " + ", ".join(c.column_name for c in t.columns))
        if lines:
            hint += "\nREAL COLUMNS of the tables your SQL uses (use only these):\n" + "\n".join(lines)
    return hint


_FILTER_LITERAL_RE = re.compile(r"(?:=|<>|!=|\bin\s*\(|\blike|,)\s*'([^']*)'", re.IGNORECASE)

_DATE_LITERAL_RE = re.compile(r"^\d{4}(-\d{2}){0,2}([ t][\d:.]+)?$")


def _filter_literals(sql: str) -> list[str]:
    """String values the SQL compares against (=, <>, IN (...), LIKE) -- not labels after
    THEN / ELSE / AS, and not dates."""
    return [v for v in _FILTER_LITERAL_RE.findall(sql) if v.strip() and not _DATE_LITERAL_RE.match(v.strip().lower())]


def _zero_is_genuine(sql: str) -> bool:
    """Zero rows with every filter value a REAL stored value: the filters are right, so zero
    is the answer (e.g. no project below a stock threshold) -- a repair round would only
    resend the same SQL. An unknown value (a likely misspelling / wrong column) still
    gets the repair."""
    # A literal compared case-sensitively (plain '=' / IN, no LOWER / UPPER / ILIKE around it) must be stored
    # in that exact case. 10 Nov retest: circle = 'EDC Agra' (stored 'edc agra') -> 0 rows kept as the answer
    # ('no 1PH installs') instead of a repair; the real figure was 536.
    return all(knowledge.known_value(v) and (_case_insensitive(sql, v) or knowledge.known_value(v, exact=True))
               for v in _filter_literals(sql))


def _case_insensitive(sql: str, literal: str) -> bool:
    lit = re.escape(f"'{literal}'")
    return bool(re.search(r"\b(?:lower|upper)\s*\([^()]*\)\s*(?:=|<>|!=|(?:not\s+)?in\s*\([^)]*?|(?:not\s+)?like\s*)" + lit,
                          sql, re.I) or re.search(r"\bilike\s*" + lit, sql, re.I))


def _table_columns(table_names: list[str], catalog) -> set[str] | None:
    """Column names of these catalog tables, for the entity check (None if any is unknown:
    then every check runs). Date-typed columns are tagged '<name>_date' so a 'ts' or
    'refreshed' timestamp still counts as a date column."""
    cols: set[str] = set()
    for name in table_names:
        t = catalog.get_table(name)
        if t is None:
            return None
        for c in t.columns:
            cols.add(c.column_name.lower())
            if re.search(r"date|time", str(c.sql_type).lower()):
                cols.add(f"{c.column_name.lower()}_date")
    return cols or None


def _is_empty(df: pd.DataFrame | None, plan: dict) -> bool:
    """No rows, or one row whose card values are all NULL/0 (a filter matched nothing)."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return True
    cards = [m["alias"] for m in plan["metrics"] if m["alias"] in df.columns]
    return len(df) == 1 and bool(cards) and all(pd.isna(df.iloc[0][c]) or df.iloc[0][c] == 0 for c in cards)


# Share-of-a-whole columns can never pass 100%. 10 Nov retest: 'Compare Kashi and Triveni on installs last month,
# SAT % and defect rate' divided all-time SAT by one month's installs -> 'SAT % 1,140,615.93', 'defect rate 14,261%'.
# Growth / change / vs-target / loading / variance columns can legitimately exceed 100 and are not matched.
_SHARE_COL = re.compile(
    r"(?:sat|defect|faulty|burnt|share|conversion|complet|accura|efficien|collect|realis|realiz|utili[sz]|coverage|"
    r"complian|concluded|confirmed|pending|eligible|invoiced|stuck|success|availability)\w*(?:pct|percent|rate|_%)"
    r"|^(?:pct|percent)_(?:of_|sat|defect|share|complet|collect|realis|utili|eligible|pending|confirmed)", re.I)
_NOT_SHARE = re.compile(r"change|growth|delta|diff|vs_?(?:prev|last|target)|increase|decrease|rise|fall|load|variance", re.I)


def _impossible_shares(df: pd.DataFrame | None) -> list[str]:
    """Share columns (SAT %, defect rate %, collection % ...) holding a value above 100."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return []
    out = []
    for c in df.columns:
        name = str(c)
        if not _SHARE_COL.search(name) or _NOT_SHARE.search(name):
            continue
        vals = pd.to_numeric(df[c], errors="coerce")
        if vals.notna().any() and float(vals.max()) > 100.5:
            out.append(f"{name} = {float(vals.max()):,.2f}")
    return out


def _run_one_extra(i: int, ex: dict, raw: dict, catalog, model: str) -> dict:
    """Safety-check and run one extra query; never raises."""
    sql = _clean_sql(ex.get("sql"))
    t0 = time.perf_counter()
    guard = sql_guard.check_sql_safety(sql, catalog)
    if not guard["ok"]:
        return {"i": i, "t0": t0, "problem": f"extra query {i} was rejected by the safety check: {guard['reason']}\nSQL: {sql[:800]}",
                "log": {"outcome": "rejected", "reason": guard["reason"][:160]}}
    plan = _to_plan({**raw, "display": {}, "labels": {}}, guard, catalog, model)
    result = executor.execute_sql(plan, catalog)
    if result.get("execution_error"):
        return {"i": i, "t0": t0, "problem": f"extra query {i} failed in the database: {result.get('error_text')}\nSQL: {sql[:800]}",
                "log": {"error": (result.get("error_text") or "")[:160]}}
    return {"i": i, "t0": t0, "log": {"rows": len(result.get("result", []))},
            "extra": {"purpose": str(ex.get("purpose") or f"extra query {i}")[:160], "sql": plan["sql"],
                      "result": result.get("result")}}


def _run_extras(state: dict, raw: dict, catalog, model: str) -> tuple[list[dict], str | None]:
    """Run the planner's extra queries in parallel (latency = the slowest one).
    Returns the results and the first problem (for a repair round), if any."""
    extras = raw.get("extra_queries") if isinstance(raw.get("extra_queries"), list) else []
    todo = [(i, ex) for i, ex in enumerate(extras[:3], 1) if isinstance(ex, dict) and _clean_sql(ex.get("sql"))]
    if not todo:
        return [], None
    with ThreadPoolExecutor(max_workers=len(todo)) as pool:
        done = list(pool.map(lambda a: _run_one_extra(a[0], a[1], raw, catalog, model), todo))
    out, problem = [], None
    for d in sorted(done, key=lambda d: d["i"]):
        _step(state, f"extra{d['i']}", d["t0"], **d["log"])
        if "extra" in d:
            out.append(d["extra"])
        problem = problem or d.get("problem")
    return out, problem


_MISSING_DATA_RE = re.compile(r"schema|not (?:available|present|provided|included|contain)|does not (?:contain|have|include)|"
                              r"missing|insufficient|no (?:data|table|column|source)|cannot be determined|lacks? ",
                              re.IGNORECASE)


def _widened_tables(triage: dict) -> list[str]:
    """Tables a refusal retry adds: the topics' tables and the tables used by the 2 best
    verified examples, minus those already sent."""
    names = business_rules.tables_for(triage["topics"])
    for ex in business_rules.pick_examples(triage.get("examples"), triage["question"], 2):
        for sql in [ex.get("sql") or ""] + [x.get("sql") or "" for x in ex.get("extra") or []]:
            names += _TABLE_REF_RE.findall(sql)
    have = {t.lower() for t in triage["tables"]}
    return [t for t in dict.fromkeys(n.lower() for n in names) if t not in have]

_PLAN_SYSTEM += "\n- " + injection.UNTRUSTED_RULE
