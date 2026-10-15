"""Everything the chatbot knows, indexed for hybrid retrieval (schema/retrieval.py).

    examples  verified question -> SQL patterns: every phrasing + note   (examples.yaml)
    sql       the same examples by their SQL: tables, columns, functions (examples.yaml)
    topics    KPI rule areas: title, description, keywords               (business_rules.yaml)
    glossary  KPI / term definitions                                     (business_rules.yaml)
    charts    when each chart type fits                                  (charts.yaml)
    tables    every catalog table as a DATA CARD: purpose, engine, rows, real date
              range, dimensions with real values, measures   (catalog + data skeleton)
    values    every geography path + every real dimension-value combination (data skeleton)
    columns   every catalog column: table, label, meaning, type and its
              REAL sample values ("SEIPL", "Meter Burnt", "EDD CHUNAR")   (live catalog + table_metadata.yaml)
    questions the question bank + every question answered with SQL, and that SQL
                                                                         (questions.yaml + logs/questions.jsonl)

Each index is cached on disk (vectors) and in memory (BM25), keyed by a hash of
its texts, so it is rebuilt only when its source changes (a YAML edit or a
catalog refresh). Queries are expanded with business_rules.yaml
retrieval_synonyms for recall; the plain question drives reranking and fuzzy
near-exact matching.
"""

from __future__ import annotations

import logging
import re
import time

from shared import config
from gmr_agents.schema import business_rules, retrieval
from gmr_agents.schema.table_metadata import SCHEMA_NAMES, SCHEMA_RE

log = logging.getLogger(__name__)

_SQL_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SQL_KEYWORDS = {
    "select", "from", "where", "and", "or", "not", "as", "on", "by", "group", "order", "desc", "asc", "limit", "case",
    "when", "then", "else", "end", "is", "null", "in", "with", "union", "all", "distinct", "having", "between",
    *SCHEMA_NAMES, "filter", "over", "partition", "true", "false", "interval", "date",
    "current_date", "join", "left", "cross", "like", "numeric", "int", "text",
}


# --------------------------------------------------------------------------- #
# Items per index: [(item id, [texts])]
# --------------------------------------------------------------------------- #

def _sql_terms(sql: str) -> str:
    """The tables, columns and functions a query uses, as plain words."""
    seen: list[str] = []
    for w in _SQL_WORD.findall(str(sql)):
        lw = w.lower()
        if lw not in _SQL_KEYWORDS and lw not in seen and not lw.isdigit():
            seen.append(lw)
    return " ".join(seen).replace("_", " ")


def sql_items() -> list[tuple[str, list[str]]]:
    items = []
    for i, ex in enumerate(business_rules.examples()):
        texts = [_sql_terms(ex["sql"])] + [_sql_terms(e.get("sql", "")) for e in ex.get("extra") or []]
        items.append((str(i), [t for t in texts if t]))
    return items


def glossary_items() -> list[tuple[str, list[str]]]:
    return [(key, [f"{key.replace('_', ' ')}: {text}"]) for key, text in business_rules.glossary().items()]


def chart_items() -> list[tuple[str, list[str]]]:
    data = business_rules._load(business_rules.CHARTS_PATH)
    return [(str(c.get("type")), [f"{c.get('type')}: {c.get('when', '')}", str(c.get("example", ""))])
            for c in data.get("charts") or [] if c.get("type")]


def _catalog_tables():
    from gmr_agents.schema.catalog import get_catalog

    tables = []
    for domain in ("wfm", "mdm"):
        try:
            tables.extend(get_catalog(domain).list_tables())
        except Exception:
            continue
    return tables


def _cards() -> dict[str, dict]:
    from gmr_agents.schema import data_skeleton

    return {c["table"]: c for c in data_skeleton.load().get("tables") or []}


def table_card(name: str) -> str:
    """One line describing a table from its data skeleton: purpose, engine, rows,
    date range, key dimensions with real values, measures."""
    from gmr_agents.schema.table_metadata import TABLE_DESCRIPTIONS

    card = _cards().get(name, {})
    bits = [f"{name} [{card.get('engine', '?')}, {card.get('rows', '?')} rows]: {TABLE_DESCRIPTIONS.get(name, '')}".strip()]
    if card.get("dates"):
        bits.append("dates " + "; ".join(f"{c} {v[0]}..{v[1]}" for c, v in card["dates"].items() if v))
    if card.get("dimensions"):
        bits.append("dims " + "; ".join(f"{c}={'/'.join(map(str, v[:6]))}" for c, v in list(card["dimensions"].items())[:8]))
    if card.get("measures"):
        bits.append("measures " + ", ".join(card["measures"][:10]))
    if card.get("keys"):
        bits.append("row = one " + " x ".join(card["keys"][:8]))
    if card.get("joins"):
        bits.append("joins " + "; ".join(card["joins"][:3]))
    return " | ".join(bits)


def table_items() -> list[tuple[str, list[str]]]:
    items = []
    for t in _catalog_tables():
        cols = ", ".join(c.column_name.replace("_", " ") for c in t.columns)
        from gmr_agents.schema import schema_dictionary

        items.append((t.table_name, [table_card(t.table_name), cols[:1500],
                                     *[x for x in schema_dictionary.text_for_index(t.table_name) if x.strip()]]))
    return items


def value_items() -> list[tuple[str, list[str]]]:
    """Every geography path and every real dimension-value combination (data skeleton)."""
    from gmr_agents.schema import data_skeleton

    skel = data_skeleton.load()
    items = []
    for i, g in enumerate(skel.get("geography") or []):
        path = " > ".join(f"{k} {v}" for k, v in g.items() if v)
        items.append((f"geo:{i}", [path]))
    for card in skel.get("tables") or []:
        for j, combo in enumerate(card.get("combos") or []):
            text = ", ".join(f"{k}={v}" for k, v in combo.items() if v)
            if text:
                items.append((f"{card['table']}:{j}", [f"{card['table']}: {text}"]))
    return items


def column_items() -> list[tuple[str, list[str]]]:
    from gmr_agents.schema.table_metadata import COLUMN_HINTS

    items = []
    for t in _catalog_tables():
        for c in t.columns:
            hint = COLUMN_HINTS.get((t.table_name, c.column_name), "")
            texts = [f"{c.header_label} ({c.column_name.replace('_', ' ')}) in {t.table_name.replace('_', ' ')}"
                     f"{': ' + hint if hint else ''} [{c.sql_type}]"]
            if c.sample_values:
                texts.append("values: " + ", ".join(str(v) for v in c.sample_values[:25]))
            items.append((f"{t.table_name}.{c.column_name}", texts))
    return items


_PAST_SQL: dict[str, dict] = {}


def question_items() -> list[tuple[str, list[str]]]:
    """Every question the chatbot has answered with SQL (outputs/logs/questions.jsonl,
    newest PAST_QUESTIONS_MAX; refusals, errors and empty results left out) plus the
    question bank (PREWARM_QUESTIONS_FILE) -- each with its asked and resolved wording.
    The SQL that answered it is kept in _PAST_SQL for past_questions_for()."""
    import json
    import os

    import yaml

    seen: dict[str, dict] = {}
    path = os.path.join(config.LOG_DIR, "questions.jsonl")
    try:
        with open(path, encoding="utf-8") as fh:
            lines = fh.readlines()[-config.PAST_QUESTIONS_MAX:]
    except OSError:
        lines = []
    for line in lines:
        try:
            r = json.loads(line)
        except ValueError:
            continue
        q = str(r.get("question") or "").strip()
        if not q or r.get("refused") or r.get("error") or not r.get("sql") or not r.get("rows"):
            continue
        seen[re.sub(r"\s+", " ", q.lower())] = {"question": q, "resolved": r.get("resolved_question") or "",
                                                "sql": r["sql"], "extra": r.get("extra_sql") or []}
    bank = config.PREWARM_QUESTIONS_FILE
    bank = bank if os.path.isabs(bank) else os.path.join(config.PROJECT_ROOT, bank)
    try:
        sections = (yaml.safe_load(open(bank, encoding="utf-8")) or {}).get("sections") or {}
    except OSError:
        sections = {}
    for group in sections.values():
        for item in group or []:
            q = str(item.get("q") or "").strip()
            key = re.sub(r"\s+", " ", q.lower())
            if q and key not in seen:
                seen[key] = {"question": q, "resolved": "", "sql": None, "extra": []}
    _PAST_SQL.clear()
    _PAST_SQL.update(seen)
    return [(key, [v["question"]] + ([v["resolved"]] if v["resolved"] and v["resolved"] != v["question"] else []))
            for key, v in seen.items()]


def kpi_chunks() -> dict:
    """kpi_definitions.yaml as logical-unit chunks (schema/chunking.py), by id.
    Rebuilt when the YAML file changes."""
    import os

    from gmr_agents.schema import chunking

    path = os.path.join(os.path.dirname(business_rules.PATH), "kpi_definitions.yaml")
    mtime = os.path.getmtime(path) if os.path.exists(path) else 0.0
    if _KPI_CHUNKS.get("mtime") != mtime:
        data = business_rules._load(path) if mtime else {}
        chunks = chunking.chunk_document(data or {}, min_tokens=config.CHUNK_MIN_TOKENS,
                                         max_tokens=config.CHUNK_MAX_TOKENS)
        _KPI_CHUNKS.update(mtime=mtime, by_id={c.chunk_id: c for c in chunks})
    return _KPI_CHUNKS["by_id"]


_KPI_CHUNKS: dict = {}


def kpi_items() -> list[tuple[str, list[str]]]:
    return [(cid, [c.text]) for cid, c in kpi_chunks().items()]


_SOURCES = {
    "examples": business_rules._example_items,
    "sql": sql_items,
    "topics": business_rules._topic_items,
    "glossary": glossary_items,
    "charts": chart_items,
    "tables": table_items,
    "columns": column_items,
    "questions": question_items,
    "values": value_items,
    "kpis": kpi_items,
}
INDEXES = tuple(_SOURCES)
_ITEMS: dict[str, tuple[float, list]] = {}


def items(index: str) -> list[tuple[str, list[str]]]:
    """Index items, memoised for RETRIEVAL_ITEMS_TTL seconds (the YAML sources
    themselves reload only when their file changes; the catalog refreshes hourly)."""
    now = time.monotonic()
    cached = _ITEMS.get(index)
    if cached and now - cached[0] < config.RETRIEVAL_ITEMS_TTL:
        return cached[1]
    built = _SOURCES[index]()
    _ITEMS[index] = (now, built)
    return built


def search(index: str, question: str, k: int, debug: list | None = None, allow=None) -> list[tuple[str, float]]:
    return retrieval.search(index, items(index), business_rules.expand_query(question), k, rerank_query=question,
                            debug=debug, allow=allow)


def warm_all() -> dict[str, int]:
    """Build/refresh every index (startup): texts indexed per index."""
    return retrieval.warm({name: items(name) for name in INDEXES})


# --------------------------------------------------------------------------- #
# What the pipeline asks for
# --------------------------------------------------------------------------- #

def examples_for(question: str, k: int) -> list[str]:
    """Example ids: the wording match fused with the SQL-pattern match (RRF)."""
    rrf_k = config.RETRIEVAL_RRF_K
    fused: dict[str, float] = {}
    for weight, hits in ((1.0, search("examples", question, k * 2)),
                         (config.RETRIEVAL_SQL_WEIGHT, search("sql", question, k * 2))):
        for rank, (item, score) in enumerate(hits):
            # Near-exact wording matches (score >= 10) keep their lead.
            fused[item] = fused.get(item, 0.0) + (score if score >= 10 else 0.0) + weight / (rrf_k + rank + 1)
    return [i for i, _s in sorted(fused.items(), key=lambda kv: -kv[1])[:k]]


def schema_hints(question: str, tables: list[str] | None = None) -> str:
    """The catalog columns most relevant to the question -- with their real values --
    as short lines for the SQL prompt ('oem in gold_meter_defect_event_day_agg:
    AEW, CAP, HPL, SEIPL'). Restricted to `tables` when given -- as a pre-filter, so
    columns of other tables never crowd the chosen tables' columns out."""
    wanted = set(tables or [])
    allow = {item for item, _t in items("columns") if item.split(".", 1)[0] in wanted} if wanted else None
    lines = []
    for item, _score in search("columns", question, config.SCHEMA_HINT_COLUMNS, allow=allow):
        table, column = item.split(".", 1)
        values = next((t for t in retrieval.item_texts("columns", item) if t.startswith("values: ")), "")
        lines.append(f"- {table}.{column}{(' -> ' + _whole_values(values[8:], 192)) if values else ''}")
    values = value_hints(question, config.VALUE_HINTS)
    if values:
        lines.append("stored values matching the question: " + " || ".join(values))
    return "\n".join(lines)


def _whole_values(values: str, max_chars: int) -> str:
    """A ', '-joined value list cut only between values, never inside one (a cut at a
    fixed length sent partial spellings such as 'S2DLC1' that a filter could copy;
    measured: 9 of 162 value lists, 7 of 33 eval prompts)."""
    if len(values) <= max_chars:
        return values
    parts = values.split(", ")
    kept: list[str] = []
    for p in parts:
        if len(", ".join([*kept, p])) > max_chars:
            break
        kept.append(p)
    return ", ".join(kept) + f", ... (+{len(parts) - len(kept)} more)"


_TABLE_REF = re.compile(rf"\b{SCHEMA_RE}\.\"?([A-Za-z0-9_]+)")


def _norm_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", str(sql).lower()).strip()


_GUARD_OK: dict[str, bool] = {}


def _passes_guard(sql: str) -> bool:
    """A past answer is replayed as a hint only if TODAY's safety check still accepts its
    SQL -- the checks grow (verified live: a cross-joined 9-row 'mi pace vs stock category
    wise' answer kept being copied after the check that rejects it was added)."""
    key = _norm_sql(sql)
    if key not in _GUARD_OK:
        from gmr_agents.schema.catalog import get_catalog
        from gmr_agents.validation import sql_guard

        try:
            _GUARD_OK[key] = any(sql_guard.check_sql_safety(sql, get_catalog(d))["ok"] for d in ("wfm", "mdm"))
        except Exception:
            log.warning("past question: safety re-check failed -- left out", exc_info=True)
            _GUARD_OK[key] = False
    return _GUARD_OK[key]


def past_questions_for(question: str, k: int, tables: list[str] | None = None) -> list[dict]:
    """The k most similar past ANSWERED questions with the SQL that answered them
    (episodic memory: adapt-don't-copy patterns, after the verified examples).

    Retrieved only when it can help (measured on 296 logged questions: 21% of hints
    repeated a verified example's SQL, 24% used tables outside the question's schema --
    and the retrieval score could not tell them from useful ones):
      * `tables` (the prompt's schema tables) is a metadata filter: a past SQL on other
        tables is left out -- the query writer cannot see their columns, so copying it
        invites unknown-column errors;
      * a past SQL identical to a verified example is left out -- the example (semantic
        memory, hand-verified) is already in the prompt."""
    items("questions")  # make sure _PAST_SQL is loaded
    wanted = {t.lower() for t in tables or []}
    verified = {_norm_sql(e.get("sql") or "") for e in business_rules.examples()}
    out = []
    for key, score in search("questions", question, k * 4):
        past = _PAST_SQL.get(key)
        if not past or not past.get("sql") or score < config.PAST_QUESTIONS_MIN_SCORE:
            continue
        if _norm_sql(past["sql"]) in verified or not _passes_guard(past["sql"]):
            continue
        used = {t.lower() for t in _TABLE_REF.findall(past["sql"])}
        if wanted and not (used and used <= wanted):
            continue
        out.append({**past, "score": round(score, 3)})
        if len(out) >= k:
            break
    return out


def value_hints(question: str, k: int) -> list[str]:
    """Stored values (geography paths / dimension combinations) matching the question --
    exact spellings and their parents, e.g. 'project kashi > zone mirzapur > division edd chunar'."""
    return [t for item, _s in search("values", question, k) for t in retrieval.item_texts("values", item)][:k]


def compact_table_card(name: str) -> str:
    """A table card without real values: purpose, engine, date range (months), dimension and
    measure names. Enough to choose tables; the SQL step gets the real stored values through
    the column hints. A/B on 20 live paraphrased / Hinglish questions: -10% tokens per
    question (triage prompt 4.5k -> 3.9k), tables matching the verified SQL 15 -> 16, repairs 2 -> 0."""
    from gmr_agents.schema.table_metadata import TABLE_DESCRIPTIONS

    card = _cards().get(name, {})
    bits = [f"{name} [{card.get('engine', '?')}]: {TABLE_DESCRIPTIONS.get(name, '')}".strip()]
    if card.get("dates"):
        bits.append("dates " + "; ".join(f"{c} {str(v[0])[:7]}..{str(v[1])[:7]}" for c, v in card["dates"].items() if v))
    if card.get("dimensions"):
        bits.append("dims " + ", ".join(list(card["dimensions"])[:8]))
    if card.get("measures"):
        bits.append("measures " + ", ".join(card["measures"][:6]))
    return " | ".join(bits)


def table_hints(question: str, k: int) -> str:
    """The k most relevant tables as compact one-line cards (for the understanding step)."""
    return "\n".join(f"- {compact_table_card(t)}" for t in tables_for(question, k))


_KNOWN_VALUES: dict = {}


def known_value(value: str) -> bool:
    """Is this string a real stored value (lower-case match) -- a catalog column's sample
    value, a data-skeleton dimension value or a geography name? A LIKE fragment matches
    when it is part of one. Rebuilt with the catalog / skeleton (RETRIEVAL_ITEMS_TTL)."""
    from gmr_agents.schema import data_skeleton

    now = time.monotonic()
    if not _KNOWN_VALUES or now - _KNOWN_VALUES["at"] >= config.RETRIEVAL_ITEMS_TTL:
        values: set[str] = set()
        for t in _catalog_tables():
            for c in t.columns:
                values.update(str(v).strip().lower() for v in c.sample_values or [])
        skel = data_skeleton.load()
        for card in skel.get("tables") or []:
            for vs in (card.get("dimensions") or {}).values():
                values.update(str(v).strip().lower() for v in vs)
        for g in skel.get("geography") or []:
            values.update(str(v).strip().lower() for v in g.values() if v)
        values.discard("")
        _KNOWN_VALUES.update(at=now, values=values)
    v = str(value).strip().lower()
    if "%" in v:
        frag = v.replace("%", "")
        return bool(frag) and any(frag in known for known in _KNOWN_VALUES["values"])
    return v in _KNOWN_VALUES["values"]


def tables_for(question: str, k: int) -> list[str]:
    return [t for t, _s in search("tables", question, k)]


def relevance(index: str, question: str) -> dict[str, float] | None:
    """Cosine relevance of every item of an index to the question (grounding guard)."""
    return retrieval.similarities(index, items(index), question)


def chart_order(question: str) -> list[str]:
    return [t for t, _s in search("charts", question, len(items("charts")))]
