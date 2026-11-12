"""SQL Writer agent nodes: controller (plan / repair -- the only LLM call), validate
(allowlist, SQL guard, Logic Critic, idempotency) and execute (Query Runner, Result Auditor).
Prompt and plan parsing: agents/sql_writer/planner.py.

Moved from gmr_agents/graph/nodes.py unchanged (llm_pipeline helpers are now called on
their own modules).
"""

from __future__ import annotations
import re
import time
from typing import Any
import sqlglot
from langchain_core.runnables import RunnableConfig
from sqlglot import exp
from gmr_agents.presentation import dashboard_charts
from gmr_agents.supervisor import context
from gmr_agents.supervisor.state import AgentState
from gmr_agents.guards import idempotency, limits, sql_validator
from gmr_agents.catalog import business_rules
from gmr_agents.catalog.catalog import get_catalog
from gmr_agents.execution.tools import calling, registry
from shared import config as settings
from shared import llm_gateway
from gmr_agents.agents import common as C
from gmr_agents.agents.sql_writer import planner as PL
from gmr_agents.agents.sql_writer import cross_domain as XD
from gmr_agents.supervisor.steps import _tick


# --------------------------------------------------------------------------- #
# controller -- decides the next action; never executes
# --------------------------------------------------------------------------- #

# The router matched the question to a verified example almost word for word: when the planner
# gives up on it ('unanswerable') or its SQL is rejected, that example's own SQL -- checked by hand
# against the dashboard -- runs instead of another free-tier attempt. Seen live 27 Oct: 'Write a CXO
# weekly summary with three risks and three actions' and 'The project totals add up to ~2.27L but the
# age buckets add to ~3.01L. Why?' were refused, 'Estimate PMPM billing value for this month' failed
# the safety check -- each one the exact wording of a verified example.
_EXAMPLE_FALLBACK_RATIO = 0.95


def _verified_example_plan(triage: dict) -> dict | None:
    if triage.get("source") != "router" or float(triage.get("match_ratio") or 0) < _EXAMPLE_FALLBACK_RATIO:
        return None
    ids = [str(e).lstrip("e") for e in triage.get("examples") or []]
    examples = business_rules.examples()
    if not ids or not ids[0].isdigit() or int(ids[0]) >= len(examples):
        return None
    ex = examples[int(ids[0])]
    return {"sql": str(ex["sql"]).strip(), "answer_kind": "breakdown", "display": {}, "labels": {},
            "extra_queries": [{"purpose": str(e.get("purpose") or ""), "sql": str(e.get("sql") or "").strip()}
                              for e in ex.get("extra") or [] if isinstance(e, dict) and e.get("sql")],
            "reasoning": ("Verified example for this wording. " + str(ex.get("note") or "")).strip()}


# Verified-SQL fast path (SQL_VERIFIED_DIRECT): the question IS a verified example's own
# wording (same words, case and punctuation aside), so that example's hand-checked SQL runs
# without a planner call (~6.7k tokens, ~5 s). Measured 28 Oct - 4 Nov: shown that very example,
# the planner wrote the same SQL for only half of these questions. Only where code decides the
# presentation anyway -- a dashboard card's rows (the card's chart) or a single-row figure
# (cards) -- since the planner's chart choice is skipped too; and never a dated SQL under a
# relative-time wording ('last month' written as '2026-09-01' would go stale). The guard, the
# entity check and the number checks all still run; a rejection or zero rows goes to the planner.
_RELATIVE_TIME_RE = re.compile(r"\b(today|yesterday|so far|to date|till date|current|currently|now|this|last|past|"
                               r"recent|latest|week|weeks|month|months|year|years|ytd|mtd|daily|monthly|weekly)\b", re.I)
_SQL_DATE_RE = re.compile(r"'\d{4}-\d{2}")


def _words(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", str(text).lower()))


def _code_presented(sql: str, question: str) -> bool:
    dialect = "postgres" if re.search(r"\bpublic\.", sql) else "presto"
    try:
        tree = sqlglot.parse_one(sql, dialect=dialect)
    except Exception:  # noqa: BLE001
        return False
    select = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    if select is None:
        return False
    columns = [e.alias_or_name for e in select.expressions]
    if dashboard_charts.kpi_for(sql, columns, question) and not dashboard_charts.user_named_type(question):
        return True
    return (isinstance(tree, exp.Select) and select.args.get("group") is None and tree.find(exp.Union) is None
            and all(e.find(exp.AggFunc) is not None for e in select.expressions))


def _verified_direct_plan(triage: dict, question: str) -> dict | None:
    if not settings.SQL_VERIFIED_DIRECT:
        return None
    plan = _verified_example_plan(triage)
    if plan is None:
        return None
    ex = business_rules.examples()[int(str(triage["examples"][0]).lstrip("e"))]
    # Only the example's own (first) wording: the others are patterns the planner adapts -- one
    # Kashi SQL also lists 'Agra mein SAT kitna pending hai' and 'How many meters are pending SAT?',
    # and running it as-is gave Kashi's 75,066 for both (checked against the database, 4 Nov).
    if _words(question) != _words((ex.get("q") or [""])[0]):
        return None
    if _SQL_DATE_RE.search(plan["sql"]) and _RELATIVE_TIME_RE.search(question):
        return None
    return plan if _code_presented(plan["sql"], question) else None


def controller_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Who controls it: the LLM writes the tool request; application code decides
    when no further tool is needed.

    * A validated, accepted result exists -> next action = final answer (no LLM call:
      re-asking the planner here would only burn quota and risk another loop).
    * Otherwise the planner LLM is asked for a `run_sql` request. On a retry it sees
      the previous SQL and the STRUCTURED error from the validator / executor.
    The returned `tool_call` is only a request -- validate_node decides if it runs.
    """
    ctx = context.get(config)
    ps = ctx.pipeline_state
    triage = state["triage"]
    if state.get("has_result"):
        return {"tool_call": None, "step_count": _tick(state)}

    attempt = state.get("attempt", 0)
    last_error = state.get("last_error")
    direct = None if ctx.example_fallback or last_error or attempt else _verified_direct_plan(triage, state["question"])
    if direct is not None:
        # One try only: a rejection or zero rows comes back here with last_error and goes to the
        # planner with its full retry budget (attempt is not spent).
        ctx.example_fallback = True
        ctx.raw_plan = direct
        C._step(ps, "plan", time.perf_counter(), attempt=attempt, model="verified example",
                outcome="verified SQL (direct)", example=triage["examples"][0])
        call = {"tool": "run_sql", "args": {"sql": PL._clean_sql(direct["sql"])}, "attempt": attempt,
                "model": "verified example"}
        return {"tool_call": call, "tool_calls": [call], "attempt": attempt, "validated": None,
                "last_error": None, "step_count": _tick(state)}
    if not ctx.base_prompt and "cross_domain" not in triage:
        # A question naming both systems (WFM + MDM): plan on both catalogs (see cross_domain.py). Decided once per
        # question; the triage dict is updated in place so validate / execute / answer see the same topics.
        other = XD.cross_domain(f"{state['question']} {triage['question']}", triage)
        triage.update(XD.widen(f"{state['question']} {triage['question']}", triage, other) if other else {"cross_domain": None})
        if triage.get("cross_domain"):
            C._step(ps, "cross_domain", time.perf_counter(), outcome=f"added {triage['cross_domain']}", topics=triage["topics"])
    catalog = get_catalog("all" if triage.get("cross_domain") else triage["domain"])
    if not ctx.base_prompt:
        t_ctx = time.perf_counter()
        ctx.base_prompt, used = PL._plan_prompt(triage, catalog, (state.get("prior_turns") or [None])[-1])
        ctx.repair_prompt = PL.repair_base(triage["question"], ctx.base_prompt)
        C._step(ps, "context", t_ctx, **C._LAST_CONTEXT.pop(triage["question"], {}))
        ps["schema_tables"] = used
        C._emit(ps, "status", message="Writing the query")
    if last_error and last_error.get("kind") == "zero_rows" and not ctx.widened:
        # Zero rows from a starved schema (lean scope sends only the tables triage named): the
        # retry also sees the topic's tables. Seen live: 'current stage of meter SC10143933' looked
        # only in unified_smart_meter_asset, and the retry -- no other table in its schema -- re-sent it.
        ctx.widened = True
        wider = PL._widened_tables(triage)
        if wider:
            t_ctx = time.perf_counter()
            ctx.base_prompt, used = PL._plan_prompt({**triage, "tables": [*triage["tables"], *wider], "source": "widened"},
                                                   catalog, (state.get("prior_turns") or [None])[-1])
            C._LAST_CONTEXT.pop(triage["question"], None)
            ctx.repair_prompt = PL.repair_base(triage["question"], ctx.base_prompt)
            C._step(ps, "context", t_ctx, outcome="zero rows -- schema widened", added=wider)
            ps["schema_tables"] = used
    prompt = ctx.base_prompt
    if last_error:
        prompt = (ctx.repair_prompt or ctx.base_prompt) + PL._repair_hint(last_error.get("sql") or "", last_error["message"], catalog)

    t0 = time.perf_counter()
    example = None if ctx.example_fallback else _verified_example_plan(triage)
    if last_error and last_error.get("kind") != "zero_rows" and example:
        res = None  # rejected or failed SQL for a verified wording: run the example, not a repair
    else:
        res = llm_gateway.chat_json("sql" if attempt == 0 else "repair", prompt, system=PL._PLAN_SYSTEM,
                                    models=settings.LLM_MODELS_SQL, max_tokens=settings.LLM_MAX_TOKENS_SQL)  # decoding per task (sql / repair) from LLM_DECODING
        if res is None and not example:
            C._step(ps, "plan", t0, attempt=attempt, outcome="no model available")
            return {"tool_call": None, "status": "busy", "attempt": attempt + 1, "step_count": _tick(state)}
    raw = res.data if res is not None else {}
    model = res.model if res is not None else "verified example"
    if example and (res is None or raw.get("answer_kind") == "unanswerable" or not PL._clean_sql(raw.get("sql"))):
        ctx.example_fallback = True
        C._step(ps, "plan", t0, attempt=attempt, outcome="verified example SQL used", example=triage["examples"][0])
        raw = example
    if str(raw.get("tool") or "") == "calculator":
        # Pure arithmetic on numbers the user gave: the calculator tool computes it exactly.
        ctx.raw_plan = raw
        C._step(ps, "plan", t0, attempt=attempt, model=model, kind="calculator")
        call = {"tool": "calculator", "args": {"expression": str(raw.get("expression") or "")},
                "attempt": attempt, "model": model}
        return {"tool_call": call, "tool_calls": [call], "attempt": attempt + 1, "validated": None,
                "last_error": None, "step_count": _tick(state)}
    sql = PL._clean_sql(raw.get("sql"))
    if (raw.get("answer_kind") == "unanswerable" or not sql) and not ctx.widened and not example \
            and triage.get("source") != "router" and (wider := PL._widened_tables(triage)):
        # 'Unanswerable' from a starved schema (lean scope = only the tables triage named): one re-plan with the
        # topic's tables. 11 Nov: 'When will Triveni reach 90% SAT at the current pace?' got the snapshot +
        # install tables but not sat_milestones (the SAT dates the pace needs) and was refused.
        C._step(ps, "plan", t0, attempt=attempt, model=model, outcome="unanswerable -- retry with topic tables")
        ctx.widened = True
        t_ctx = time.perf_counter()
        ctx.base_prompt, used = PL._plan_prompt({**triage, "tables": [*triage["tables"], *wider], "source": "widened"},
                                               catalog, (state.get("prior_turns") or [None])[-1])
        C._LAST_CONTEXT.pop(triage["question"], None)
        ctx.repair_prompt = PL.repair_base(triage["question"], ctx.base_prompt)
        C._step(ps, "context", t_ctx, outcome="unanswerable -- schema widened", added=wider)
        ps["schema_tables"] = used
        t0 = time.perf_counter()
        res = llm_gateway.chat_json("sql", ctx.base_prompt, system=PL._PLAN_SYSTEM,
                                    models=settings.LLM_MODELS_SQL, max_tokens=settings.LLM_MAX_TOKENS_SQL)
        if res is None:
            C._step(ps, "plan", t0, attempt=attempt, outcome="no model available")
            return {"tool_call": None, "status": "busy", "attempt": attempt + 1, "step_count": _tick(state)}
        raw, model = res.data, res.model
        sql = PL._clean_sql(raw.get("sql"))
    if raw.get("answer_kind") == "unanswerable" or not sql:
        C._step(ps, "plan", t0, attempt=attempt, model=model, outcome="unanswerable")
        C._finish(ps, str(raw.get("reason") or "This can't be answered from the available data."), refused=True,
                  source="planner", agent=triage["domain"])
        return {"tool_call": None, "status": "refused", "refused": True, "attempt": attempt + 1,
                "step_count": _tick(state)}
    ctx.raw_plan = raw
    C._step(ps, "plan", t0, attempt=attempt, model=model, kind=str(raw.get("answer_kind") or ""),
            reasoning=str(raw.get("reasoning") or "")[:200])
    call = {"tool": str(raw.get("tool") or "run_sql"), "args": {"sql": sql}, "attempt": attempt, "model": model}
    return {"tool_call": call, "tool_calls": [call], "attempt": attempt + 1, "validated": None,
            "last_error": None, "step_count": _tick(state)}


# --------------------------------------------------------------------------- #
# validate -- application code decides whether the request may run
# --------------------------------------------------------------------------- #

# plumbing error kinds -> the kinds the controller's repair prompt and audit use
_ERROR_KIND = {"unknown_tool": "not_allowed", "missing_tool": "not_allowed", "rejected": "validation",
               "invalid_args": "invalid_args", "parse": "parse"}


def validate_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Who controls it: application code, in this order --
    1. allowlist: the requested tool must be registered (tools/registry.py)
    2. SQL guard: one read-only SELECT on catalog tables (no INSERT/UPDATE/DELETE/DDL)
    3. entity check (once): the question's places / OEMs / years ... are in the SQL
    4. idempotency: the same normalized request already ran in this question?
         - it succeeded -> reuse its result (no second execution)
         - it failed    -> do not run it again; tell the controller it repeated itself
    A failure becomes `last_error` for the controller; nothing is raised."""
    ctx = context.get(config)
    ps = ctx.pipeline_state
    triage, call = state["triage"], state["tool_call"] or {}
    t0 = time.perf_counter()
    C._emit(ps, "status", message="Validating the query")
    if call.get("tool") == "calculator":
        prepared = calling.prepare({"tool": "calculator", "args": call.get("args")}, registry.TOOLS)
        if isinstance(prepared, calling.ToolOutcome):
            err = {"kind": _ERROR_KIND.get(prepared.payload["error_kind"], "invalid_args"), "sql": "",
                   "message": prepared.payload["error"]}
            C._step(ps, "validate", t0, outcome="rejected", tool="calculator", reason=err["message"][:160])
            return {"last_error": err, "errors": [err], "step_count": _tick(state)}
        ctx.prepared = prepared
        expr = " ".join(prepared.args.expression.split())
        key = idempotency.request_key("calculator", {"expression": expr}, {})
        C._step(ps, "validate", t0, outcome="ok", tool="calculator")
        return {"last_error": None, "step_count": _tick(state),
                "validated": {"key": key, "tool": "calculator", "sql": f"calculator: {expr}", "reuse": False}}
    sql = str((call.get("args") or {}).get("sql") or "") if isinstance(call.get("args"), dict) else ""

    # Plumbing (tools/calling.py): parse -> allowlist lookup -> Pydantic args -> the tool's
    # own validator (run_sql: the SQL guard). Any failure comes back as a structured result.
    catalog = get_catalog("all" if triage.get("cross_domain") else triage["domain"])
    ctx.tool_ctx = registry.SqlToolContext(catalog=catalog)
    prepared = calling.prepare({"tool": call.get("tool"), "args": call.get("args")}, registry.TOOLS, ctx.tool_ctx)
    if isinstance(prepared, calling.ToolOutcome):
        kind = _ERROR_KIND.get(prepared.payload["error_kind"], prepared.payload["error_kind"])
        err = {"kind": kind, "sql": sql, "message": prepared.payload["error"]}
        C._step(ps, "validate", t0, outcome="not_allowed" if kind == "not_allowed" else "rejected",
                reason=err["message"][:160])
        return {"last_error": err, "errors": [err], "step_count": _tick(state)}
    tool = registry.get(prepared.tool)
    verdict = ctx.tool_ctx.verdict
    ctx.prepared = prepared
    ctx.plan = ctx.tool_ctx.plan = PL._to_plan(ctx.raw_plan, verdict["guard"], catalog, call.get("model") or "")

    update: dict[str, Any] = {"step_count": _tick(state)}
    # Every repaired SQL is checked again -- seen live 27 Oct: a repair fixed the flagged table but added a stage
    # filter nobody asked for, and ran unchecked. The same problems twice in a row run anyway (the repair could
    # not fix them; refusing would lose the answer).
    if settings.VERIFY_ENTITIES and not limits.retries_exhausted(state["attempt"]):
        extra_sql = [str(e.get("sql") or "") for e in ctx.raw_plan.get("extra_queries") or [] if isinstance(e, dict)]
        mismatch = sql_validator.check_entities(triage["question"], [ctx.plan["sql"], *extra_sql],
                                                columns=PL._table_columns(ctx.plan["tables"], catalog),
                                                asked=state.get("question"))
        update["entities_checked"] = True
        before = [e for e in state.get("errors") or [] if e.get("kind") == "entity_mismatch"]
        if mismatch and before and set(mismatch["problems"]) <= set(before[-1].get("problems") or []):
            C._step(ps, "validate", t0, outcome="entity_mismatch_kept", problems=mismatch["problems"][:3])
            mismatch = None
        if mismatch:
            err = {**mismatch, "sql": ctx.plan["sql"]}
            C._step(ps, "validate", t0, outcome="entity_mismatch", problems=mismatch["problems"][:3])
            return {**update, "last_error": err, "errors": [err]}

    key = idempotency.request_key(tool.name, {"sql": verdict["normalized_sql"]}, {"domain": triage["domain"]})
    seen = (state.get("executed") or {}).get(key)
    if seen and not seen.get("ok"):
        err = {"kind": "duplicate", "sql": ctx.plan["sql"],
               "message": f"You produced the same SQL as an earlier attempt, which failed: {seen.get('error')}. "
                          "Write a DIFFERENT query that fixes that problem."}
        C._step(ps, "validate", t0, outcome="duplicate_of_failed")
        return {**update, "last_error": err, "errors": [err]}
    C._step(ps, "validate", t0, outcome="ok", reuse=bool(seen and seen.get("ok") and tool.idempotent),
            tables=ctx.plan["tables"])
    return {**update, "last_error": None,
            "validated": {"key": key, "sql": ctx.plan["sql"], "engine": verdict["data_engine"],
                          "reuse": bool(seen and seen.get("ok") and tool.idempotent)}}


# --------------------------------------------------------------------------- #
# execute -- application code runs the approved request; the database answers
# --------------------------------------------------------------------------- #

def execute_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Who controls it: application code calls the executor (tools/sql_executor.py);
    the database executes. Only reached with a `validated` request. Outcomes:
      ok rows        -> has_result (controller will then decide: final answer)
      DB error       -> structured error -> controller retries (bounded)
      zero rows      -> every filter value a stored value: accepted (zero is the answer);
                        else once: sent back as "check your filter values"; kept as fallback
      extra failed   -> main result kept as fallback; controller asked to fix the extra
    Every execution is recorded under its idempotency key."""
    ctx = context.get(config)
    ps = ctx.pipeline_state
    triage, validated = state["triage"], state["validated"]
    key = validated["key"]
    t0 = time.perf_counter()
    if validated.get("tool") == "calculator":
        return _execute_calculator(state, ctx, ps, key, t0)
    C._emit(ps, "status", message="Running the query")
    db_queries = sum(1 for v in (state.get("executed") or {}).values() if v.get("db"))
    if settings.QUESTION_MAX_DB_QUERIES and db_queries >= settings.QUESTION_MAX_DB_QUERIES and not validated.get("reuse"):
        err = {"kind": "db_query_limit", "sql": ctx.plan["sql"],
               "message": f"Database query limit reached ({db_queries} queries this question)."}
        C._step(ps, "execute", t0, outcome="db_query_limit")
        return {"step_count": _tick(state), "last_error": err, "errors": [err]}

    if validated.get("reuse") and key in ctx.results_by_key:
        result, failure = ctx.results_by_key[key], None
        C._step(ps, "execute", t0, reused=True)
    else:
        # Plumbing executes the prepared call; exceptions become a structured failure.
        outcome = calling.execute(ctx.prepared, registry.TOOLS, ctx.tool_ctx)
        result = outcome.raw if outcome.success else None
        failure = None if outcome.success else outcome.payload
        C._step(ps, "execute", t0, rows=len((result or {}).get("result", [])),
                error=(failure or {}).get("error", "")[:160] or None)
    base = {"step_count": _tick(state)}

    if failure:
        kind = failure["error_kind"] if failure["error_kind"] in ("connectivity", "database_error") else "tool_error"
        err = {"kind": kind, "message": failure["error"], "sql": ctx.plan["sql"]}
        return {**base, "last_error": err, "errors": [err],
                "executed": {key: {"ok": False, "db": True, "error": err["message"][:300]}}}

    ctx.results_by_key[key] = result
    record = {"executed": {key: {"ok": True, "db": True, "rows": len(result.get("result", []))}}}
    zero = PL._is_empty(result.get("result"), ctx.plan) and not state.get("zero_rows_retried")
    if zero and PL._zero_is_genuine(ctx.plan["sql"]):
        # Every filter value is a real stored value: zero IS the answer -- no repair round
        # (it would only send the same SQL back, at a full SQL prompt's cost).
        C._step(ps, "zero_rows", time.perf_counter(), outcome="kept -- every filter value is a stored value",
                values=PL._filter_literals(ctx.plan["sql"])[:6])
        zero = False
    if zero and not limits.retries_exhausted(state["attempt"]):
        ctx.fallback = {"plan": ctx.plan, "result": result, "extras": []}
        err = {"kind": "zero_rows", "sql": ctx.plan["sql"],
               "message": "The query ran but matched ZERO rows. Check filter values against the real values listed in "
                          "SCHEMA (case, spelling, which column holds the value), the date range, and whether "
                          "another table in SCHEMA holds this value. If zero is genuinely correct, return the same SQL."}
        # A repeat of this SQL after "zero is correct" must be accepted, so it is not a failed key.
        return {**base, **record, "zero_rows_retried": True, "has_fallback": True, "last_error": err, "errors": [err]}

    impossible = [] if state.get("implausible_retried") else PL._impossible_shares(result.get("result"))
    if impossible and not limits.retries_exhausted(state["attempt"]):
        ctx.fallback = {"plan": ctx.plan, "result": result, "extras": []}
        C._step(ps, "implausible", time.perf_counter(), outcome="share above 100% -- sent back", values=impossible[:3])
        err = {"kind": "implausible", "sql": ctx.plan["sql"],
               "message": f"The query ran, but {'; '.join(impossible[:3])}: a share / rate of a whole cannot exceed "
                          "100%, so its numerator and denominator cover different scopes (e.g. all-time SAT over one "
                          "month's installs, or a JOIN that multiplied rows). Compute both from the same rows, period "
                          "and place, as the rules and verified examples define the KPI."}
        return {**base, **record, "implausible_retried": True, "has_fallback": True, "last_error": err, "errors": [err]}

    empty = [] if state.get("implausible_retried") else PL._empty_periods(result.get("result"), ctx.plan.get("tables"))
    if empty and not limits.retries_exhausted(state["attempt"]):
        ctx.fallback = {"plan": ctx.plan, "result": result, "extras": []}
        C._step(ps, "implausible", time.perf_counter(), outcome="empty period row -- sent back", values=empty[:3])
        err = {"kind": "implausible", "sql": ctx.plan["sql"],
               "message": f"The query ran, but the row(s) for {', '.join(empty)} are all zero / empty although the table "
                          "holds data for that period: the period filter dropped them (e.g. months taken from another "
                          "table's list). Filter this table's own period column directly, e.g. data_date IN "
                          "(DATE '2026-06-01', DATE '2026-07-01') for the months the question names."}
        return {**base, **record, "implausible_retried": True, "has_fallback": True, "last_error": err, "errors": [err]}

    doubled = [] if state.get("implausible_retried") else PL._double_counted(result.get("result"))
    if doubled and not limits.retries_exhausted(state["attempt"]):
        ctx.fallback = {"plan": ctx.plan, "result": result, "extras": []}
        C._step(ps, "implausible", time.perf_counter(), outcome="double count -- sent back", values=doubled[:1])
        err = {"kind": "implausible", "sql": ctx.plan["sql"],
               "message": f"The query ran, but {doubled[0]}: the same items were counted twice (a UNION of the same "
                          "table, or of two sources holding the same meters). Count each item once, as the rules and "
                          "verified examples define the KPI; do not add a measure to itself."}
        return {**base, **record, "implausible_retried": True, "has_fallback": True, "last_error": err, "errors": [err]}

    extras, extra_problem = PL._run_extras(ps, ctx.raw_plan, get_catalog("all" if triage.get("cross_domain") else triage["domain"]), ctx.plan.get("planner") or "")
    if extra_problem and not limits.retries_exhausted(state["attempt"]):
        ctx.fallback = {"plan": ctx.plan, "result": result, "extras": extras}
        err = {"kind": "extra_query", "sql": ctx.plan["sql"], "message": f"The main query worked, but {extra_problem}"}
        return {**base, **record, "has_fallback": True, "last_error": err, "errors": [err]}
    ctx.result, ctx.extras = result, extras
    return {**base, **record, "has_result": True, "last_error": None}


def _execute_calculator(state: AgentState, ctx, ps, key: str, t0: float) -> dict[str, Any]:
    """Run the calculator through the plumbing layer; its result becomes a one-row
    table so the normal answer path (cards, number check, follow-ups) applies."""
    import pandas as pd

    C._emit(ps, "status", message="Calculating")
    outcome = calling.execute(ctx.prepared, registry.TOOLS)
    base = {"step_count": _tick(state)}
    if not outcome.success:
        err = {"kind": "tool_error", "sql": "", "message": outcome.payload["error"]}
        C._step(ps, "execute", t0, tool="calculator", error=err["message"][:160])
        return {**base, "last_error": err, "errors": [err], "executed": {key: {"ok": False, "error": err["message"]}}}
    expr, value = outcome.raw["expression"], outcome.raw["result"]
    C._step(ps, "execute", t0, tool="calculator", rows=1)
    ctx.result = {"result": pd.DataFrame([{"expression": expr, "result": value}]), "group_count": 1}
    ctx.extras = []
    ctx.plan = {"engine": "calculator", "data_engine": None, "answer_kind": "number", "sql": f"calculator: {expr}",
                "database": None, "tables": [], "labels": {"result": "Result", "expression": "Calculation"},
                "display": {"cards": ["result"]}, "metrics": [{"alias": "result", "label": "Result"}],
                "group_by": [], "headline_alias": "result", "filters": [], "warnings": [],
                "planner": (state.get("tool_call") or {}).get("model") or "", "table_title": "Calculation",
                "reasoning": str(ctx.raw_plan.get("reasoning") or "")[:600], "skip_scope_count": True}
    return {**base, "has_result": True, "last_error": None, "executed": {key: {"ok": True, "rows": 1}}}
