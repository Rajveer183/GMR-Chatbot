"""Graph nodes. Each node does one job and returns only the state keys it changes;
LangGraph merges the update and picks the next node via graph/routing.py.

    node              who decides / acts
    ----------------  ------------------------------------------------------------
    understand_node   LLM (triage) -- or the no-LLM router for a verified wording
    controller_node   LLM writes the tool request (SQL); code decides "done" once a
                      validated result exists. Never executes anything.
    validate_node     application code: allowlist, SQL guard, entity check, idempotency
    execute_node      application code calls the executor; the database runs the SQL
    answer_node       LLM writes the summary from the rows; code checks its numbers
    reply_node        returns the triage reply (chitchat / out of scope)
    explain_node      LLM explains a KPI from the rules + glossary (no SQL)
    give_up_node      code: retries exhausted -> best kept result or honest refusal
    step_limit_node   code: AGENT_MAX_STEPS reached -> terminate safely

Prompt building, retrieval of long-term memory (verified examples, rules, schema
hints, past questions) and answer writing reuse gmr_agents/llm_pipeline.py, so the
graph changes the orchestration, not what the LLM is shown.
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.runnables import RunnableConfig

from gmr_agents import language
from gmr_agents import llm_pipeline as P
from gmr_agents.graph import context
from gmr_agents.graph.state import AgentState
from gmr_agents.guards import idempotency, limits, sql_validator
from gmr_agents.schema import business_rules
from gmr_agents.schema.catalog import get_catalog
from gmr_agents.tools import calling, registry
from shared import config as settings
from shared import llm_gateway


def _tick(state: AgentState) -> int:
    return state.get("step_count", 0) + 1


# --------------------------------------------------------------------------- #
# understand -- controller, first decision: what kind of request is this?
# --------------------------------------------------------------------------- #

def understand_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Who controls it: the LLM (triage call) decides kind / topics / tables /
    resolved follow-up question; a near-exact wording of a verified example is
    routed by code without an LLM call (P._route)."""
    ctx = context.get(config)
    ps = ctx.pipeline_state
    prior = state.get("prior_turns") or []
    P._emit(ps, "status", message="Understanding your question")
    ps["memory_turns"] = prior
    triage = P._route(ps, prior)
    if triage is None:
        triage = P._triage(ps, prior)
    if triage is None:
        P._finish(ps, P._BUSY, refused=True, source="none (AI busy)")
        return {"route": "busy", "status": "busy", "refused": True, "step_count": _tick(state)}
    P.fix_question(ps, triage, prior)  # place scope + the user's own category words, by code
    # Response language by code (gmr_agents/language.py): explicit request > dropdown > the
    # message's own language. Routing, SQL and checks only ever see the English question.
    triage["language"], ps["detected_language"] = language.resolve(
        state["question"], ps.get("language_pref") or language.AUTO, triage.get("language"), prior)
    ps["target_language"] = triage["language"]
    ps["triage"] = triage
    ps["routing"] = {"domain": triage["domain"], "topics": triage["topics"], "tables": triage["tables"],
                     "source": triage["source"]}
    if triage["question"] and triage["question"] != state["question"]:
        ps["resolved_question"] = triage["question"]
    if triage["followup"]:
        ps["followup"] = "llm"
    route = triage["kind"] if triage["kind"] in ("explain", "chitchat", "out_of_scope") else "data"
    return {"route": route, "triage": triage,
            "attempt": 0, "status": "running", "step_count": _tick(state)}


def reply_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Who controls it: application code returns the reply the triage LLM wrote."""
    triage = state["triage"]
    ps = context.get(config).pipeline_state
    P._finish(ps, triage["reply"] or "How can I help with GMR's meter programme data?",
              refused=triage["kind"] == "out_of_scope", source=triage["source"])
    # The triage LLM was asked for REPLY LANGUAGE (explicit / dropdown) or the user's own language,
    # but does not always comply: record what the reply really is in. A mismatch -- and the canned
    # English fallback -- is translated afterwards (main._in_language).
    ps["answer_language"] = language.written_in(triage["reply"], triage["language"]) if triage["reply"] else "en"
    return {"status": "refused" if triage["kind"] == "out_of_scope" else "answered", "step_count": _tick(state)}


def explain_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Who controls it: the LLM explains from the rules + glossary; no tool runs."""
    ps = context.get(config).pipeline_state
    triage = state["triage"]
    written = P._explain(ps, triage)
    P._finish(ps, written["summary"], refused=written["refused"], source=written["source"],
              followups=written["followups"], insights=written["insights"], agent=triage["domain"])
    return {"status": "refused" if written["refused"] else "answered", "step_count": _tick(state)}


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
    catalog = get_catalog(triage["domain"])
    if not ctx.base_prompt:
        t_ctx = time.perf_counter()
        ctx.base_prompt, used = P._plan_prompt(triage, catalog, (state.get("prior_turns") or [None])[-1])
        ctx.repair_prompt = P.repair_base(triage["question"], ctx.base_prompt)
        P._step(ps, "context", t_ctx, **P._LAST_CONTEXT.pop(triage["question"], {}))
        ps["schema_tables"] = used
        P._emit(ps, "status", message="Writing the query")
    last_error = state.get("last_error")
    if last_error and last_error.get("kind") == "zero_rows" and not ctx.widened:
        # Zero rows from a starved schema (lean scope sends only the tables triage named): the
        # retry also sees the topic's tables. Seen live: 'current stage of meter SC10143933' looked
        # only in unified_smart_meter_asset, and the retry -- no other table in its schema -- re-sent it.
        ctx.widened = True
        wider = P._widened_tables(triage)
        if wider:
            t_ctx = time.perf_counter()
            ctx.base_prompt, used = P._plan_prompt({**triage, "tables": [*triage["tables"], *wider], "source": "widened"},
                                                   catalog, (state.get("prior_turns") or [None])[-1])
            P._LAST_CONTEXT.pop(triage["question"], None)
            ctx.repair_prompt = P.repair_base(triage["question"], ctx.base_prompt)
            P._step(ps, "context", t_ctx, outcome="zero rows -- schema widened", added=wider)
            ps["schema_tables"] = used
    prompt = ctx.base_prompt
    if last_error:
        prompt = (ctx.repair_prompt or ctx.base_prompt) + P._repair_hint(last_error.get("sql") or "", last_error["message"], catalog)

    t0 = time.perf_counter()
    example = None if ctx.example_fallback else _verified_example_plan(triage)
    if last_error and last_error.get("kind") != "zero_rows" and example:
        res = None  # rejected or failed SQL for a verified wording: run the example, not a repair
    else:
        res = llm_gateway.chat_json("sql" if attempt == 0 else "repair", prompt, system=P._PLAN_SYSTEM,
                                    models=settings.LLM_MODELS_SQL, max_tokens=settings.LLM_MAX_TOKENS_SQL)  # decoding per task (sql / repair) from LLM_DECODING
        if res is None and not example:
            P._step(ps, "plan", t0, attempt=attempt, outcome="no model available")
            return {"tool_call": None, "status": "busy", "attempt": attempt + 1, "step_count": _tick(state)}
    raw = res.data if res is not None else {}
    model = res.model if res is not None else "verified example"
    if example and (res is None or raw.get("answer_kind") == "unanswerable" or not P._clean_sql(raw.get("sql"))):
        ctx.example_fallback = True
        P._step(ps, "plan", t0, attempt=attempt, outcome="verified example SQL used", example=triage["examples"][0])
        raw = example
    if str(raw.get("tool") or "") == "calculator":
        # Pure arithmetic on numbers the user gave: the calculator tool computes it exactly.
        ctx.raw_plan = raw
        P._step(ps, "plan", t0, attempt=attempt, model=model, kind="calculator")
        call = {"tool": "calculator", "args": {"expression": str(raw.get("expression") or "")},
                "attempt": attempt, "model": model}
        return {"tool_call": call, "tool_calls": [call], "attempt": attempt + 1, "validated": None,
                "last_error": None, "step_count": _tick(state)}
    sql = P._clean_sql(raw.get("sql"))
    if raw.get("answer_kind") == "unanswerable" or not sql:
        P._step(ps, "plan", t0, attempt=attempt, model=model, outcome="unanswerable")
        P._finish(ps, str(raw.get("reason") or "This can't be answered from the available data."), refused=True,
                  source="planner", agent=triage["domain"])
        return {"tool_call": None, "status": "refused", "refused": True, "attempt": attempt + 1,
                "step_count": _tick(state)}
    ctx.raw_plan = raw
    P._step(ps, "plan", t0, attempt=attempt, model=model, kind=str(raw.get("answer_kind") or ""),
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
    P._emit(ps, "status", message="Validating the query")
    if call.get("tool") == "calculator":
        prepared = calling.prepare({"tool": "calculator", "args": call.get("args")}, registry.TOOLS)
        if isinstance(prepared, calling.ToolOutcome):
            err = {"kind": _ERROR_KIND.get(prepared.payload["error_kind"], "invalid_args"), "sql": "",
                   "message": prepared.payload["error"]}
            P._step(ps, "validate", t0, outcome="rejected", tool="calculator", reason=err["message"][:160])
            return {"last_error": err, "errors": [err], "step_count": _tick(state)}
        ctx.prepared = prepared
        expr = " ".join(prepared.args.expression.split())
        key = idempotency.request_key("calculator", {"expression": expr}, {})
        P._step(ps, "validate", t0, outcome="ok", tool="calculator")
        return {"last_error": None, "step_count": _tick(state),
                "validated": {"key": key, "tool": "calculator", "sql": f"calculator: {expr}", "reuse": False}}
    sql = str((call.get("args") or {}).get("sql") or "") if isinstance(call.get("args"), dict) else ""

    # Plumbing (tools/calling.py): parse -> allowlist lookup -> Pydantic args -> the tool's
    # own validator (run_sql: the SQL guard). Any failure comes back as a structured result.
    catalog = get_catalog(triage["domain"])
    ctx.tool_ctx = registry.SqlToolContext(catalog=catalog)
    prepared = calling.prepare({"tool": call.get("tool"), "args": call.get("args")}, registry.TOOLS, ctx.tool_ctx)
    if isinstance(prepared, calling.ToolOutcome):
        kind = _ERROR_KIND.get(prepared.payload["error_kind"], prepared.payload["error_kind"])
        err = {"kind": kind, "sql": sql, "message": prepared.payload["error"]}
        P._step(ps, "validate", t0, outcome="not_allowed" if kind == "not_allowed" else "rejected",
                reason=err["message"][:160])
        return {"last_error": err, "errors": [err], "step_count": _tick(state)}
    tool = registry.get(prepared.tool)
    verdict = ctx.tool_ctx.verdict
    ctx.prepared = prepared
    ctx.plan = ctx.tool_ctx.plan = P._to_plan(ctx.raw_plan, verdict["guard"], catalog, call.get("model") or "")

    update: dict[str, Any] = {"step_count": _tick(state)}
    # Every repaired SQL is checked again -- seen live 27 Oct: a repair fixed the flagged table but added a stage
    # filter nobody asked for, and ran unchecked. The same problems twice in a row run anyway (the repair could
    # not fix them; refusing would lose the answer).
    if settings.VERIFY_ENTITIES and not limits.retries_exhausted(state["attempt"]):
        extra_sql = [str(e.get("sql") or "") for e in ctx.raw_plan.get("extra_queries") or [] if isinstance(e, dict)]
        mismatch = sql_validator.check_entities(triage["question"], [ctx.plan["sql"], *extra_sql],
                                                columns=P._table_columns(ctx.plan["tables"], catalog),
                                                asked=state.get("question"))
        update["entities_checked"] = True
        before = [e for e in state.get("errors") or [] if e.get("kind") == "entity_mismatch"]
        if mismatch and before and set(mismatch["problems"]) <= set(before[-1].get("problems") or []):
            P._step(ps, "validate", t0, outcome="entity_mismatch_kept", problems=mismatch["problems"][:3])
            mismatch = None
        if mismatch:
            err = {**mismatch, "sql": ctx.plan["sql"]}
            P._step(ps, "validate", t0, outcome="entity_mismatch", problems=mismatch["problems"][:3])
            return {**update, "last_error": err, "errors": [err]}

    key = idempotency.request_key(tool.name, {"sql": verdict["normalized_sql"]}, {"domain": triage["domain"]})
    seen = (state.get("executed") or {}).get(key)
    if seen and not seen.get("ok"):
        err = {"kind": "duplicate", "sql": ctx.plan["sql"],
               "message": f"You produced the same SQL as an earlier attempt, which failed: {seen.get('error')}. "
                          "Write a DIFFERENT query that fixes that problem."}
        P._step(ps, "validate", t0, outcome="duplicate_of_failed")
        return {**update, "last_error": err, "errors": [err]}
    P._step(ps, "validate", t0, outcome="ok", reuse=bool(seen and seen.get("ok") and tool.idempotent),
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
    P._emit(ps, "status", message="Running the query")
    db_queries = sum(1 for v in (state.get("executed") or {}).values() if v.get("db"))
    if settings.QUESTION_MAX_DB_QUERIES and db_queries >= settings.QUESTION_MAX_DB_QUERIES and not validated.get("reuse"):
        err = {"kind": "db_query_limit", "sql": ctx.plan["sql"],
               "message": f"Database query limit reached ({db_queries} queries this question)."}
        P._step(ps, "execute", t0, outcome="db_query_limit")
        return {"step_count": _tick(state), "last_error": err, "errors": [err]}

    if validated.get("reuse") and key in ctx.results_by_key:
        result, failure = ctx.results_by_key[key], None
        P._step(ps, "execute", t0, reused=True)
    else:
        # Plumbing executes the prepared call; exceptions become a structured failure.
        outcome = calling.execute(ctx.prepared, registry.TOOLS, ctx.tool_ctx)
        result = outcome.raw if outcome.success else None
        failure = None if outcome.success else outcome.payload
        P._step(ps, "execute", t0, rows=len((result or {}).get("result", [])),
                error=(failure or {}).get("error", "")[:160] or None)
    base = {"step_count": _tick(state)}

    if failure:
        kind = failure["error_kind"] if failure["error_kind"] in ("connectivity", "database_error") else "tool_error"
        err = {"kind": kind, "message": failure["error"], "sql": ctx.plan["sql"]}
        return {**base, "last_error": err, "errors": [err],
                "executed": {key: {"ok": False, "db": True, "error": err["message"][:300]}}}

    ctx.results_by_key[key] = result
    record = {"executed": {key: {"ok": True, "db": True, "rows": len(result.get("result", []))}}}
    zero = P._is_empty(result.get("result"), ctx.plan) and not state.get("zero_rows_retried")
    if zero and P._zero_is_genuine(ctx.plan["sql"]):
        # Every filter value is a real stored value: zero IS the answer -- no repair round
        # (it would only send the same SQL back, at a full SQL prompt's cost).
        P._step(ps, "zero_rows", time.perf_counter(), outcome="kept -- every filter value is a stored value",
                values=P._filter_literals(ctx.plan["sql"])[:6])
        zero = False
    if zero and not limits.retries_exhausted(state["attempt"]):
        ctx.fallback = {"plan": ctx.plan, "result": result, "extras": []}
        err = {"kind": "zero_rows", "sql": ctx.plan["sql"],
               "message": "The query ran but matched ZERO rows. Check filter values against the real values listed in "
                          "SCHEMA (case, spelling, which column holds the value), the date range, and whether "
                          "another table in SCHEMA holds this value. If zero is genuinely correct, return the same SQL."}
        # A repeat of this SQL after "zero is correct" must be accepted, so it is not a failed key.
        return {**base, **record, "zero_rows_retried": True, "has_fallback": True, "last_error": err, "errors": [err]}

    extras, extra_problem = P._run_extras(ps, ctx.raw_plan, get_catalog(triage["domain"]), ctx.plan.get("planner") or "")
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

    P._emit(ps, "status", message="Calculating")
    outcome = calling.execute(ctx.prepared, registry.TOOLS)
    base = {"step_count": _tick(state)}
    if not outcome.success:
        err = {"kind": "tool_error", "sql": "", "message": outcome.payload["error"]}
        P._step(ps, "execute", t0, tool="calculator", error=err["message"][:160])
        return {**base, "last_error": err, "errors": [err], "executed": {key: {"ok": False, "error": err["message"]}}}
    expr, value = outcome.raw["expression"], outcome.raw["result"]
    P._step(ps, "execute", t0, tool="calculator", rows=1)
    ctx.result = {"result": pd.DataFrame([{"expression": expr, "result": value}]), "group_count": 1}
    ctx.extras = []
    ctx.plan = {"engine": "calculator", "data_engine": None, "answer_kind": "number", "sql": f"calculator: {expr}",
                "database": None, "tables": [], "labels": {"result": "Result", "expression": "Calculation"},
                "display": {"cards": ["result"]}, "metrics": [{"alias": "result", "label": "Result"}],
                "group_by": [], "headline_alias": "result", "filters": [], "warnings": [],
                "planner": (state.get("tool_call") or {}).get("model") or "", "table_title": "Calculation",
                "reasoning": str(ctx.raw_plan.get("reasoning") or "")[:600], "skip_scope_count": True}
    return {**base, "has_result": True, "last_error": None, "executed": {key: {"ok": True, "rows": 1}}}


# --------------------------------------------------------------------------- #
# answer -- the LLM interprets the rows; code checks the numbers it wrote
# --------------------------------------------------------------------------- #

def answer_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Who controls it: application code builds the KPI cards / chart / table from
    the rows (exactly what the planner asked for); the LLM writes the summary,
    insights and picks follow-ups from verified questions; the number check
    (validation/number_check.py) sends wrong figures back once and repairs the rest."""
    ctx = context.get(config)
    ps = ctx.pipeline_state
    triage = state["triage"]
    if ctx.result is None and ctx.fallback:
        ctx.plan, ctx.result, ctx.extras = ctx.fallback["plan"], ctx.fallback["result"], ctx.fallback["extras"]
    plan, result, extras = ctx.plan, ctx.result, ctx.extras
    plan["extra_sql"] = [ex["sql"] for ex in extras]
    # A chart type the user named outranks the dashboard's chart. Both wordings: the rewritten
    # question can drop it -- seen live, 'Give me a pie chart of defective meters by defect type'
    # became 'Show the count of defective meters by defect type' and was drawn as bars.
    plan["question"] = f"{state['question']} || {triage['question']}"
    ps.update(plan=plan, result=result, agent=triage["domain"], domain=triage["domain"],
              row_count=len(result.get("result", [])))
    t0 = time.perf_counter()
    blocks = P._blocks(plan, result)
    for i, ex in enumerate(extras, 2):
        blocks["tables"].extend(P._blocks({**plan, "metrics": [], "display": {"table": True}, "labels": {},
                                           "table_title": ex["purpose"][:80]}, ex)["tables"])
        if blocks["tables"]:
            blocks["tables"][-1]["id"] = f"extra_table_{i}"
    P._step(ps, "present", t0, kpi=len(blocks["kpis"]), chart=len(blocks["charts"]), table=len(blocks["tables"]))
    P._emit(ps, "partial", kpis=blocks["kpis"], charts=blocks["charts"], tables=blocks["tables"],
            resolved_question=ps.get("resolved_question"))
    written = P._write_answer(ps, triage, plan, result, extras)
    ps.update(kpis=blocks["kpis"], charts=blocks["charts"], tables=blocks["tables"], summary=written["summary"],
              insights=written["insights"], followups=written["followups"], refused=False,
              narration_source=written["source"])
    return {"status": "answered", "summary": written["summary"], "insights": written["insights"],
            "followups": written["followups"], "refused": False, "step_count": _tick(state)}


# --------------------------------------------------------------------------- #
# termination
# --------------------------------------------------------------------------- #

def give_up_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Who controls it: application code. Retries are exhausted (or the AI is busy,
    or the database is unreachable): answer from a kept zero-row / partial result
    when there is one, otherwise refuse honestly with the last structured error."""
    ctx = context.get(config)
    if ctx.fallback:
        return {"has_result": True, "step_count": _tick(state)}
    err = state.get("last_error") or {}
    if llm_gateway.limit_hit():
        message = ("This question needed more processing than one answer is allowed -- please ask a narrower "
                   "question (one KPI, one project or one period).")
    elif state.get("status") == "busy":
        message = P._BUSY
    elif err.get("kind") in ("connectivity", "db_query_limit"):
        message = err["message"] if err["kind"] == "connectivity" else (
            "This question needed more database queries than one answer is allowed -- please narrow it down.")
    else:
        message = f"I couldn't build a working query for this ({str(err.get('message') or 'no usable query')[:300]})."
    P._finish(ctx.pipeline_state, message, refused=True, source="planner", agent=state["triage"]["domain"])
    return {"status": "refused" if state.get("status") != "busy" else "busy", "refused": True,
            "step_count": _tick(state)}


def step_limit_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Who controls it: application code. AGENT_MAX_STEPS was reached: stop the loop
    no matter what the controller wants next; answer from a kept result if any."""
    ctx = context.get(config)
    if ctx.fallback or ctx.result is not None:
        return {"has_result": True, "status": "step_limit"}
    P._finish(ctx.pipeline_state, "I stopped after too many attempts without a reliable result -- please rephrase "
              "or narrow the question.", refused=True, source="step_limit", agent=state["triage"]["domain"])
    return {"status": "step_limit", "refused": True}
