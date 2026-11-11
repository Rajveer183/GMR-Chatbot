"""Termination nodes (no LLM): retries or steps exhausted -> best kept result or an honest refusal.

Moved from gmr_agents/graph/nodes.py unchanged (llm_pipeline helpers are now called on
their own modules).
"""

from __future__ import annotations
from typing import Any
from langchain_core.runnables import RunnableConfig
from gmr_agents.supervisor import context
from gmr_agents.supervisor.state import AgentState
from shared import llm_gateway
from gmr_agents.agents import common as C
from gmr_agents.supervisor.steps import _tick


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
        message = C._BUSY
    elif err.get("kind") in ("connectivity", "db_query_limit"):
        message = err["message"] if err["kind"] == "connectivity" else (
            "This question needed more database queries than one answer is allowed -- please narrow it down.")
    else:
        # Plain words only: the guard's reason ('mixes postgres and athena tables -- put the athena part in
        # extra_queries', 'column sat_date could not be resolved') was shown to users as the answer on 11 Nov.
        # The reason stays in the trace (validate steps) and the log.
        C.log.info("give up: last error %s", str(err.get("message") or "")[:300])
        message = ("I couldn't get a reliable figure for this from the data. Try asking for one KPI with a place or "
                   "period -- for example 'SAT % by project' or 'installations by zone in October'.")
    C._finish(ctx.pipeline_state, message, refused=True, source="planner", agent=state["triage"]["domain"])
    return {"status": "refused" if state.get("status") != "busy" else "busy", "refused": True,
            "step_count": _tick(state)}


def step_limit_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Who controls it: application code. AGENT_MAX_STEPS was reached: stop the loop
    no matter what the controller wants next; answer from a kept result if any."""
    ctx = context.get(config)
    if ctx.fallback or ctx.result is not None:
        return {"has_result": True, "status": "step_limit"}
    C._finish(ctx.pipeline_state, "I stopped after too many attempts without a reliable result -- please rephrase "
              "or narrow the question.", refused=True, source="step_limit", agent=state["triage"]["domain"])
    return {"status": "step_limit", "refused": True}
