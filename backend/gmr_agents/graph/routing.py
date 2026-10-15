"""Conditional edges -- which node runs next.

Who controls it: LangGraph calls these after a node finishes; they are plain
application code reading the state. The LLM influences the path only through
what it put in the state (a tool_call or not); it cannot jump to a node, skip
validation or extend a loop. The step guard is checked first on every edge out
of the loop, so termination never depends on the LLM.

    START -> understand -+-> controller --(tool_call)--> validate --ok--> execute --> controller --(no tool)--> answer -> END
                         |        ^                          |                |
                         |        +------ retry (error) -----+----------------+
                         |        |  retries exhausted / busy -> give_up -> answer (fallback) | END
                         |        +  steps exhausted ---------> step_limit -> answer (kept result) | END
                         +-> explain -> END     +-> reply -> END     +-> (busy) END
"""

from __future__ import annotations

import re
from typing import Literal

from gmr_agents.graph.state import AgentState
from gmr_agents.guards import limits


def after_understand(state: AgentState) -> Literal["controller", "explain", "reply", "__end__"]:
    route = state.get("route")
    if route == "data":
        return "controller"
    if route == "explain":
        return "explain"
    if route in ("chitchat", "out_of_scope"):
        return "reply"
    return "__end__"  # busy: understand_node already wrote the "AI busy" answer


def after_controller(state: AgentState) -> Literal["validate", "answer", "give_up", "step_limit", "__end__"]:
    if limits.steps_exhausted(state.get("step_count", 0)):
        return "step_limit"
    if state.get("tool_call"):
        return "validate"                      # LLM requested a tool -> it must pass validation first
    if state.get("has_result"):
        return "answer"                        # no tool needed: a validated result is ready
    if state.get("status") == "busy":
        return "give_up"                       # no model answered: fall back or refuse
    return "__end__"                           # refused / unanswerable: answer already written


def _error_fingerprint(err: dict) -> str:
    """kind + message with the SQL-specific details (quoted names, numbers) masked."""
    # Only numbers (line / column positions, counts) are masked: 'column "x" does not exist' twice
    # is a loop, two different parse errors are not.
    msg = re.sub(r"\d+", "_", str(err.get("message") or "").lower())
    return f"{err.get('kind')}|{' '.join(msg.split())[:200]}"


def repeated_error(state: AgentState) -> bool:
    """Loop guard: the last two failed attempts hit the same error (even with different
    SQL). Errors that a repeat is meant to resolve (zero rows -> 'return the same SQL if
    zero is right') are excluded."""
    errs = [e for e in state.get("errors") or [] if e.get("kind") not in ("zero_rows", "extra_query")]
    return len(errs) >= 2 and _error_fingerprint(errs[-1]) == _error_fingerprint(errs[-2])


def _retry_or_give_up(state: AgentState) -> Literal["controller", "give_up", "step_limit"]:
    if limits.steps_exhausted(state.get("step_count", 0)):
        return "step_limit"
    err = state.get("last_error") or {}
    if err.get("kind") in ("connectivity", "db_query_limit"):
        return "give_up"                       # database unreachable / query budget spent: retrying cannot help
    if repeated_error(state):
        return "give_up"                       # loop guard: same error twice in a row -- a third try won't differ
    if limits.retries_exhausted(state.get("attempt", 0)):
        return "give_up"
    return "controller"                        # bounded retry with the structured error in state


def after_validate(state: AgentState) -> Literal["execute", "controller", "give_up", "step_limit"]:
    if state.get("last_error"):
        return _retry_or_give_up(state)
    if limits.steps_exhausted(state.get("step_count", 0)):
        return "step_limit"
    return "execute"


def after_execute(state: AgentState) -> Literal["controller", "give_up", "step_limit"]:
    if state.get("last_error"):
        return _retry_or_give_up(state)
    return "controller"                        # controller sees has_result and decides: final answer


def after_give_up(state: AgentState) -> Literal["answer", "__end__"]:
    return "answer" if state.get("has_result") else "__end__"


def after_step_limit(state: AgentState) -> Literal["answer", "__end__"]:
    return "answer" if state.get("has_result") else "__end__"
