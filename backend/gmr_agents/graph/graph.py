"""Builds and runs the agent graph.

Who controls it: LangGraph -- this file wires nodes and conditional edges, compiles
the graph with a checkpointer, and enforces the recursion limit; run_graph() is
the entry point called by llm_pipeline.run_agent.

Checkpointing
    Development: InMemorySaver -- after every node LangGraph saves the state under
    thread_id = the chat session id, so `graph.get_state(config)` shows exactly where
    a question stopped (useful when debugging a refusal), and a later turn on the same
    thread starts from a fresh per-question state (we pass all request fields).
    Production: swap in a database-backed saver with the same interface (e.g.
    langgraph-checkpoint-postgres' PostgresSaver on the existing RDS) -- one line
    below; nodes and routing do not change. Conversation history itself is the
    session store's job (short-term memory), not the checkpointer's.
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from gmr_agents.graph import context, nodes, routing
from gmr_agents.graph.state import AgentState
from gmr_agents.guards import limits


def build_graph(checkpointer=None):
    g = StateGraph(AgentState)
    g.add_node("understand", nodes.understand_node)
    g.add_node("controller", nodes.controller_node)
    g.add_node("validate", nodes.validate_node)
    g.add_node("execute", nodes.execute_node)
    g.add_node("answer", nodes.answer_node)
    g.add_node("reply", nodes.reply_node)
    g.add_node("explain", nodes.explain_node)
    g.add_node("give_up", nodes.give_up_node)
    g.add_node("step_limit", nodes.step_limit_node)

    g.add_edge(START, "understand")
    g.add_conditional_edges("understand", routing.after_understand,
                            {"controller": "controller", "explain": "explain", "reply": "reply", "__end__": END})
    g.add_conditional_edges("controller", routing.after_controller,
                            {"validate": "validate", "answer": "answer", "give_up": "give_up",
                             "step_limit": "step_limit", "__end__": END})
    g.add_conditional_edges("validate", routing.after_validate,
                            {"execute": "execute", "controller": "controller", "give_up": "give_up",
                             "step_limit": "step_limit"})
    g.add_conditional_edges("execute", routing.after_execute,
                            {"controller": "controller", "give_up": "give_up", "step_limit": "step_limit"})
    g.add_conditional_edges("give_up", routing.after_give_up, {"answer": "answer", "__end__": END})
    g.add_conditional_edges("step_limit", routing.after_step_limit, {"answer": "answer", "__end__": END})
    g.add_edge("answer", END)
    g.add_edge("reply", END)
    g.add_edge("explain", END)
    return g.compile(checkpointer=checkpointer)


CHECKPOINTER = InMemorySaver()          # production: a database-backed saver (see module docstring)
GRAPH = build_graph(CHECKPOINTER)


def run_graph(pipeline_state: dict[str, Any], prior_turns: list[dict] | None, session_id: str | None) -> dict[str, Any]:
    """Run one question through the graph. Fills `pipeline_state` (the dict
    run_agent returns) and returns the final graph state."""
    tid = pipeline_state["trace_id"]
    context.open_run(tid, pipeline_state)
    config = {"configurable": {"thread_id": session_id or tid, "trace_id": tid},
              "recursion_limit": limits.RECURSION_LIMIT}
    initial: AgentState = {
        "question": pipeline_state["question"], "session_id": session_id, "trace_id": tid,
        "prior_turns": _compact_turns(prior_turns), "step_count": 0, "attempt": 0, "status": "running",
        "tool_call": None, "validated": None, "last_error": None, "has_result": False, "has_fallback": False,
        "entities_checked": False, "zero_rows_retried": False, "route": "data",
        "triage": {}, "summary": "", "insights": [], "followups": [], "refused": False,
        # None = reset (see state.py reducers): the thread's checkpoint holds the last question's lists
        "tool_calls": None, "errors": None, "executed": None,
    }
    try:
        final = GRAPH.invoke(initial, config)
    finally:
        context.close_run(tid)
    pipeline_state["graph"] = {"status": final.get("status"), "steps": final.get("step_count"),
                               "loop_detected": routing.repeated_error(final),
                               "attempts": final.get("attempt"), "errors": [e.get("kind") for e in final.get("errors") or []]}
    return final


def _compact_turns(prior_turns: list[dict] | None) -> list[dict]:
    """Session memory goes into the checkpointed state, so keep it JSON-safe."""
    keep = ("question", "resolved_question", "summary", "context", "kind")
    return [{k: t.get(k) for k in keep if k in t} for t in (prior_turns or [])]
