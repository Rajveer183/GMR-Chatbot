"""Router agent node: understand (code router, else the triage LLM) and reply
(chit-chat / out of scope). Prompt and parsing: agents/router/triage.py.

Moved from gmr_agents/graph/nodes.py unchanged (llm_pipeline helpers are now called on
their own modules).
"""

from __future__ import annotations
from typing import Any
from langchain_core.runnables import RunnableConfig
from gmr_agents.resolvers import language
from gmr_agents.supervisor import context
from gmr_agents.supervisor.state import AgentState
from gmr_agents.agents import common as C
from gmr_agents.agents.router import triage as RT
from gmr_agents.supervisor.steps import _tick


# --------------------------------------------------------------------------- #
# understand -- controller, first decision: what kind of request is this?
# --------------------------------------------------------------------------- #

def understand_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Who controls it: the LLM (triage call) decides kind / topics / tables /
    resolved follow-up question; a near-exact wording of a verified example is
    routed by code without an LLM call (RT._route)."""
    ctx = context.get(config)
    ps = ctx.pipeline_state
    prior = state.get("prior_turns") or []
    C._emit(ps, "status", message="Understanding your question")
    ps["memory_turns"] = prior
    triage = RT._route(ps, prior)
    if triage is None:
        triage = RT._triage(ps, prior)
    if triage is None:
        C._finish(ps, C._BUSY, refused=True, source="none (AI busy)")
        return {"route": "busy", "status": "busy", "refused": True, "step_count": _tick(state)}
    RT.fix_question(ps, triage, prior)  # place scope + the user's own category words, by code
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
    C._finish(ps, triage["reply"] or "How can I help with GMR's meter programme data?",
              refused=triage["kind"] == "out_of_scope", source=triage["source"])
    # The triage LLM was asked for REPLY LANGUAGE (explicit / dropdown) or the user's own language,
    # but does not always comply: record what the reply really is in. A mismatch -- and the canned
    # English fallback -- is translated afterwards (main._in_language).
    ps["answer_language"] = language.written_in(triage["reply"], triage["language"]) if triage["reply"] else "en"
    return {"status": "refused" if triage["kind"] == "out_of_scope" else "answered", "step_count": _tick(state)}
