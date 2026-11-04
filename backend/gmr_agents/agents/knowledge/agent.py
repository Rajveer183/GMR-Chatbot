"""Knowledge agent node: explain a KPI / term from retrieved chunks (no SQL).
Prompt and grounding: agents/knowledge/explain.py.

Moved from gmr_agents/graph/nodes.py unchanged (llm_pipeline helpers are now called on
their own modules).
"""

from __future__ import annotations
from typing import Any
from langchain_core.runnables import RunnableConfig
from gmr_agents.supervisor import context
from gmr_agents.supervisor.state import AgentState
from gmr_agents.agents import common as C
from gmr_agents.agents.knowledge import explain as KN
from gmr_agents.supervisor.steps import _tick


def explain_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Who controls it: the LLM explains from the rules + glossary; no tool runs."""
    ps = context.get(config).pipeline_state
    triage = state["triage"]
    written = KN._explain(ps, triage)
    C._finish(ps, written["summary"], refused=written["refused"], source=written["source"],
              followups=written["followups"], insights=written["insights"], agent=triage["domain"])
    return {"status": "refused" if written["refused"] else "answered", "step_count": _tick(state)}
