"""Step counter shared by every graph node (the step guard in supervisor/routing.py reads it).

Moved from gmr_agents/graph/nodes.py unchanged (llm_pipeline helpers are now called on
their own modules).
"""

from __future__ import annotations
from gmr_agents.supervisor.state import AgentState


def _tick(state: AgentState) -> int:
    return state.get("step_count", 0) + 1
