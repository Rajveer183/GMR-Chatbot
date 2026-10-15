"""Loop limits for the agent graph.

Who controls it: application code (values from .env) -- enforced by the routing
functions in gmr_agents/graph/routing.py, with LangGraph's own recursion_limit as
a hard backstop. The LLM cannot extend either.

    AGENT_MAX_STEPS    nodes one question may run (controller, validate, execute, ...)
    LLM_MAX_REPAIRS    controller retries after a structured error (existing setting)
"""

from __future__ import annotations

from shared import config

MAX_STEPS: int = config.AGENT_MAX_STEPS
MAX_REPAIRS: int = config.LLM_MAX_REPAIRS
# LangGraph counts every node it runs; leave head-room above MAX_STEPS so our own
# step guard (which ends the run with an honest answer) always fires first.
RECURSION_LIMIT: int = MAX_STEPS * 2 + 10


def steps_exhausted(step_count: int) -> bool:
    return step_count >= MAX_STEPS


def retries_exhausted(attempt: int) -> bool:
    """attempt = controller calls made so far (0-based count of the next one)."""
    return attempt > MAX_REPAIRS
