"""Graph nodes. Each node does one job and returns only the state keys it changes;
LangGraph merges the update and picks the next node via supervisor/routing.py.

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

The nodes now live with their agents (agents/*/agent.py, supervisor/termination.py); this
module re-exports them so existing imports keep working.
"""

from __future__ import annotations

from gmr_agents.supervisor.steps import _tick  # noqa: F401
from gmr_agents.agents.router.agent import understand_node, reply_node  # noqa: F401
from gmr_agents.agents.knowledge.agent import explain_node  # noqa: F401
from gmr_agents.agents.sql_writer.agent import _EXAMPLE_FALLBACK_RATIO, _verified_example_plan, controller_node, _ERROR_KIND, validate_node, execute_node, _execute_calculator  # noqa: F401
from gmr_agents.agents.narrator.agent import answer_node  # noqa: F401
from gmr_agents.supervisor.termination import give_up_node, step_limit_node  # noqa: F401
