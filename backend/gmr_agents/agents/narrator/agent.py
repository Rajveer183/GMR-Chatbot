"""Answer Narrator agent node: KPI cards / chart / table by code (presentation/), the
summary by the answer LLM (agents/narrator/writer.py).

Moved from gmr_agents/graph/nodes.py unchanged (llm_pipeline helpers are now called on
their own modules).
"""

from __future__ import annotations
import time
from typing import Any
from langchain_core.runnables import RunnableConfig
from gmr_agents.supervisor import context
from gmr_agents.supervisor.state import AgentState
from gmr_agents.presentation import blocks as BL
from gmr_agents.agents import common as C
from gmr_agents.agents.narrator import writer as WR
from gmr_agents.supervisor.steps import _tick


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
    blocks = BL._blocks(plan, result)
    for i, ex in enumerate(extras, 2):
        blocks["tables"].extend(BL._blocks({**plan, "metrics": [], "display": {"table": True}, "labels": {},
                                           "table_title": ex["purpose"][:80]}, ex)["tables"])
        if blocks["tables"]:
            blocks["tables"][-1]["id"] = f"extra_table_{i}"
    C._step(ps, "present", t0, kpi=len(blocks["kpis"]), chart=len(blocks["charts"]), table=len(blocks["tables"]))
    C._emit(ps, "partial", kpis=blocks["kpis"], charts=blocks["charts"], tables=blocks["tables"],
            resolved_question=ps.get("resolved_question"))
    written = WR._write_answer(ps, triage, plan, result, extras)
    ps.update(kpis=blocks["kpis"], charts=blocks["charts"], tables=blocks["tables"], summary=written["summary"],
              insights=written["insights"], followups=written["followups"], refused=False,
              narration_source=written["source"])
    return {"status": "answered", "summary": written["summary"], "insights": written["insights"],
            "followups": written["followups"], "refused": False, "step_count": _tick(state)}
