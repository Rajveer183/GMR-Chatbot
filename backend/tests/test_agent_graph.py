"""Scenario tests for the LangGraph agent (gmr_agents/graph/) -- no LLM quota used.

The controller / answer LLM is replaced by scripted replies; everything else is
real: allowlist, SQL guard, idempotency guard, executor, live database, routing,
step limit. Each scenario prints the path the graph took.

    python tests/test_agent_graph.py
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8")

from gmr_agents import llm_pipeline as P  # noqa: E402
from gmr_agents.guards import limits  # noqa: E402
from shared import config, llm_gateway  # noqa: E402

_MAX_DB = config.QUESTION_MAX_DB_QUERIES
GOOD = "SELECT total_meter_installations AS meters_installed FROM public.gold_dashboard_stats_snapshot WHERE singleton"
UNSAFE = "DELETE FROM public.gold_dashboard_stats_snapshot"
DB_ERROR = "SELECT total_meter_installations / 0 AS boom FROM public.gold_dashboard_stats_snapshot WHERE singleton"


def calc(expression) -> dict:
    return {"tool": "calculator", "expression": expression, "reasoning": "scripted arithmetic"}


def plan(sql: str, tool: str = "run_sql") -> dict:
    return {"tool": tool, "sql": sql, "answer_kind": "number", "reasoning": "scripted",
            "display": {"cards": ["meters_installed"]}, "labels": {"meters_installed": "Meters Installed"}}


def install(script: list[dict]) -> list[str]:
    """Replace the LLM: planner calls pop `script`; answer calls return a summary
    built from the real rows. Returns the log of which tasks were called."""
    calls: list[str] = []

    def fake_chat_json(task, prompt, **_kw):
        calls.append(task)
        if task in ("sql", "repair"):
            if not script:
                return None  # no model available
            return llm_gateway.LLMResult(script.pop(0), "scripted", "scripted", None, None, 0.0)
        return llm_gateway.LLMResult({"summary": "A total of 4,214,678 meters have been installed.",
                                      "insights": [], "followups": [1, 2, 3]}, "scripted", "scripted", None, None, 0.0)

    llm_gateway.chat_json = fake_chat_json
    llm_gateway.chat_json_stream = lambda task, prompt, **kw: fake_chat_json(task, prompt)
    P._route = lambda state, prior: {"kind": "data", "question": state["question"], "followup": False, "domain": "wfm",
                                     "language": "en", "parts": [], "examples": [], "assumption": "",
                                     "topics": ["dashboard_stats"], "tables": [], "reply": "", "source": "scripted"}
    return calls


def run(name: str, script: list[dict], *, max_steps: int | None = None, max_repairs: int | None = None,
        question: str = "How many meters have been installed so far?", cost_limit_after: int | None = None,
        db_down: bool = False, max_db: int | None = None) -> None:
    limits.MAX_STEPS = max_steps or config.AGENT_MAX_STEPS
    limits.MAX_REPAIRS = config.LLM_MAX_REPAIRS if max_repairs is None else max_repairs
    config.QUESTION_MAX_DB_QUERIES = max_db if max_db is not None else _MAX_DB
    calls = install(script)
    if cost_limit_after is not None:
        # Emulate the gateway's per-question cap: after N planner calls _admit refuses with COST_LIMIT_EXCEEDED.
        inner = llm_gateway.chat_json

        def capped(task, prompt, **kw):
            if task in ("sql", "repair") and calls.count("sql") + calls.count("repair") >= cost_limit_after:
                llm_gateway._SCOPE.get().limit_hit = "COST_LIMIT_EXCEEDED: test cap"
                return None
            return inner(task, prompt, **kw)
        llm_gateway.chat_json = capped
    from gmr_agents.tools import sql_executor
    real_run_sql = sql_executor.run_sql
    if db_down:
        sql_executor.run_sql = lambda plan, catalog: {"ok": False, "result": {}, "error": {
            "kind": "connectivity", "message": "The database did not respond in time (timeout)."}}
    try:
        state = P.run_agent(question)
    finally:
        sql_executor.run_sql = real_run_sql
    path = [s["step"] for s in state["trace"] if s["step"] in ("plan", "validate", "execute", "present", "answer")]
    detail = [f"{s['step']}:{s.get('outcome') or ('reused' if s.get('reused') else s.get('error') or s.get('rows', ''))}"
              for s in state["trace"] if s["step"] in ("validate", "execute")]
    print(f"\n=== {name}")
    print("  path     :", " -> ".join(path))
    print("  verdicts :", ", ".join(map(str, detail)))
    print("  graph    :", state.get("graph"))
    print("  LLM calls:", calls)
    print("  answer   :", state.get("summary"))
    print("  execution:", {k: v for k, v in (state.get("execution") or {}).items()
                           if k in ("steps", "llm_calls", "tool_calls", "db_queries", "termination")})


if __name__ == "__main__":
    run("1. happy path", [plan(GOOD)])
    run("2. unsafe SQL is rejected, retry succeeds", [plan(UNSAFE), plan(GOOD)])
    run("3. tool not on the allowlist", [plan(GOOD, tool="shell"), plan(GOOD)])
    run("4. database error -> structured error -> retry", [plan(DB_ERROR), plan(GOOD)])
    run("5. same failing SQL twice -> second is not executed", [plan(DB_ERROR), plan(DB_ERROR), plan(GOOD)])
    run("6. LLM keeps producing bad requests -> retries exhausted", [plan(UNSAFE)] * 10, max_repairs=2)
    run("7. maximum step limit", [plan(UNSAFE)] * 10, max_steps=4, max_repairs=50)
    run("8. calculator question", [calc("((1200 - 150) / 1200) * 100")], question="What is (1200 - 150) / 1200 as a percentage?")
    run("9. malformed calculator call (code) -> rejected, recovers", [calc("__import__('os').system('dir')"), calc("1050 / 1200 * 100")],
        question="What is 1050 out of 1200 in percent?")
    run("10. DB timeout -> structured error, no retry loop", [plan(GOOD), plan(GOOD)], db_down=True)
    run("11. cost limit -> terminated", [plan(UNSAFE), plan("DROP TABLE public.x"), plan("SELECT * FROM nowhere_table")] * 3,
        cost_limit_after=2, max_repairs=50)
    run("12. database query cap", [plan(DB_ERROR), plan(GOOD.replace("meters_installed", "meters_installed, 1 AS x"))], max_db=1)
    run("13. loop guard: same error twice -> stop, no 3rd attempt",
        [plan(DB_ERROR), plan(DB_ERROR.replace("AS boom", "AS boom2")), plan(GOOD)], max_repairs=5)
