"""The database query executor -- independent of the LLM.

Who controls it: application code calls it; the database runs the query. It only
ever receives SQL that already passed gmr_agents/guards/sql_validator.py, and it
knows nothing about prompts or models. Read-only: Postgres (WFM / MDM gold tables)
or Athena, chosen by the guard's `data_engine`.

Errors never raise out of here: they come back as a structured error the
controller can read ("the database returned: column x does not exist").
"""

from __future__ import annotations

from typing import Any

from gmr_agents.sql import executor


def run_sql(plan: dict, catalog) -> dict[str, Any]:
    """Execute one validated plan. Returns {"ok": True, "result": <executor result>}
    or {"ok": False, "error": {...}, "result": <executor result>}."""
    try:
        result = executor.execute_sql(plan, catalog)
    except Exception as exc:  # a driver bug must not crash the graph
        return {"ok": False, "error": {"kind": "tool_error", "message": f"{type(exc).__name__}: {exc}"}, "result": {}}
    if result.get("execution_error"):
        kind = "connectivity" if result.get("connectivity_error") else "database_error"
        message = (result.get("warnings") or [""])[0] if kind == "connectivity" else \
            f"The database returned an error: {result.get('error_text')}"
        return {"ok": False, "error": {"kind": kind, "message": message}, "result": result}
    return {"ok": True, "result": result}
