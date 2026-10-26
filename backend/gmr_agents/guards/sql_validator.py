"""SQL validation guard -- runs before anything reaches the database.

Who controls it: application code. Two layers:
  1. sql_guard.check_sql_safety (existing, AST-based via sqlglot): exactly one
     read-only SELECT; INSERT / UPDATE / DELETE / DROP / ALTER / TRUNCATE / CREATE /
     GRANT / COPY ... are rejected; only catalog tables; returns the normalized SQL
     and which engine (Postgres / Athena) runs it.
  2. entity_check (existing): the places / OEMs / categories / SAT stages / years /
     top-N named in the question must appear in the SQL's filters -- catches a
     valid-but-wrong query ("Kashi" asked, no Kashi filter) before it runs.
A failure is returned as a structured error for the controller, never raised.
"""

from __future__ import annotations

from typing import Any

from gmr_agents.validation import entity_check, sql_guard


def validate_sql(sql: str, catalog) -> dict[str, Any]:
    """{"ok": True, "normalized_sql", "data_engine"} or {"ok": False, "error"}."""
    if not sql.strip():
        return {"ok": False, "error": {"kind": "validation", "message": "No SQL was produced."}}
    guard = sql_guard.check_sql_safety(sql, catalog)
    if not guard["ok"]:
        return {"ok": False, "error": {"kind": "validation", "message": f"Rejected by the safety check: {guard['reason']}"}}
    return {"ok": True, "normalized_sql": guard["normalized_sql"], "data_engine": guard["data_engine"], "guard": guard}


def check_entities(question: str, sqls: list[str], columns: set[str] | None = None,
                   asked: str | None = None) -> dict[str, Any] | None:
    """A structured error when the SQL misses an entity the question names, else None.
    `columns` (the SQL's tables' columns) skips checks those tables cannot satisfy; `asked` is the
    user's own wording, for checks the resolved question can blur."""
    problems = entity_check.check(question, sqls, columns=columns, asked=asked)
    if not problems:
        return None
    return {"kind": "entity_mismatch", "message": "The SQL does not match the question: " + "; ".join(problems),
            "problems": problems}
