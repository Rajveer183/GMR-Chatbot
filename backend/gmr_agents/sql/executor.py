"""SQL execution: plan -> result bundle (rows, headline, scope counts) from
Athena or Postgres for the LLM pipeline.

Athena bills by bytes scanned, so the two follow-up queries run only when
the answer needs them: the in-scope row count only for an aggregate (to tell
"an aggregate over zero rows" from "rows found"), the group count only when
the result hit its LIMIT.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import pandas as pd

from shared import config
import sqlglot
from sqlglot import exp

from gmr_agents.schema.catalog import AthenaCatalog
from db_connection.readonly import assert_read_only
from gmr_agents.sql.connections import athena_query, pooled_postgres_connection
from gmr_agents.sql import hierarchy
from gmr_agents.sql.dialect import DIALECT, guard_division, has_aggregate

log = logging.getLogger(__name__)


def run_sql(sql: str, database: str | None, engine: str) -> pd.DataFrame:
    """Dispatches to the correct connector -- the one place execute_sql
    (and its own scope-rows/group-count follow-up queries) needs to know
    which physical engine a validated query targets. `engine` comes from
    gmr_agents/validation/sql_guard.py::check_sql's own table-lookup, not guessed
    from the SQL text -- that guard already verified every table in this
    query is the SAME engine before this ever runs."""
    sql = guard_division(sql, "postgres" if engine == "postgres" else DIALECT)
    assert_read_only(sql)  # database-layer read-only check (db_connection/readonly.py), both engines
    if engine == "postgres":
        # Place hierarchy repair (sql/hierarchy.py): blank parent levels filled from the data's
        # own hierarchy, and stock with no place labelled 'Not installed yet (in stock)'.
        hierarchy.refresh_in_background()
        hierarchy.wait_first_load()
        fixed = hierarchy.rewrite(sql)
        if fixed != sql:
            assert_read_only(fixed)
            try:
                return hierarchy.natural_order(hierarchy.relabel(sql, _run_postgres(fixed)))
            except Exception as exc:
                log.warning("hierarchy: rewritten query failed (%s) -- running the original", exc)
        return hierarchy.natural_order(hierarchy.relabel(sql, _run_postgres(sql)))
    return hierarchy.natural_order(athena_query(sql, database=database))


def _run_postgres(sql: str) -> pd.DataFrame:
    try:
        with pooled_postgres_connection() as conn:
            return _read_with_timeout(sql, conn)
    except Exception as exc:
        # A pooled connection the server dropped while it sat idle fails
        # on first use -- that one is closed by the pool; retry once on
        # a fresh one. A statement timeout is not retried.
        if not _is_stale_connection_error(exc):
            raise
        log.info("postgres: pooled connection was stale (%s) -- retrying once.", exc)
        with pooled_postgres_connection() as conn:
            return _read_with_timeout(sql, conn)


def _read_with_timeout(sql: str, conn) -> pd.DataFrame:
    """Agent queries run under AGENT_SQL_TIMEOUT_MS (gold tables answer in ~50 ms; a query
    still running after seconds is a bad plan the planner should simplify). The pooled
    connection's own default (background catalog jobs) is restored afterwards."""
    with conn.cursor() as cur:
        cur.execute(f"SET statement_timeout = {int(config.AGENT_SQL_TIMEOUT_MS)}")
    try:
        return pd.read_sql_query(sql, conn)
    finally:
        try:
            with conn.cursor() as cur:
                cur.execute("RESET statement_timeout")
        except Exception:
            pass


_STALE_CONNECTION_HINTS = ("server closed the connection", "connection already closed", "ssl syscall error",
                           "ssl connection has been closed", "terminating connection")


def _is_stale_connection_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(h in text for h in _STALE_CONNECTION_HINTS)


# Keyword-sniffed rather than exception-type-matched on purpose: psycopg2's
# OperationalError covers both a connection failure AND some server-side
# errors, and boto3/Athena's own connectivity exceptions aren't imported
# here at all (this module never imports botocore directly) -- checking the
# message text works for both connectors without needing either's exception
# hierarchy, and every phrase below was confirmed to actually appear in a
# real driver's error message for exactly this class of failure.
_CONNECTIVITY_ERROR_HINTS = (
    "timed out", "timeout", "could not connect", "connection refused",
    "connection reset", "server closed the connection", "network is unreachable",
    "endpoint connection error", "name or service not known", "temporary failure in name resolution",
)


def _is_connectivity_error(exc: Exception) -> bool:
    text = str(exc).lower()
    if "statement timeout" in text or "canceling statement" in text:
        return False  # the QUERY was too slow (repairable by a simpler query), the database is fine
    return any(hint in text for hint in _CONNECTIVITY_ERROR_HINTS)


def _safe_number(value: Any) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(num) or math.isinf(num):
        return 0.0
    return num


def _looks_numeric(value: Any) -> bool:
    """Same rule as sql_executor.py's own _looks_numeric -- None/NaN counts
    as numeric (a real zero from an aggregate over zero rows), only a
    genuinely non-null non-numeric value is rejected."""
    if value is None:
        return True
    try:
        num = float(value)
    except (TypeError, ValueError):
        return False
    return True if math.isnan(num) else not math.isinf(num)


def _strip_limit(parsed: exp.Select) -> exp.Select:
    copy = parsed.copy()
    copy.set("limit", None)
    return copy


def _scope_count_sql(parsed: exp.Select, dialect: str) -> str:
    scoped = exp.Select(expressions=[exp.alias_(exp.Count(this=exp.Star()), "n")])
    scoped.set("from_", parsed.args.get("from_"))
    scoped.set("joins", parsed.args.get("joins"))
    scoped.set("where", parsed.args.get("where"))
    return scoped.sql(dialect=dialect)


def _group_count_sql(parsed: exp.Select, dialect: str) -> str:
    unlimited = _strip_limit(parsed)
    return f"SELECT COUNT(*) AS n FROM ({unlimited.sql(dialect=dialect)}) AS wrapped"


def _describe_filters(structured: dict) -> list[str]:
    described = []
    for f in structured.get("filters") or []:
        op = f.get("op", "=")
        described.append(f'{f.get("table")}.{f.get("column")} {op} {f.get("value")}')
    for j in structured.get("joins") or []:
        described.append(
            f'{j.get("left_table")}.{j.get("left_column")} = '
            f'{j.get("right_table")}.{j.get("right_column")} ({j.get("type", "inner")} join)'
        )
    return described


def execute_sql(plan: dict, catalog: AthenaCatalog) -> dict[str, Any]:
    """Runs the LLM planner's SQL exactly as written (gmr_agents/llm_pipeline.py)
    and returns the result bundle: rows, headline, counts, and on failure the
    database's own error text for the repair step."""
    metrics = [m["alias"] for m in plan.get("metrics") or []]
    group_by = [g["alias"] for g in plan.get("group_by") or []]
    sql = plan["sql"]
    database = plan.get("database")
    engine = plan.get("data_engine") or "athena"
    dialect = "postgres" if engine == "postgres" else DIALECT
    headline_alias = plan.get("headline_alias")
    warnings: list[str] = []

    try:
        # Local materialised results (outputs/sql_results.json): the same SQL since the last data refresh
        # is served without touching Postgres / Athena.
        from gmr_agents import answer_cache
        from gmr_agents.schema import knowledge_store

        version = answer_cache.current_version()
        cached = knowledge_store.get_result(sql, engine, version)
        if cached is not None:
            result = pd.DataFrame(cached[1], columns=cached[0])
            plan["result_cache"] = "hit"
        else:
            result = run_sql(sql, database, engine)
            knowledge_store.put_result(sql, engine, version, [str(c) for c in result.columns],
                                       result.astype(object).where(result.notna(), None).values.tolist())
    except Exception as exc:
        log.warning("%s execution failed for plan %r: %s", engine, sql, exc)
        if _is_connectivity_error(exc):
            # A connection timeout/refusal/reset is NOT the same failure as
            # a bad column or a genuinely empty result -- confirmed live:
            # this used to return the exact same generic "couldn't be run
            # as written" message for both, misleading a user into
            # thinking their QUESTION was wrong when the real issue was
            # the RDS/Athena endpoint being unreachable. "execution_error"
            # also tells node_execute's own zero-rows retry (pipeline.py) not
            # to bother -- retrying a fundamentally unreachable connection
            # with a different SQL plan can only fail the same way again,
            # burning an extra LLM call and ~20s for nothing (observed
            # directly: it did exactly this before this flag existed).
            warning = (
                f"The {engine} data source is temporarily unreachable (a connection problem, "
                f"not an issue with your question) -- please try again in a moment."
            )
        else:
            warning = ("This question couldn't be run against the data as written. "
                       "Try rephrasing it, or naming a specific column or table.")
        return {
            "result": pd.DataFrame(), "scope_rows": 0, "group_count": 0,
            "filters_applied": _describe_filters(plan), "headline": None,
            "metrics": metrics, "group_by": group_by,
            "warnings": [warning], "execution_error": True,
            # The database's own message -- fed back to the LLM planner's repair step.
            "error_text": str(exc)[:600], "connectivity_error": _is_connectivity_error(exc),
        }

    parsed = sqlglot.parse_one(sql, dialect=dialect)
    aggregated = has_aggregate(parsed)

    scope_rows = len(result)
    # The LLM pipeline detects empty answers itself and skips this second scan
    # (on Athena it doubled the query time).
    if aggregated and not plan.get("skip_scope_count"):
        # Only pay for this extra scan when it's actually needed to tell
        # "a real aggregate over zero rows" apart from "rows found" -- see
        # module docstring.
        try:
            scope_df = run_sql(_scope_count_sql(parsed, dialect), database, engine)
            scope_rows = int(scope_df.iloc[0]["n"])
        except Exception as exc:
            log.warning("scope_rows query failed for plan %r: %s", sql, exc)
            scope_rows = len(result)

    group_count = len(result)
    limit_node = parsed.args.get("limit")
    try:
        limit_value = int(limit_node.expression.name) if limit_node is not None else None
    except (AttributeError, TypeError, ValueError):
        limit_value = None
    # Fewer rows than the LIMIT means the result is already complete -- its
    # length IS the count, so skip the second full scan (~2-4 s on Athena).
    if limit_node is not None and (limit_value is None or len(result) >= limit_value):
        try:
            group_df = run_sql(_group_count_sql(parsed, dialect), database, engine)
            group_count = int(group_df.iloc[0]["n"])
        except Exception as exc:
            log.warning("group_count query failed for plan %r: %s", sql, exc)
            group_count = len(result)

    if result.empty:
        return {
            "result": result, "scope_rows": int(scope_rows), "group_count": 0,
            "filters_applied": _describe_filters(plan), "headline": None,
            "metrics": metrics, "group_by": group_by,
            "warnings": ["No rows match those filters."],
        }

    headline = None
    headline_metric = next((m for m in plan.get("metrics") or [] if m.get("alias") == headline_alias), None)
    if headline_metric is not None and headline_alias in result.columns and _looks_numeric(result.iloc[0][headline_alias]):
        top = result.iloc[0]
        if group_by:
            label = " / ".join(str(top[g]) for g in group_by if g in result.columns)
        else:
            label = headline_metric.get("source_column", headline_alias)
        headline = {"label": label, "value": _safe_number(top[headline_alias]), "column": headline_alias, "unit": None}

    if aggregated and scope_rows == 0:
        warnings = warnings + [
            "No rows match those filters -- the figure shown reflects zero matching records, not a computed total."
        ]

    return {
        "result": result, "scope_rows": int(scope_rows), "group_count": int(group_count),
        "filters_applied": _describe_filters(plan), "headline": headline,
        "metrics": metrics, "group_by": group_by, "warnings": warnings,
    }
