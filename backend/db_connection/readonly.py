"""Read-only enforcement at the database layer -- the last line of defence.

Who controls it: application code, below everything else: it runs inside
db_connection/ right before a statement is sent to Postgres or Athena, whatever
called it (the agent, the catalog refresh, the data skeleton, a future tool).

Layers, outermost first (any one alone stops a write):
  1. SQL guard (gmr_agents/validation/sql_guard.py): one SELECT on catalog tables
  2. assert_read_only() here: only SELECT / WITH (and SHOW / DESCRIBE for catalog
     metadata) as a single statement; no write or DDL keyword outside string literals
  3. Postgres sessions open with default_transaction_read_only=on, so the server
     itself refuses any write ("cannot execute ... in a read-only transaction")

Recommended on the infrastructure side (not settable from code): a dedicated
database role with SELECT-only grants, and an Athena workgroup / IAM policy
without write permissions on the data lake.
"""

from __future__ import annotations

import re

_ALLOWED_FIRST = {"SELECT", "WITH", "SHOW", "DESCRIBE", "DESC", "VALUES"}
# Statements that do not START with an allowed word are already refused; these are the
# words that could smuggle a write INSIDE a SELECT / WITH (e.g. Postgres data-modifying
# CTEs: WITH x AS (DELETE ... RETURNING ...)). Kept short so function / column names
# such as REPLACE() or "comment" never trip it.
_FORBIDDEN = {"INSERT", "UPDATE", "DELETE", "MERGE", "UPSERT", "DROP", "ALTER", "CREATE", "TRUNCATE", "GRANT",
              "REVOKE", "COPY", "CALL", "EXECUTE", "VACUUM", "UNLOAD", "OPTIMIZE"}
# SELECT ... FOR UPDATE takes row locks (it can block writers): refused too.
_SELECT_SAFE_PHRASES = re.compile(r"\bFOR\s+UPDATE\b", re.IGNORECASE)


class ReadOnlyViolation(PermissionError):
    """A statement that could change data or schema was about to reach the database."""


def _strip(sql: str) -> str:
    """Remove comments and quoted literals/identifiers so keywords inside them don't count."""
    sql = re.sub(r"--[^\n]*", " ", sql)
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    sql = re.sub(r"'(?:[^']|'')*'", "''", sql)
    sql = re.sub(r'"(?:[^"]|"")*"', '""', sql)
    sql = re.sub(r"`[^`]*`", "``", sql)
    return sql


def assert_read_only(sql: str) -> None:
    """Raise ReadOnlyViolation unless `sql` is one read-only statement."""
    body = _strip(str(sql or "")).strip().rstrip(";").strip()
    if not body:
        raise ReadOnlyViolation("empty statement")
    if ";" in body:
        raise ReadOnlyViolation("more than one statement")
    if _SELECT_SAFE_PHRASES.search(body):
        raise ReadOnlyViolation("row locking (FOR UPDATE) is not allowed")
    words = re.findall(r"[A-Za-z_]+", body)
    first = words[0].upper() if words else ""
    if first not in _ALLOWED_FIRST:
        raise ReadOnlyViolation(f"only read-only statements may run (got {first or '?'})")
    bad = sorted({w.upper() for w in words} & _FORBIDDEN)
    if bad:
        raise ReadOnlyViolation(f"write / DDL keyword(s) not allowed: {', '.join(bad)}")
