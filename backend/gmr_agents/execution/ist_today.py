"""'Today' is the IST date -- also in the database.

The prompts tell the SQL writer today's date in IST (agents/common.py) and the dashboard's own
cards use (NOW() AT TIME ZONE 'Asia/Kolkata')::date, but both database sessions run in UTC: from
00:00 to 05:30 IST a bare CURRENT_DATE is still yesterday, so 'yesterday' / 'last 7 days' /
'this month' answers were a day behind in that window.

anchor() swaps a bare CURRENT_DATE for the IST date expression of the engine -- ONLY near the IST
day boundary (23:55-05:35 IST). The rest of the day the UTC and IST dates are equal, and the SQL
sent to the database is left byte-identical (same plan, same rows, same error text). The minutes of
margin cover a clock difference between this machine and the database; the IST expression is the
right date at any hour, so the margin can't make an answer wrong.

Applied to the text sent to the database only: the SQL the pipeline logs, caches, shows and repairs
keeps CURRENT_DATE, and a database error message is mapped back to it.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30))

_EXPR = {
    "postgres": "(NOW() AT TIME ZONE 'Asia/Kolkata')::date",
    "athena": "CAST(current_timestamp AT TIME ZONE 'Asia/Kolkata' AS date)",
}
# String literals, quoted identifiers and comments are matched first and kept as they are, so
# 'current_date' inside them is never touched; a qualified or quoted column (t.current_date) and a
# function-call form are not the keyword either.
_TOKENS = re.compile(r"""('(?:[^']|'')*')|("(?:[^"]|"")*")|(--[^\n]*)|(/\*.*?\*/)"""
                     r"""|(?<![\w.])(current_date)\b(?!\s*\()""", re.IGNORECASE | re.DOTALL)
_START = 23 * 60 + 55   # 23:55 IST
_END = 5 * 60 + 35      # 05:35 IST (UTC date catches up with IST at 05:30)


def today() -> "datetime.date":
    """Today's date in IST, whatever time zone this machine runs in."""
    return datetime.now(IST).date()


def near_day_boundary(now: datetime | None = None) -> bool:
    """True while the UTC date can differ from the IST date (with a few minutes' margin)."""
    ist = (now or datetime.now(timezone.utc)).astimezone(IST)
    minutes = ist.hour * 60 + ist.minute
    return minutes >= _START or minutes < _END


def anchor(sql: str, engine: str, now: datetime | None = None) -> str:
    """`sql` with bare CURRENT_DATE as the IST date for `engine` ('postgres' / 'athena').
    Returns `sql` itself (the same string) outside the day-boundary window or when there is nothing to swap."""
    expr = _EXPR.get("postgres" if engine == "postgres" else "athena")
    if "current_date" not in sql.lower() or not near_day_boundary(now):
        return sql

    def swap(m: re.Match) -> str:
        return expr if m.group(5) else m.group(0)

    return _TOKENS.sub(swap, sql)


def unanchor_error(exc: BaseException, engine: str) -> BaseException:
    """The database error with the IST expression shown as CURRENT_DATE again, so a repair
    prompt quotes the SQL the writer actually wrote (same text, same tokens)."""
    expr = _EXPR.get("postgres" if engine == "postgres" else "athena")
    try:
        exc.args = tuple(a.replace(expr, "CURRENT_DATE") if isinstance(a, str) else a for a in exc.args)
    except Exception:
        pass
    return exc
