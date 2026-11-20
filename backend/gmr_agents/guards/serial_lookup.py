"""A meter serial filtered on consumer_id, rewritten to meter_no by code.

hes_gold.dim_meter.consumer_id holds digits only, so `consumer_id = 'AL1157049'` (or the zero-stripped
`regexp_replace(TRIM(consumer_id), '^0+', '') = regexp_replace('AL1157049', '^0+', '')`) can never match a row.
Users call a serial a 'consumer' ('Details of consumer AL1157049'); the entity check sends that back for a repair,
but on the last attempt it is not checked and the 0-row SQL ran -- 20 Nov: 'No data available for consumer
AL1157049'. The rewrite only touches a predicate that is always false, so it cannot turn a right answer wrong.
Runs before the SQL guard: the rewritten text is guarded like any other.
"""

from __future__ import annotations

import re

_SERIAL_RE = re.compile(r"[A-Za-z]{1,4}\d{6,10}")


def _wrapped(inner: str) -> str:
    """inner, TRIM/UPPER/LOWER(inner), or regexp_replace(<either>, '..'[, '..']) -- balanced parentheses only."""
    one = rf"(?:(?:trim|upper|lower)\s*\(\s*{inner}\s*\)|{inner})"
    return rf"(?:regexp_replace\s*\(\s*{one}\s*,\s*'[^']*'\s*(?:,\s*'[^']*'\s*)?\)|{one})"


_COL = _wrapped(r"(?:\w+\.)?consumer_id")
_LIT = _wrapped(r"'\s*[A-Za-z]{1,4}\d{6,10}\s*'")
_EQ_RE = re.compile(rf"{_COL}\s*=\s*{_LIT}", re.I)
_ALIAS_RE = re.compile(r"(\w+)\.consumer_id", re.I)


def fix(sql: str) -> str:
    """The SQL with `<consumer_id expr> = '<serial>'` replaced by `meter_no = '<SERIAL>'` -- dim_meter queries only."""
    if not sql or "consumer_id" not in sql.lower() or not re.search(r"\bhes_gold\.dim_meter\b", sql, re.I):
        return sql

    def sub(m: re.Match) -> str:
        alias = _ALIAS_RE.search(m.group(0))
        col = f"{alias.group(1)}.meter_no" if alias else "meter_no"
        literal = m.group(0)[m.group(0).index("=") + 1:]
        return f"{col} = '{_SERIAL_RE.search(literal).group(0).upper()}'"
    return _EQ_RE.sub(sub, sql)
