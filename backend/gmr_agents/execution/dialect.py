"""SQL dialect facts shared by the executor, the guard and the verifier."""

from __future__ import annotations

import sqlglot
from sqlglot import exp

# Every query is parsed and rendered as Presto (Athena's engine). The WFM
# gold Postgres templates are written in the common subset both accept.
DIALECT = "presto"

AGGREGATE_NODES = (exp.Sum, exp.Avg, exp.Max, exp.Min, exp.Count)


def has_aggregate(tree: exp.Expression) -> bool:
    """True when the query computes SUM / AVG / MAX / MIN / COUNT anywhere."""
    return any(next(tree.find_all(cls), None) is not None for cls in AGGREGATE_NODES)


def _safe_denominator(node: exp.Expression) -> bool:
    """A denominator that can never be zero (or is already guarded)."""
    while isinstance(node, (exp.Paren, exp.Cast)):
        node = node.this
    if isinstance(node, exp.Nullif):
        return True
    if isinstance(node, exp.Literal) and node.is_number:
        try:
            return float(node.this) != 0
        except ValueError:
            return False
    return False


def guard_division(sql: str, read: str) -> str:
    """`sql` with every `x / y` rewritten to `x / NULLIF(y, 0)`, so a zero
    denominator (e.g. a month with no supply) gives NULL for that row instead of
    failing the whole query with 'division by zero'. Unchanged when every
    division is already safe or the SQL does not parse."""
    try:
        tree = sqlglot.parse_one(sql, read=read)
    except Exception:
        return sql
    changed = False
    for div in list(tree.find_all(exp.Div)):
        den = div.expression
        if den is None or _safe_denominator(den):
            continue
        den.replace(exp.Nullif(this=den.copy(), expression=exp.Literal.number(0)))
        changed = True
    return tree.sql(dialect=read) if changed else sql
