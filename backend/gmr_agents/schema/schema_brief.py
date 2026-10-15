"""Compact schema explanation for the LLM SQL planner.

The old prompt_context() (catalog.py) spent ~20K tokens per question, mostly
on boilerplate repeated per column. That cannot fit Groq's free tier (8K
tokens/minute per model), so this renders the same facts densely:

    ### public.gold_mi_installation_day_agg  (postgres, 565,960 rows)
    Installations per day by geography / category / OEM.
    - period_start date: day of the installation
    - project varchar = agra | kashi | triveni
    - meter_count bigint: meters installed

One line per column: name, type, its business meaning (COLUMN_HINTS, cut to
one sentence) and -- for low-cardinality Postgres columns -- the real stored
values to filter on. The tables are the ones the triage LLM chose (plus its
rule topics' tables), rendered until the token budget is used.
"""

from __future__ import annotations

import re

from gmr_agents.schema.catalog import TableMeta
from gmr_agents.schema.table_metadata import COLUMN_HINTS
from shared.llm_gateway import estimate_tokens

_SENTENCE_END_RE = re.compile(r"(?<=[.;])\s")
_WS_RE = re.compile(r"\s+")


def _short(text: str | None, limit: int) -> str:
    if not text:
        return ""
    text = _WS_RE.sub(" ", str(text)).strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    m = list(_SENTENCE_END_RE.finditer(cut))
    if m and m[-1].start() > limit // 2:
        return cut[: m[-1].start()].rstrip()
    return cut.rsplit(" ", 1)[0] + "..."


def _values(values: list[str] | None, limit: int) -> str:
    if not values:
        return ""
    shown = [str(v) for v in values[:limit]]
    more = f" | ...(+{len(values) - limit})" if len(values) > limit else ""
    return " = " + " | ".join(shown) + more


def render_table(t: TableMeta, *, hint_chars: int = 150, max_values: int = 14, in_scope: set[str] | None = None) -> str:
    lines = [f"### {t.sql_reference}  ({t.engine}, {t.row_count:,} rows)"]
    if t.description:
        lines.append(_short(t.description, 260))
    if t.is_large:
        lines.append("LARGE table: aggregates MUST filter by a date column or a zone/circle/division equality.")
    for c in t.columns:
        hint = _short(COLUMN_HINTS.get((t.table_name, c.column_name)), hint_chars)
        line = f"- {c.column_name} {c.sql_type}{_values(c.sample_values, max_values)}"
        if hint:
            line += f": {hint}"
        lines.append(line)
    joins = []
    for j in t.join_candidates:
        if j.left_table != t.table_name:
            continue
        if in_scope is not None and j.right_table not in in_scope:
            continue
        joins.append(f"{j.left_column} -> {j.right_table}.{j.right_column}")
    if joins:
        lines.append("joins: " + "; ".join(joins))
    return "\n".join(lines)


def build(priority: list[TableMeta], extra: list[TableMeta], *, token_budget: int = 2600) -> tuple[str, list[str]]:
    """Render `priority` tables (always, compressed if needed) then `extra`
    tables while they fit the budget. Returns (text, table names used)."""
    chosen: list[TableMeta] = []
    seen: set[str] = set()
    for t in priority + extra:
        if t.table_name not in seen:
            seen.add(t.table_name)
            chosen.append(t)
    in_scope = {t.table_name for t in chosen}

    blocks: list[str] = []
    used: list[str] = []
    total = 0
    priority_names = {t.table_name for t in priority}
    for t in chosen:
        block = render_table(t, in_scope=in_scope)
        cost = estimate_tokens(block)
        if total + cost > token_budget:
            if t.table_name not in priority_names:
                continue
            # A topic table must be shown: drop the column meanings to fit.
            block = render_table(t, hint_chars=0, max_values=8, in_scope=in_scope)
            cost = estimate_tokens(block)
            if total + cost > token_budget * 1.25:
                continue
        blocks.append(block)
        used.append(t.table_name)
        total += cost
    return "\n\n".join(blocks), used
