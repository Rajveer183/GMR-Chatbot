"""Data freshness of the tables an answer read: 'data up to <date>' for the sources line, and
a warning when a table is behind. Built by code from the data skeleton's MIN..MAX per date
column (retrieval/data_skeleton.py, rebuilt daily in the background) -- no query on the
question path, never by the LLM.

Which date column: the ones the SQL actually uses, else every period column of the table.
Bookkeeping columns (created_at, updated_at, valid_from, refreshed_at ...) say when a row was
written, not which period it covers -- skipped. So is feeder_loss.current_date (always today)
and any column whose MAX is in the future (due / planned dates).

Cadence: a column NAMED as a month (month_date, raised_month ...) whose values sit on the 1st is
monthly ('data up to Aug 2026'; behind when the latest month is older than last month).
Everything else is daily ('data up to 1 Aug 2026'; behind when more than DATA_STALE_DAYS days
older than the day the skeleton measured it -- measuring against the skeleton's date, not
today, keeps a day-old skeleton from raising a false alarm). Not 'starts and ends on the 1st':
MDM data_date is daily data that happened to stop on 1 Aug (11 Nov) -- 'up to Aug 2026' would
have claimed the whole month. Tables with no period column (snapshots) get nothing here.
"""

from __future__ import annotations

import re
from datetime import date, datetime

from shared import config

_META_COLS = re.compile(
    r"^(?:created|updated|modified|inserted|loaded|refreshed|ingested|synced|deleted)_(?:at|on|date|ts|time)$"
    r"|^valid_(?:from|to)$|^current_date$|^eligibility_cutoff_date$|^period_start$", re.I)


def _day(v) -> date | None:
    if not v:
        return None
    try:
        return datetime.fromisoformat(str(v)[:10]).date()
    except ValueError:
        return None


def _months_between(a: date, b: date) -> int:
    return (b.year - a.year) * 12 + (b.month - a.month)


def _stale_days() -> int:
    return int(getattr(config, "DATA_STALE_DAYS", 3) or 0)


def check(tables: list[str], sql: str | None, skeleton: dict | None = None) -> list[dict]:
    """One entry per table with a period column:
    {table, column, latest (ISO date), monthly, stale, behind ('3 months' / '5 days' / '')}."""
    if skeleton is None:
        from gmr_agents.retrieval import data_skeleton

        skeleton = data_skeleton.load()
    cards = {c.get("table"): c for c in (skeleton or {}).get("tables") or []}
    measured = _day((skeleton or {}).get("built_at")) or date.today()
    sql_text = (sql or "").lower()
    out: list[dict] = []
    for raw in tables or []:
        name = str(raw).split(".")[-1].strip('"').lower()
        card = cards.get(name)
        if not card or any(o["table"] == name for o in out):
            continue
        ranges = {c: (_day(v[0]), _day(v[1])) for c, v in (card.get("dates") or {}).items()
                  if v and not _META_COLS.search(c)}
        ranges = {c: r for c, r in ranges.items() if r[1] and r[1] <= measured}
        if not ranges:
            continue
        used = {c: r for c, r in ranges.items() if re.search(rf"\b{re.escape(c.lower())}\b", sql_text)}
        col, (lo, hi) = max((used or ranges).items(), key=lambda kv: kv[1][1])
        monthly = "month" in col.lower() and bool(lo and lo.day == 1 and hi.day == 1)
        if monthly:
            lag = _months_between(hi, measured)
            stale = lag > 1
            behind = f"{lag} months" if stale else ""
        else:
            lag = (measured - hi).days
            limit = _stale_days()
            stale = bool(limit) and lag > limit
            behind = f"{lag} days" if stale else ""
        out.append({"table": name, "column": col, "latest": hi.isoformat(), "monthly": monthly,
                    "stale": stale, "behind": behind})
    return out


def label(item: dict) -> str:
    d = _day(item["latest"])
    return f"data up to {d:%b %Y}" if item["monthly"] else f"data up to {d.day} {d:%b %Y}"


def warnings(items: list[dict]) -> list[str]:
    """One plain sentence per stale table (several stale tables of the same date share one)."""
    by_date: dict[tuple, list[str]] = {}
    for i in items:
        if i["stale"]:
            by_date.setdefault((label(i), i["behind"]), []).append(i["table"])
    return [f"{', '.join(t)}: {lab} -- {behind} behind; newer figures are not loaded yet."
            for (lab, behind), t in by_date.items()]
