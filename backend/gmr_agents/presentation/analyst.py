"""Analyst: facts about a result computed by code for the answer writer -- partial periods,
short weeks, the date window, per-category totals and the compact result preview.

Moved from gmr_agents/llm_pipeline.py unchanged.
"""

from __future__ import annotations
import re
from datetime import datetime
import pandas as pd
from shared import config, llm_gateway
from gmr_agents.execution import executor
import logging
from gmr_agents.agents.common import _ANSWER_ROWS, _IST
from gmr_agents.presentation.blocks import _tidy_ts

log = logging.getLogger("gmr_agents.llm_pipeline")  # same logger name as before the split


def _partial_periods(df: pd.DataFrame | None) -> tuple[str | None, set[str], str]:
    """(period column, its partial labels, a PERIODS line) for a month- or day-wise RESULT.
    Only the current IST month (monthly rows) or today (daily rows) is partial -- the model
    otherwise guessed 'partial' for whichever month came last, e.g. a complete August."""
    if not isinstance(df, pd.DataFrame) or len(df) < 2:
        return None, set(), ""
    now = datetime.now(_IST)
    for col in [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]:
        raw = df[col].astype(str).str.strip()
        raw = raw[raw.str.match(r"^\d{4}-\d{2}-\d{2}")]
        if raw.nunique() < 2:
            continue
        dates = pd.to_datetime(raw.str[:10], errors="coerce").dropna()
        if dates.empty:
            continue
        if (dates.dt.day == 1).all():
            current = {v for v, d in zip(raw, dates) if (d.year, d.month) == (now.year, now.month)}
            unit, today_label = "month", f"{now:%Y-%m}"
        else:
            current = {v for v, d in zip(raw, dates) if d.date() == now.date()}
            unit, today_label = "day", f"{now:%Y-%m-%d}"
        if current:
            line = (f"PERIODS (checked by code): {', '.join(sorted(current))} is the current {unit} ({today_label}) "
                    f"and still partial -- say so, and keep it out of highest / lowest / drop claims. "
                    f"Every other {unit} in RESULT is complete.")
        else:
            line = (f"PERIODS (checked by code): every {unit} in RESULT is complete (the current {unit} is "
                    f"{today_label}, not in RESULT) -- do NOT call any of them partial.")
        return col, current, line
    return None, set(), ""


_WEEK_LABEL_RE = re.compile(r"^(\d{1,2})\s+([A-Za-z]{3})[a-z]*\s*[-–]\s*(\d{1,2})\s+([A-Za-z]{3})[a-z]*$")


def _short_weeks(df: pd.DataFrame | None) -> tuple[str | None, dict[str, int]]:
    """(week column, {label: days}) for month-anchored weeks shorter than 7 days -- the
    29-end bucket has 1-3 days, so its TOTALS always look like the 'lowest week'
    (Triveni, Aug 2026: 29-31 Aug called the lowest with 3 days of installs)."""
    if not isinstance(df, pd.DataFrame) or len(df) < 3:
        return None, {}
    for col in [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]:
        raw = df[col].astype(str).str.strip()
        short: dict[str, int] = {}
        text = raw[raw.str.match(_WEEK_LABEL_RE)]
        if len(text) >= 2:  # '29 Aug - 31 Aug'
            for v in text:
                m = _WEEK_LABEL_RE.match(v)
                days = int(m.group(3)) - int(m.group(1)) + 1
                if m.group(2).lower() == m.group(4).lower() and 0 < days < 7:
                    short[v] = days
            return col, short
        dated = raw[raw.str.match(r"^\d{4}-\d{2}-\d{2}")]
        dates = pd.to_datetime(dated.str[:10], errors="coerce").dropna()
        if dates.nunique() >= 2 and dates.dt.day.isin([1, 8, 15, 22, 29]).all() and (dates.dt.day != 1).any():
            for v, d in zip(dated, dates):  # a weekly bucket starting on the 29th: 1-3 days
                if d.day == 29:
                    short[v] = d.days_in_month - 28
            return col, short
    return None, {}


_DATE_COLS = {"period_start", "mi_date"}


def _date_window(plan: dict, df: pd.DataFrame | None) -> str:
    """The date range the SQL really covered -- its period_start bounds evaluated by Postgres -- plus the
    row grain. The answer only sees the first rows of RESULT: live, 23 Oct, 'meters installed in last 3
    months' returned 92 day rows newest-first, and the answer called the window '22 Aug to 22 Oct'
    (it was 22 Jul .. 22 Oct) and a day 'the peak month'."""
    sql = plan.get("sql") or ""
    if plan.get("data_engine") != "postgres" or not sql:
        return ""
    try:
        import sqlglot
        from sqlglot import exp

        root = sqlglot.parse_one(sql, read="postgres")
        lows, highs = [], []
        for node in root.find_all(exp.Between, exp.GTE, exp.GT, exp.LTE, exp.LT, exp.EQ):
            col = node.this
            if not isinstance(col, exp.Column) or col.name.lower() not in _DATE_COLS:
                continue
            if isinstance(node, exp.Between):
                lo, hi = node.args.get("low"), node.args.get("high")
            elif isinstance(node, exp.EQ):
                lo = hi = node.expression
            else:
                lo, hi = (node.expression, None) if isinstance(node, (exp.GTE, exp.GT)) else (None, node.expression)
            for bound, bucket, shift in ((lo, lows, " + 1" if isinstance(node, exp.GT) else ""),
                                         (hi, highs, " - 1" if isinstance(node, exp.LT) else "")):
                if bound is not None and not bound.find(exp.Column, exp.Select):
                    bucket.append(f"CAST({bound.sql(dialect='postgres')} AS date){shift}")
        if not lows and not highs and any(t.name.lower().endswith(("_day_agg", "_month_agg"))
                                          for t in root.find_all(exp.Table)):
            # No date asked = all time here, while the dashboard card defaults to the last 30 days.
            return ("DATE WINDOW (checked by code): no date filter -- the figure is ALL TIME to date. Say 'so far' / "
                    "'all time' in the answer; the dashboard card shows the last 30 days by default, so the user "
                    "may see a smaller number there.")
        if not lows or not highs:
            return ""
        hi = f"GREATEST({', '.join(highs)})"
        if any(t.name.lower().endswith("_month_agg") for t in root.find_all(exp.Table)):
            hi = f"(DATE_TRUNC('month', {hi}) + INTERVAL '1 month - 1 day')::date"  # a month row covers its month
        row = executor._run_postgres(f"SELECT LEAST({', '.join(lows)}) AS lo, {hi} AS hi").iloc[0]
        window = f"{pd.Timestamp(row['lo']):%d %b %Y} .. {pd.Timestamp(row['hi']):%d %b %Y}"
    except Exception:  # a hint only -- never fail the answer over it
        log.debug("date window: could not evaluate", exc_info=True)
        return ""
    grain = ""
    if isinstance(df, pd.DataFrame) and len(df) >= 2:
        for c in df.columns:
            vals = pd.to_datetime(df[c].astype(str).str[:10], format="%Y-%m-%d", errors="coerce").dropna()
            if len(vals) >= 2:
                grain = (" Rows are per MONTH." if (vals.dt.day == 1).all() else
                         " Rows are per WEEK (month-anchored)." if vals.dt.day.isin([1, 8, 15, 22, 29]).all() else
                         " Rows are per DAY -- call a row a day, never a month, whatever the column is named.")
                break
    return (f"DATE WINDOW (evaluated by code from the SQL): {window}.{grain} State exactly this period; RESULT "
            "may show only the first rows, so never infer the start or end date from the rows you see.")


def _result_facts(df: pd.DataFrame | None, max_cols: int = 4) -> str:
    """Highest / lowest item per numeric column, Total row excluded -- so 'which is lagging'
    never depends on the model reading the middle of a table (or mistaking Total for a row).
    The current (partial) month / day is left out too."""
    if not isinstance(df, pd.DataFrame) or len(df) < 2:
        return ""
    labels = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if not labels:
        return ""
    # A ROLLUP Total in any label column (e.g. Month='', Category='Total') is not an item.
    is_total = pd.Series(False, index=df.index)
    for c in labels:
        is_total |= df[c].astype(str).str.strip().str.lower().isin({"total", "grand total"})
    rows = df[~is_total]
    period_col, partial, _ = _partial_periods(df)
    week_col, short = _short_weeks(df)
    if week_col and not period_col:
        period_col = week_col
    # Month x Category results: highest / lowest month per category, never across categories.
    label = period_col or labels[0]
    groups = [c for c in labels if c != label][:1] if period_col else []
    fmt = lambda v: str(int(v)) if float(v).is_integer() else f"{v:.4f}".rstrip("0").rstrip(".")
    lines = []
    if is_total.sum() == 1 and len(rows) >= 2:
        # The overall figure is the Total row, never the item that shares the place's name: 'agra
        # zone wise' once reported the AGRA zone row (193.96 days) as the Agra project (171.03).
        tot = df[is_total].iloc[0]
        nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and pd.notna(tot[c])][:8]
        lines.append("- OVERALL (Total row = the whole scope asked): "
                     + ", ".join(f"{c} {fmt(float(tot[c]))}" for c in nums))
        lines.append(f"- BREAKDOWN ITEMS ({len(rows)}; each is one part, never the overall): "
                     + ", ".join(str(v) for v in rows[labels[0]].head(15)))
    if period_col and groups and is_total.any():
        # Month x Category with a Total row per month: the overall figure is the grand-total row
        # (every label blank / Total), else the sum of the per-month Totals.
        blank = pd.Series(True, index=df.index)
        for c in labels:
            blank &= df[c].isna() | df[c].astype(str).str.strip().str.lower().isin({"", "none", "nan", "total", "grand total"})
        per_period = df[is_total & ~blank]
        for col in [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not _RATE_COL_RE.search(str(c))][:max_cols]:
            v = df.loc[blank, col].iloc[0] if blank.sum() == 1 else (
                pd.to_numeric(per_period[col], errors="coerce").sum() if len(per_period) > 1 else None)
            if v is not None and pd.notna(v):
                lines.append(f"- OVERALL {col} (all {groups[0]} values, all periods) = {fmt(float(v))}")
    for key, part in (rows.groupby(groups[0], sort=False) if groups else [(None, rows)]):
        if len(lines) >= 4 * max_cols:
            break
        body = part[~part[label].astype(str).str.strip().isin(partial)] if partial else part
        prefix = f"{key} " if key is not None else ""
        for col in [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])][:max_cols]:
            # Sum over the periods asked for, so 'how many ... monthwise' can lead with the total.
            if period_col and len(part) > 1 and not _RATE_COL_RE.search(str(col)):
                lines.append(f"- {prefix}{col}: total across the {len(part)} periods = {fmt(pd.to_numeric(part[col], errors='coerce').sum())}")
            # A short week's COUNT is not comparable with full weeks (a rate such as productivity is).
            cmp = body
            if short and not _RATE_COL_RE.search(str(col)):
                cmp = body[~body[week_col].astype(str).str.strip().isin(short)]
            vals = pd.to_numeric(cmp[col], errors="coerce")
            if len(cmp) < 2 or vals.notna().sum() < 2 or vals.nunique() < 2:
                continue
            hi, lo = vals.idxmax(), vals.idxmin()
            lines.append(f"- {prefix}{col}: highest {cmp.at[hi, label]} ({fmt(vals[hi])}), "
                         f"lowest {cmp.at[lo, label]} ({fmt(vals[lo])})"
                         + (" (full weeks only)" if cmp is not body else ""))
    return "\n".join(lines)


# Columns that must not be summed across periods (rates, averages, stock levels, distinct
# counts -- daily active technicians summed over a week are not technicians or technician-days).
_RATE_COL_RE = re.compile(r"pct|percent|%|rate|ratio|share|avg|average|mean|per_|_per|productivity|pace|"
                          r"days|cover|stock|balance|cumulative|to_date|so_far|remaining|pending|"
                          r"technician|active|distinct|unique", re.I)


def _result_preview(df: pd.DataFrame | None, max_tokens: int | None = None) -> tuple[str, int]:
    """Query rows for the answer writer as CSV: at most ANSWER_PREVIEW_ROWS rows and
    MAX_TOOL_RESULT_TOKENS tokens, cut between rows (never inside one)."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return "(no rows)", 0
    shown = df.head(_ANSWER_ROWS).copy()
    for col in shown.columns:
        # Exact values so the summary matches the cards: whole numbers as ints, others to 4 decimals.
        if pd.api.types.is_float_dtype(shown[col]):
            shown[col] = shown[col].map(lambda v: "" if pd.isna(v) else (
                str(int(v)) if float(v).is_integer() else f"{v:.4f}".rstrip("0")))
        elif not pd.api.types.is_numeric_dtype(shown[col]):
            shown[col] = shown[col].map(lambda v: _tidy_ts(str(v)) if isinstance(v, (str, datetime)) else v)
    if len(shown) == 1 and len(shown.columns) > 6:
        # One wide row (a single meter's journey): 'column: value' lines -- in a 22-column CSV
        # the writer paired dates with the wrong headers (PMPM invoice <-> lumpsum collection).
        lines = [f"{c}: {'' if v is None or (not isinstance(v, str) and pd.isna(v)) else v}"
                 for c, v in shown.iloc[0].items()]
    else:
        lines = shown.to_csv(index=False).splitlines()
    budget = max_tokens or config.MAX_TOOL_RESULT_TOKENS
    kept, used = [], 0
    for i, line in enumerate(lines):
        cost = llm_gateway.estimate_tokens(line) - 8
        if i > 0 and used + cost > budget:
            break
        kept.append(line)
        used += cost
    return "\n".join(kept) + "\n", len(df)
