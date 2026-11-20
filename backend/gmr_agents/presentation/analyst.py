"""Analyst: facts about a result computed by code for the answer writer -- partial periods,
short weeks, the date window, per-category totals and the compact result preview.

Moved from gmr_agents/llm_pipeline.py unchanged.
"""

from __future__ import annotations
import itertools
import re
from datetime import datetime
import pandas as pd
from shared import config, llm_gateway
from gmr_agents.execution import executor
import logging
from gmr_agents.agents.common import _ANSWER_ROWS, _IST
from gmr_agents.presentation.blocks import _tidy_ts

log = logging.getLogger("gmr_agents.llm_pipeline")  # same logger name as before the split

# Two column rules, kept apart on purpose:
# _RATIO_COL_RE -- a ratio / average, never added up across anything (rows or periods).
# _RATE_COL_RE  -- also never summed ACROSS PERIODS: ratios plus stock levels, running balances, distinct counts
#                  (daily active technicians summed over a week are not technicians or technician-days).
#                  'pending' per zone still adds up across zones, so a share of a Total row uses _RATIO_COL_RE.
_RATIO_COL_RE = re.compile(r"pct|percent|rate|ratio|share|avg|average|mean|per_|_per", re.I)
_RATE_COL_RE = re.compile(r"pct|percent|%|rate|ratio|share|avg|average|mean|per_|_per|productivity|pace|"
                          r"days|cover|stock|balance|cumulative|to_date|so_far|remaining|pending|"
                          r"technician|active|distinct|unique", re.I)
# A count column whose name also hits _RATE_COL_RE ('meters_older_than_90_days' is a count, not days).
_COUNT_COL_RE = re.compile(r"(?:meters|meter_count|tickets|consumers|complaints|installations)(?:_|$)", re.I)


def _num(v: float) -> str:
    """Exact figure as text: whole numbers as ints, others to at most 4 decimals ('3.00001' -> '3', never '3.')."""
    return str(int(v)) if float(v).is_integer() else f"{v:.4f}".rstrip("0").rstrip(".")


def partial_periods(df: pd.DataFrame | None, spans: dict[str, str] | None = None) -> tuple[str | None, set[str], str]:
    """(period column, its partial labels, a PERIODS line) for a month- or day-wise RESULT.
    Only the current IST month (monthly rows) or today (daily rows) is partial -- the model
    otherwise guessed 'partial' for whichever month came last, e.g. a complete August.
    `spans` ({'YYYY-MM': '20-31 Jul'}, from month_spans) adds the partial FIRST / last month of a rolling
    window: 'last 4 months' asked 20 Nov = 19 Jul .. 19 Nov, so July holds 19-31 Jul only."""
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
            if spans:
                partial = {v for v in raw if v[:7] in spans}
                if partial:
                    named = "; ".join(f"{_month_name(v)} covers only {spans[v[:7]]}" for v in sorted(set(partial)))
                    return col, partial | current, (
                        f"PERIODS (checked by code): partial months -- {named}. Write each of them with its days in "
                        f"brackets, e.g. '{_month_name(min(partial))} ({spans[min(partial)[:7]]})'. Their totals cover "
                        "fewer days, so they are NOT comparable with a full month: keep them out of highest / lowest "
                        "claims and never call the change into or out of them a rise, fall, decline or drop (compare "
                        "per day if needed). Every other month in RESULT is complete.")
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


_THIS_MONTH_Q = re.compile(r"\b(this|current|curr?ent) month\b|month[- ]to[- ]date|\bmtd\b|\b(?:is|iss) mahine\b", re.I)
_LAST_MONTH_Q = re.compile(r"\b(last|previous|prev|prior|pichle|pichhle) month\b|\bmonth before\b", re.I)
_THIS_COL = re.compile(r"(?:^|_)(this|current|curr|cur|mtd)(?:_|$)", re.I)
_LAST_COL = re.compile(r"(?:^|_)(last|previous|prev|prior|lm)(?:_|$)", re.I)


def month_to_date(question: str, df: pd.DataFrame | None) -> str:
    """'This month vs last month' where RESULT has no period column (one wide row: installed_this_month,
    installed_last_month): the current month holds only the days up to yesterday, the last one is complete.
    12 Nov 2026: 'Compare installations this month vs last month' -> '24,211 vs 107,317, a decrease of 77%
    ... follows the month-end push pattern' (11 days vs 31). Per-day rates computed here make it like-for-like."""
    if not isinstance(df, pd.DataFrame) or df.empty or len(df) > 15:
        return ""
    if not (_THIS_MONTH_Q.search(question or "") and _LAST_MONTH_Q.search(question or "")):
        return ""
    if partial_periods(df)[0]:
        return ""  # month-wise rows: the PERIODS line already flags the current month
    now = datetime.now(_IST)
    days_now = now.day - 1  # complete days: data runs to yesterday
    if days_now < 1:
        return ""
    prev_end = now.replace(day=1) - pd.Timedelta(days=1)
    days_prev = prev_end.day
    head = (f"MONTH TO DATE (checked by code): this month ({now:%b %Y}) holds only {days_now} complete day(s) "
            f"(1-{days_now} {now:%b}); last month ({prev_end:%b %Y}) is complete ({days_prev} days). The raw totals are "
            "NOT like-for-like: never call their difference a drop / rise, and never explain it. Say the month is "
            "still in progress and compare per day")
    nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    pairs = []
    for c in nums:
        if not _THIS_COL.search(c):
            continue
        base = _THIS_COL.sub("_", c).strip("_")
        match = next((o for o in nums if o != c and _LAST_COL.search(o) and _LAST_COL.sub("_", o).strip("_") == base), None)
        if match:
            name = re.sub(r"(?:^|_)month(?:_|$)", "_", base).strip("_").replace("_", " ")
            pairs.append((name or "value", c, match))
    if not pairs:
        return head + f" (this month's figure / {days_now}, last month's / {days_prev})."
    label = next((c for c in df.columns if c not in nums), None)
    lines = []
    for _, row in df.head(10).iterrows():
        tag = f"{row[label]}: " if label is not None else ""
        for base, c_now, c_prev in pairs:
            a, b = row[c_now], row[c_prev]
            if pd.isna(a) or pd.isna(b):
                continue
            ra, rb = float(a) / days_now, float(b) / days_prev
            chg = f" ({(ra - rb) / rb * 100:+.1f}% per day)" if rb else ""
            lines.append(f"- {tag}{base}: {ra:,.1f} per day this month vs {rb:,.1f} per day last month{chg}")
    return head + ":\n" + "\n".join(lines) if lines else head + "."


_SCALE = {"million": 1e6, "mn": 1e6, "m": 1e6, "lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "crore": 1e7, "cr": 1e7,
          "thousand": 1e3, "k": 1e3, "billion": 1e9}
_STATED_RE = re.compile(r"(?<![\w.,-])(\d{1,3}(?:,\d{2,3})+|\d+(?:\.\d+)?)\s*(million|mn|lakhs?|lac|crore|cr|thousand|billion|k|m)?(?![\w%])",
                        re.I)


_THRESHOLD_RE = re.compile(r"(?:more|less|fewer|greater|higher|lower) than\s*$|\b(?:over|above|below|under|exceeding|"
                           r"exceeds?|cross|crossing|at least|at most|upto|up to|top|bottom|first|last|add|adding|hire|"
                           r"if|per|every|each|within|beyond|reach|target|to|of|ticket|consumer|number|no\.?|id)\s*$"
                           r"|[<>=]\s*$", re.I)


_CLEAR_ASK_RE = re.compile(r"\bwhen\b[^?]*\b(?:clear\w*|finish\w*|complet\w*|done|run(?:s)? out|exhaust\w*)|"
                           r"\bby when\b|\bwhat date\b|\bwhich date\b", re.I)
_DAYS_COL_RE = re.compile(r"^(?:\w+_)?days(?:_(?:required|needed|left|remaining|to_\w+))?$", re.I)


def clear_date(question: str, frames: list, today: pd.Timestamp | None = None) -> str:
    """'When will it be cleared?' with a one-row day count from the SQL: the date by code. 19 Nov: 37,917 pending at
    3,000 a day = 13 days was written 'by November 2' (2 December); the model adds days to today by hand."""
    if not _CLEAR_ASK_RE.search(question):
        return ""
    today = today or pd.Timestamp(datetime.now(_IST).date())
    for f in frames:
        if not isinstance(f, pd.DataFrame) or len(f) != 1:
            continue
        for c in f.columns:
            if not _DAYS_COL_RE.match(str(c)) or str(c).lower().endswith(("_ago", "_since", "_old")):
                continue
            v = pd.to_numeric(pd.Series([f.iloc[0][c]]), errors="coerce").iloc[0]
            if pd.notna(v) and 0 < float(v) <= 3650:
                n = int(-(-float(v) // 1))
                return (f"CLEAR DATE (computed by code): {n} days from today ({today:%d %b %Y}) = "
                        f"{today + pd.Timedelta(days=n):%d %b %Y}. Quote this date; never work out a date yourself.")
    return ""


def stated_figures(question: str, frames: list) -> str:
    """A large figure the user states as a fact ('Kashi has already installed 5 million meters, so how many are
    left?') that no value in the results comes near: the answer must correct it, not build on it. 12 Nov: the
    answer skipped the 5 million (the card: 1,384,855). Rupee amounts and years are left out."""
    q = str(question or "")
    stated = []
    for m in _STATED_RE.finditer(q):
        before = q[max(0, m.start() - 4):m.start()].lower()
        if "rs" in before or "₹" in before or re.match(r"(19|20)\d\d$", m.group(1)):
            continue
        if _THRESHOLD_RE.search(q[max(0, m.start() - 24):m.start()]) or (not m.group(2) and len(m.group(1)) >= 7
                                                                         and "," not in m.group(1)):
            continue  # 'more than 1,000', 'cross 45 lakh', 'ticket 2034060' -- a filter / target / id, not a claim
        val = float(m.group(1).replace(",", "")) * _SCALE.get((m.group(2) or "").lower(), 1)
        if val >= 1000:
            stated.append((m.group(0).strip(), val))
    if not stated:
        return ""
    vals = []
    for f in frames:
        if isinstance(f, pd.DataFrame) and not f.empty and len(f) <= 200:
            for c in f.columns:
                v = pd.to_numeric(f[c], errors="coerce").dropna()
                vals += [abs(float(x)) for x in v]
    if not vals:
        return ""
    off = [(t, v) for t, v in stated if not any(abs(x - v) <= 0.15 * v for x in vals)]
    if not off:
        return ""
    return ("PREMISE CHECK (checked by code): the question states " + ", ".join(t for t, _ in off) + " as a fact, but no "
            "figure in RESULT is close to it. If RESULT measures the same thing, open the answer by saying the stated "
            "figure does not match the data and give the real one -- never build on the stated figure.")


_WEEK_LABEL_RE = re.compile(r"^(\d{1,2})\s+([A-Za-z]{3})[a-z]*\s*[-–]\s*(\d{1,2})\s+([A-Za-z]{3})[a-z]*$")


def short_weeks(df: pd.DataFrame | None) -> tuple[str | None, dict[str, int]]:
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

# (sql, IST day) -> bounds: the writer, the facts and the table/chart labels all ask for one answer's window;
# Postgres evaluates it once. Bounded so a long-running server does not grow it.
_BOUNDS_CACHE: dict[tuple[str, str], object] = {}


def _window_bounds(plan: dict) -> tuple[pd.Timestamp, pd.Timestamp] | str | None:
    """(first day, last day) the SQL's date filter covers, evaluated by Postgres; 'all_time' for a day/month
    table with no date filter; None when unknown (Athena, no filter on a snapshot, an expression on a column)."""
    sql = plan.get("sql") or ""
    if plan.get("data_engine") != "postgres" or not sql:
        return None
    from gmr_agents.execution import ist_today
    key = (sql, str(ist_today.today()))
    if key in _BOUNDS_CACHE:
        return _BOUNDS_CACHE[key]
    got: object = None
    try:
        import sqlglot
        from sqlglot import exp

        root = sqlglot.parse_one(sql, read="postgres")
        lows, highs = [], []
        for node in root.find_all(exp.Between, exp.GTE, exp.GT, exp.LTE, exp.LT, exp.EQ):
            col = node.this
            # Any *_date column too: 19 Nov, O&M closed tickets filtered closed_date to the last 31 days and this note
            # said 'no date filter -- ALL TIME', so the answer called a 31-day figure 'all time'.
            if not isinstance(col, exp.Column) or not (col.name.lower() in _DATE_COLS or col.name.lower().endswith("_date")):
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
            got = "all_time"
        elif lows and highs:
            hi = f"GREATEST({', '.join(highs)})"
            if any(t.name.lower().endswith("_month_agg") for t in root.find_all(exp.Table)):
                # A month row covers its month -- up to today for the current one. 19 Nov: 'Installations by project
                # this month' was told '01 Nov .. 30 Nov 2026' and the answer said '01-30 Nov' on the 19th.
                hi = f"LEAST((DATE_TRUNC('month', {hi}) + INTERVAL '1 month - 1 day')::date, CURRENT_DATE)"
            row = executor._run_postgres(f"SELECT LEAST({', '.join(lows)}) AS lo, {hi} AS hi").iloc[0]
            got = (pd.Timestamp(row["lo"]), pd.Timestamp(row["hi"]))
    except Exception:  # a hint only -- never fail the answer over it
        log.debug("date window: could not evaluate", exc_info=True)
        return None
    if len(_BOUNDS_CACHE) >= 256:
        _BOUNDS_CACHE.pop(next(iter(_BOUNDS_CACHE)), None)  # oldest first (dicts keep insertion order)
    _BOUNDS_CACHE[key] = got
    return got


_MONTH_KEY_RE = re.compile(r"^(\d{4})-(0[1-9]|1[0-2])(?:-01)?(?:[ T]00:00(?::00)?)?$")


def _month_name(value: str) -> str:
    """'2026-07-01' / '2026-07' -> 'Jul 2026' (the table's month label)."""
    return f"{pd.Timestamp(value[:7] + '-01'):%b %Y}"


def month_spans(plan: dict, df: pd.DataFrame | None) -> dict[str, str]:
    """{'YYYY-MM': '19-31 Jul'} for the months of a month-wise RESULT that its date window covers only in part.
    User, 20 Nov: 'last 4 months' = rolling to yesterday -- July from the 19th, Aug-Oct full, November to the
    19th -- and a partial month says its days in brackets. Only the first and last month can be partial.
    A running balance (SUM .. OVER) is a month-END figure, so its first month is not partial."""
    if not isinstance(df, pd.DataFrame) or len(df) < 2:
        return {}
    bounds = _window_bounds(plan)
    if not isinstance(bounds, tuple):
        return {}
    lo, hi = bounds
    for col in [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]:
        vals = [str(v).strip() for v in df[col] if str(v).strip() and not str(v).strip().lower().startswith(("total", "grand total"))]
        if len(set(vals)) < 2 or not all(_MONTH_KEY_RE.match(v) for v in vals):
            continue
        months = sorted({v[:7] for v in vals})
        out: dict[str, str] = {}
        first, last = pd.Timestamp(months[0] + "-01"), pd.Timestamp(months[-1] + "-01")
        end_last = last + pd.offsets.MonthEnd(0)
        running = bool(re.search(r"\bover\s*\(", (plan.get("sql") or "").lower()))
        if (lo.year, lo.month) == (first.year, first.month) and lo.day > 1 and not running:
            out[months[0]] = f"{lo.day}-{(first + pd.offsets.MonthEnd(0)).day} {first:%b}"
        if (hi.year, hi.month) == (last.year, last.month) and hi.day < end_last.day:
            start = lo.day if months[-1] in out else 1
            out[months[-1]] = f"{start}-{hi.day} {last:%b}"
        return out
    return {}


def date_window(plan: dict, df: pd.DataFrame | None) -> str:
    """The date range the SQL really covered -- its period_start bounds evaluated by Postgres -- plus the
    row grain. The answer only sees the first rows of RESULT: live, 23 Oct, 'meters installed in last 3
    months' returned 92 day rows newest-first, and the answer called the window '22 Aug to 22 Oct'
    (it was 22 Jul .. 22 Oct) and a day 'the peak month'."""
    bounds = _window_bounds(plan)
    if bounds == "all_time":
        # No date asked = all time here, while the dashboard card defaults to the last 30 days.
        return ("DATE WINDOW (checked by code): no date filter -- the figure is ALL TIME to date. Say 'so far' / "
                "'all time' in the answer; the dashboard card shows the last 30 days by default, so the user "
                "may see a smaller number there.")
    if not isinstance(bounds, tuple):
        return ""
    window = f"{bounds[0]:%d %b %Y} .. {bounds[1]:%d %b %Y}"
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


_TOTAL_COL_RE = re.compile(r"^(?:total|grand_total|total_\w+|\w+_total)$", re.I)


def headline_column(df: pd.DataFrame | None) -> tuple[str, list[str]] | None:
    """(total column, the columns it is the exact sum of) -- e.g. total = consumer + dtr + feeder.
    19 Nov: 'MI progress for Varanasi this month' was answered as 9,011 (the consumer column, listed first)
    instead of the total 9,265; the 'total' column was in the facts but nothing said it was the headline."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cands = [c for c in nums if _TOTAL_COL_RE.match(str(c)) and not _RATE_COL_RE.search(str(c))]
    if not cands:
        return None
    values = df[nums].apply(pd.to_numeric, errors="coerce").fillna(0)  # once, not per combination tried
    for cand in cands:
        others = [c for c in nums if c != cand and not _RATE_COL_RE.search(str(c)) and not _TOTAL_COL_RE.match(str(c))][:6]
        target = values[cand]
        if len(others) < 2 or not target.abs().sum():
            continue
        # The largest set of columns whose row-wise sum is the total (consumer + dtr + feeder, not sat_done too).
        for size in range(len(others), 1, -1):
            for parts in itertools.combinations(others, size):
                if ((values[list(parts)].sum(axis=1) - target).abs() < 0.5).all():
                    return cand, list(parts)
    return None


def _total_row_lines(df: pd.DataFrame, rows: pd.DataFrame, is_total: pd.Series, labels: list, numeric: list) -> list[str]:
    """One Total row: the overall figure, the breakdown items and each item's share of the Total."""
    # The overall figure is the Total row, never the item that shares the place's name: 'agra
    # zone wise' once reported the AGRA zone row (193.96 days) as the Agra project (171.03).
    tot = df[is_total].iloc[0]
    nums = [c for c in numeric if pd.notna(tot[c])][:8]
    lines = ["- OVERALL (Total row = the whole scope asked): " + ", ".join(f"{c} {_num(float(tot[c]))}" for c in nums),
             f"- BREAKDOWN ITEMS ({len(rows)}; each is one part, never the overall): "
             + ", ".join(str(v) for v in rows[labels[0]].head(15))]
    # Shares of the Total, computed here so a detailed answer can say 'Kashi = 45.6%' / 'top 3 = 71%'
    # without the model doing arithmetic (18 Nov: 'in detail' answers only repeated the totals).
    share_col = next((c for c in nums if not _RATIO_COL_RE.search(str(c)) and not re.search("days", str(c), re.I)), None)
    if share_col and 2 <= len(rows) <= 40 and float(tot[share_col] or 0) > 0:
        vals = pd.to_numeric(rows[share_col], errors="coerce").fillna(0)
        order = vals.sort_values(ascending=False)
        top = [f"{rows.loc[i, labels[0]]} {100 * v / float(tot[share_col]):.1f}%" for i, v in order.head(5).items()]
        lines.append(f"- SHARE OF TOTAL by {share_col}: " + ", ".join(top)
                     + (f"; top 3 together {100 * order.head(3).sum() / float(tot[share_col]):.1f}%"
                        if len(rows) > 3 else ""))
    return lines


def _item_sum_lines(rows: pd.DataFrame, numeric: list, max_cols: int) -> list[str]:
    """No Total row: the overall figure is the sum of the items (counts / money only, never rates). 11 Nov: a
    per-project revenue-at-risk table had no Total row and the answer quoted Agra's row (Rs 111.9 cr) as the
    programme figure."""
    additive = [c for c in numeric
                if (not _RATE_COL_RE.search(str(c)) or _COUNT_COL_RE.match(str(c)))
                and not _RATIO_COL_RE.search(str(c))]
    lines = []
    for col in additive[:max_cols]:
        vals = pd.to_numeric(rows[col], errors="coerce")
        if vals.notna().sum() >= 2:
            lines.append(f"- {col}: sum across the {int(vals.notna().sum())} items = {_num(vals.sum())} "
                         "(no Total row -- this is the overall figure)")
    return lines


def _period_overall_lines(df: pd.DataFrame, labels: list, is_total: pd.Series, numeric: list, group: str,
                          max_cols: int) -> list[str]:
    """Month x Category with a Total row per month: the overall figure is the grand-total row
    (every label blank / Total), else the sum of the per-month Totals."""
    blank = pd.Series(True, index=df.index)
    for c in labels:
        blank &= df[c].isna() | df[c].astype(str).str.strip().str.lower().isin({"", "none", "nan", "total", "grand total"})
    per_period = df[is_total & ~blank]
    lines = []
    for col in [c for c in numeric if not _RATE_COL_RE.search(str(c))][:max_cols]:
        v = df.loc[blank, col].iloc[0] if blank.sum() == 1 else (
            pd.to_numeric(per_period[col], errors="coerce").sum() if len(per_period) > 1 else None)
        if v is not None and pd.notna(v):
            lines.append(f"- OVERALL {col} (all {group} values, all periods) = {_num(float(v))}")
    return lines


def _high_low_lines(rows: pd.DataFrame, label: str, groups: list, numeric: list, period_col: str | None,
                    partial: set, week_col: str | None, short: dict, max_cols: int, budget: int) -> list[str]:
    """Per group (or the whole table): the total over the periods, then the highest / lowest row of each
    numeric column -- partial periods and short weeks left out. Stops at a group once `budget` lines exist."""
    lines: list[str] = []
    for key, part in (rows.groupby(groups[0], sort=False) if groups else [(None, rows)]):
        if len(lines) >= budget:
            break
        body = part[~part[label].astype(str).str.strip().isin(partial)] if partial else part
        prefix = f"{key} " if key is not None else ""
        for col in numeric[:max_cols]:
            # Sum over the periods asked for, so 'how many ... monthwise' can lead with the total.
            if period_col and len(part) > 1 and not _RATE_COL_RE.search(str(col)):
                lines.append(f"- {prefix}{col}: total across the {len(part)} periods = "
                             f"{_num(pd.to_numeric(part[col], errors='coerce').sum())}")
            # A short week's COUNT is not comparable with full weeks (a rate such as productivity is).
            cmp = body
            if short and not _RATE_COL_RE.search(str(col)):
                cmp = body[~body[week_col].astype(str).str.strip().isin(short)]
            vals = pd.to_numeric(cmp[col], errors="coerce")
            if len(cmp) < 2 or vals.notna().sum() < 2 or vals.nunique() < 2:
                continue
            hi, lo = vals.idxmax(), vals.idxmin()
            lines.append(f"- {prefix}{col}: highest {cmp.at[hi, label]} ({_num(vals[hi])}), "
                         f"lowest {cmp.at[lo, label]} ({_num(vals[lo])})"
                         + (" (full weeks only)" if cmp is not body else ""))
    return lines


def result_facts(df: pd.DataFrame | None, max_cols: int = 4, spans: dict[str, str] | None = None) -> str:
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
        # 'Total Journey' / 'Total meters' rows are totals too, never a stage to rank (11 Nov: the journey
        # bottleneck was named as Total Journey, 186.76 days, instead of Installation to SAT, 86.2).
        is_total |= df[c].astype(str).str.strip().str.lower().str.match(r"(?:grand )?total\b")
    rows = df[~is_total]
    period_col, partial, _ = partial_periods(df, spans)
    week_col, short = short_weeks(df)
    if week_col and not period_col:
        period_col = week_col
    # Month x Category results: highest / lowest month per category, never across categories.
    label = period_col or labels[0]
    groups = [c for c in labels if c != label][:1] if period_col else []
    lines = []
    # A total column made of the other columns leads every list below and is named as the headline figure.
    head = headline_column(rows if len(rows) else df)
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if head:
        numeric = [head[0], *[c for c in numeric if c != head[0]]]
        lines.append(f"- HEADLINE FIGURE = column {head[0]} ({' + '.join(map(str, head[1]))}); the overall count "
                     f"(e.g. 'meters installed') is {head[0]}, the other columns are only its parts")
    if is_total.sum() == 1 and len(rows) >= 2:
        lines += _total_row_lines(df, rows, is_total, labels, numeric)
    if not is_total.any() and not period_col and 2 <= len(rows) <= 60:
        lines += _item_sum_lines(rows, numeric, max_cols)
    if period_col and groups and is_total.any():
        lines += _period_overall_lines(df, labels, is_total, numeric, groups[0], max_cols)
    lines += _high_low_lines(rows, label, groups, numeric, period_col, partial, week_col, short, max_cols,
                             budget=4 * max_cols - len(lines))
    return "\n".join(lines)


def result_preview(df: pd.DataFrame | None, max_tokens: int | None = None) -> tuple[str, int]:
    """Query rows for the answer writer as CSV: at most ANSWER_PREVIEW_ROWS rows and
    MAX_TOOL_RESULT_TOKENS tokens, cut between rows (never inside one)."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return "(no rows)", 0
    shown = df.head(_ANSWER_ROWS).copy()
    for col in shown.columns:
        # Exact values so the summary matches the cards: whole numbers as ints, others to 4 decimals.
        if pd.api.types.is_float_dtype(shown[col]):
            shown[col] = shown[col].map(lambda v: "" if pd.isna(v) else _num(v))
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
        # -8: estimate_tokens adds a per-message overhead a CSV line doesn't carry (an empty line would go negative).
        cost = max(llm_gateway.estimate_tokens(line) - 8, 1)
        if i > 0 and used + cost > budget:
            break
        kept.append(line)
        used += cost
    return "\n".join(kept) + "\n", len(df)
