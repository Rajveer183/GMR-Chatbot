"""Root-cause breakdown for 'why did X rise / fall' questions, computed by code.

The answer model reading two periods x 12 zones and 'naming the driver' picks the biggest
row, not the biggest CHANGE, and cannot tell how much of the total move each place explains.
Here the change per group is exact: previous vs current period, each group's share of the
total change, the groups that moved against the trend, and groups that appeared / dropped
out. The answer writer quotes these and turns the top drivers into actions. No LLM call;
nothing is added unless the question asks why / what drove / what changed.

Result shapes handled (the planner asks for one of them on 'why' questions):
    long   one row per (period, group) with a measure     -> the latest two periods compared
    wide   one row per group, one column per period        -> 'previous' / 'current' by name,
           (prev_* / curr_*, last_month / this_month,         or by the dates in the names
           sep_2026 / oct_2026 ...)
Rates (%, average, ratio ...) do not add up across groups: their per-group change is given in
points and ranked, without a 'share of the total change'.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from gmr_agents.presentation.stats import _split


def _fmt(v) -> str:
    v = float(v)
    return f"{v:,.0f}" if v.is_integer() else f"{v:,.2f}"

_WHY_RE = re.compile(
    r"\b(?:why|kyu|kyon|kyun|kyoon|kaaran|wajah|how come|reason(?:s)? (?:for|behind)|what (?:drove|caused|explains|changed|happened)|"
    r"driv(?:er|ers|ing|en) (?:of|behind)|root cause|behind the (?:drop|fall|rise|jump|dip|increase|decrease|change)|"
    r"(?:explain|break ?down) (?:the )?(?:drop|fall|rise|jump|dip|increase|decrease|change|gap|difference)|"
    r"(?:who|which \w+) (?:caused|drove|pulled|dragged|contributed))", re.I)
_RATE_RE = re.compile(r"pct|percent|%|rate|ratio|share|avg|average|mean|median|index|per_|_per\b|days|score|factor|loss", re.I)
_PREV_RE = re.compile(r"prev|previous|prior|last_?(?:month|week|year|period|quarter)|before|earlier|baseline|old|^p1_|_p1$|base_", re.I)
_CURR_RE = re.compile(r"curr|current|this_?(?:month|week|year|period|quarter)|latest|now|recent|after|^p2_|_p2$|new_", re.I)
_DELTA_RE = re.compile(r"change|delta|diff|growth|gap|variance|vs", re.I)


def asked(question: str) -> bool:
    return bool(_WHY_RE.search(question or ""))


def _name_date(col: str) -> pd.Timestamp | None:
    s = str(col).replace("_", " ").strip()
    if not re.search(r"\d{4}|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec", s, re.I):
        return None
    for fmt in ("%b %Y", "%B %Y", "%Y %m", "%Y-%m", "%Y %m %d", "%Y-%m-%d", "%b %y"):
        try:
            return pd.to_datetime(s, format=fmt)
        except (ValueError, TypeError):
            continue
    return None


def _wide_pair(rows: pd.DataFrame, nums: list[str]) -> tuple[str, str] | None:
    """(previous column, current column) of a one-row-per-group result, or None."""
    cand = [c for c in nums if not _DELTA_RE.search(str(c))]
    prev = [c for c in cand if _PREV_RE.search(str(c))]
    curr = [c for c in cand if _CURR_RE.search(str(c)) and c not in prev]
    if len(prev) == 2 and not curr:
        # 'last_week_closed' vs 'week_before_closed': next to 'before' / 'previous', 'last' is the current one.
        strong = [c for c in prev if re.search(r"before|prev|prior|earlier|baseline|old", str(c), re.I)]
        if len(strong) == 1:
            curr = [c for c in prev if c != strong[0]]
            prev = strong
    if len(prev) == 1 and len(curr) == 1:
        return prev[0], curr[0]
    dated = sorted(((d, c) for c in cand if (d := _name_date(c)) is not None), key=lambda x: x[0])
    if len(dated) >= 2:
        return dated[-2][1], dated[-1][1]
    return None


def _running(periods: pd.Series, today: pd.Timestamp) -> pd.Timestamp | None:
    """The latest period when it is still in progress (this month / today / this week), else None."""
    if len(periods) < 2:
        return None
    last, step = pd.Timestamp(periods.iloc[-1]), pd.Timestamp(periods.iloc[-1]) - pd.Timestamp(periods.iloc[-2])
    if step >= pd.Timedelta(days=27):
        return last if (last.year, last.month) == (today.year, today.month) else None
    if step <= pd.Timedelta(days=1):
        return last if last.normalize() >= today.normalize() else None
    return last if last.normalize() > today.normalize() - pd.Timedelta(days=7) else None


def _pick(nums: list[str], question: str) -> str:
    """The measure the question names (most shared words), else the first one."""
    words = set(re.findall(r"[a-z]{3,}", question.lower()))
    return max(nums, key=lambda c: (len(words & set(re.findall(r"[a-z]{3,}", str(c).lower()))), -nums.index(c)))


def _long_pair(rows: pd.DataFrame, label: str, dates: pd.Series, nums: list[str], today: pd.Timestamp, question: str):
    """Per-group (previous, current) frame from period x group rows, plus the period names."""
    groups = [c for c in rows.columns if c != label and not pd.api.types.is_numeric_dtype(rows[c])]
    if not groups or len(dates.unique()) < 2:
        return None
    # The running period, judged on the DISTINCT periods: per row (several groups a month) the median
    # gap is 0 days and a month would pass for daily data.
    partial = _running(pd.Series(sorted(dates.dropna().unique())), today)
    if partial is not None:
        keep = (dates != partial).to_numpy()
        rows, dates = rows[keep].reset_index(drop=True), dates[keep].reset_index(drop=True)
    periods = sorted(dates.dropna().unique())
    if len(periods) < 2:
        return None
    p0, p1 = periods[-2], periods[-1]
    g = groups[0]
    m = _pick(nums, question)
    frame = pd.DataFrame({"group": rows[g].astype(str), "d": dates.values, "v": pd.to_numeric(rows[m], errors="coerce")})
    prev = frame[frame.d == p0].groupby("group")["v"].sum(min_count=1)
    curr = frame[frame.d == p1].groupby("group")["v"].sum(min_count=1)
    both = pd.concat([prev.rename("prev"), curr.rename("curr")], axis=1)
    fmt = (lambda t: f"{pd.Timestamp(t):%b %Y}") if (pd.Timestamp(p1) - pd.Timestamp(p0)).days >= 27 \
        else (lambda t: f"{pd.Timestamp(t).date()}")
    return both, m, g, fmt(p0), fmt(p1), (fmt(partial) if partial is not None else None)


def _lines(both: pd.DataFrame, measure: str, group: str, p0: str, p1: str, partial: str | None = None) -> list[str]:
    both = both.copy()
    rate = bool(_RATE_RE.search(str(measure)))
    unit = " points" if rate and re.search(r"pct|percent|%|rate|share", str(measure), re.I) else ""
    appeared = [str(i) for i, r in both.iterrows() if pd.isna(r.prev) and pd.notna(r.curr)]
    dropped = [str(i) for i, r in both.iterrows() if pd.notna(r.prev) and pd.isna(r.curr)]
    if not rate:
        both = both.fillna(0)
    both["chg"] = both["curr"] - both["prev"]
    both = both.dropna(subset=["chg"])
    if len(both) < 2:
        return []
    out = [f"- compared: {measure} by {group}, {p0} vs {p1} ({len(both)} {group} values)"]
    if (both["chg"] == 0).all() and both["curr"].abs().sum() > 0:
        out.append(f"- every {group} has exactly the same figure in both periods: the source most likely has no "
                   "history (a snapshot read twice). Do NOT say 'no change' -- say the period-by-period history is "
                   "not stored and give the current figures")
        return out
    if partial:
        out.append(f"- {partial} is still in progress: left out, the last two complete periods are compared")
    if not rate:
        t0, t1 = both["prev"].sum(), both["curr"].sum()
        total = t1 - t0
        pct = f" ({total / t0 * 100:+.1f}%)" if t0 else ""
        out.append(f"- total: {_fmt(t0)} -> {_fmt(t1)}, change {total:+,.0f}{pct}")
        if total == 0:
            out.append("- no net change; the moves below cancel out")
            sign = 1.0
        else:
            sign = np.sign(total)
        same = both[np.sign(both["chg"]) == sign].sort_values("chg", key=lambda s: -s.abs())
        against = both[np.sign(both["chg"]) == -sign].sort_values("chg", key=lambda s: -s.abs())
        for name, r in same.head(3).iterrows():
            share = f", {r.chg / total * 100:.0f}% of the total change" if total else ""
            out.append(f"- driver: {name} {_fmt(r.prev)} -> {_fmt(r.curr)} ({r.chg:+,.0f}{share})")
        top3 = same.head(3)["chg"].sum()
        if total and len(same) > 3:
            out.append(f"- the top {min(3, len(same))} explain {top3 / total * 100:.0f}% of the total change "
                       f"({len(same)} {group} values moved the same way)")
        for name, r in against.head(2).iterrows():
            out.append(f"- moved against the trend: {name} {_fmt(r.prev)} -> {_fmt(r.curr)} ({r.chg:+,.0f})")
    else:
        both = both.sort_values("chg", key=lambda s: -s.abs())
        avg = both["chg"].mean()
        out.append(f"- {measure} is a rate: per-{group} changes do not add up to a total; "
                   f"average change across {group} values {avg:+.2f}{unit}")
        for name, r in both.head(3).iterrows():
            out.append(f"- biggest move: {name} {_fmt(r.prev)} -> {_fmt(r.curr)} ({r.chg:+.2f}{unit})")
        up, down = int((both["chg"] > 0).sum()), int((both["chg"] < 0).sum())
        out.append(f"- {up} rose, {down} fell")
    if appeared:
        out.append(f"- only in {p1}: {', '.join(appeared[:5])}")
    if dropped:
        out.append(f"- only in {p0} (no figure in {p1}): {', '.join(dropped[:5])}")
    return out


def breakdown(question: str, df: pd.DataFrame | None, today: pd.Timestamp | None = None) -> str:
    """The DRIVERS block for the answer prompt ('' unless the question asks why and the rows allow it)."""
    if not asked(question) or not isinstance(df, pd.DataFrame) or len(df) < 2:
        return ""
    today = today or pd.Timestamp.now().normalize()
    rows, label, dates, nums = _split(df)
    if dates is not None and label is not None and pd.api.types.is_numeric_dtype(rows[label]):
        # A numeric column whose NAME says month ('prev_month_sat') is a measure, not the period.
        dates = None
        label = next((c for c in rows.columns if not pd.api.types.is_numeric_dtype(rows[c])), None)
    if not nums:
        nums = [c for c in rows.columns if c != label and pd.api.types.is_numeric_dtype(rows[c])]
    if not nums:
        return ""
    lines: list[str] = []
    if dates is not None and label:
        got = _long_pair(rows, label, dates, nums, today, question)
        if got:
            lines = _lines(*got)
        elif not [c for c in rows.columns if c != label and not pd.api.types.is_numeric_dtype(rows[c])]:
            # Period totals only. 11 Nov: 'why did O&M closures fall in October' got two monthly totals and the
            # answer added 'the drop is not isolated to consumer categories' -- a split it never saw.
            return ("DRIVERS: RESULT has totals per period only, no split by place or category. State the change, "
                    "and say a breakdown (by zone / category) shows where it came from -- never say which place or "
                    "category caused it, or that it is / is not limited to one.")
    elif label:
        pair = _wide_pair(rows, [c for c in rows.columns if c != label and pd.api.types.is_numeric_dtype(rows[c])])
        if pair:
            both = rows.set_index(rows[label].astype(str))[[pair[0], pair[1]]]
            both.columns = ["prev", "curr"]
            both = both.apply(pd.to_numeric, errors="coerce")
            measure = re.sub(_PREV_RE, "", str(pair[0])).strip("_ ") or str(pair[0])
            lines = _lines(both, measure, label, str(pair[0]), str(pair[1]))
    if not lines:
        return ""
    head = ("DRIVERS (computed exactly by code from RESULT -- where the change happened, per group). Lead with the "
            "verdict, name the top drivers with these figures and their share of the change, mention any group that "
            "moved against the trend, and make the LAST insight one concrete action aimed at the top driver(s) (e.g. 'review crew deployment in X'). This shows WHERE the "
            "change came from, not the field-level cause: never invent one (staff, weather, supply) the data does not show.")
    return "\n".join([head, *lines[:12]])
