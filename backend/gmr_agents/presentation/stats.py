"""Statistics computed by code for analytical questions (correlation, regression, variance,
outliers, change points, moving average, autocorrelation, trend projection with an uncertainty
range, Monte Carlo / bootstrap ranges, PCA, Pareto concentration, tiers).

Why code: the answer model reading 30 rows and 'estimating' a correlation or a standard
deviation is guesswork that reads like a result. Here the figures are exact, the methods that
actually ran are named, and methods that did NOT run (SHAP, HMM, Granger, Bayesian networks ...)
are listed so the answer says so instead of pretending. Costs no LLM call; the extra prompt
text appears only when the question asks for analysis (see `wanted`).
"""

from __future__ import annotations

import math
import re
import warnings

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# What the question asks for
# --------------------------------------------------------------------------- #

_ASK = {
    # Words that are also KPI names are excluded: 'voltage deviation index', 'distribution transformer',
    # 'cluster' (= project), 'lagging zone', 'risk bucket', '80% SAT' must not switch analysis on.
    "corr": r"correlat|relationship|related to|relate[sd]? |go (?:together|with)|move together|associat|depend(?:s|ent|ency)? on|"
            r"regress|driv(?:e|es|er|ers|ing)\b|explain(?:s)? (?:most|the variation)|causal|caus(?:e|es|ed|ing)\b|factors?\b|"
            r"granger|cointegrat|leading indicator|lagging indicator|leads? .{0,25}or lag|feature importance|\bshap\b|attribut|"
            r"with (?:more|higher|fewer|lower|less) .{0,80}\b(?:also )?(?:show|have|get|see|record)s? (?:more|higher|fewer|lower|less)|"
            r"\balso (?:have|show|see)s? (?:more|higher)|\bhigher .{0,40}\bgo with\b",
    "spread": r"standard deviation|variation|decompos|\bstd\b|varian|variab|spread|volatil|dispersion|consisten|coefficient of variation|"
              r"heteroscedast|significan|confiden|uncertain|residual|error distribution|distribution(?! transformer)|stabl[ey]|stability",
    "outlier": r"outlier|anomal|unusual|abnormal|extreme|z.?score|deviat(?:e|es|ing)\b|deviation from",
    "forecast": r"forecast|predict|projection|\bproject (?:installations|the|demand|for)|extrapolat|next (?:week|month|quarter|\d)|"
                r"at (?:the|this) (?:current )?(?:pace|rate)|if the trend continues|long.?term|horizon|will (?:we|it|they)\b|"
                r"runs? out|by (?:december|january|end of)|backtest|forecast accuracy|how accurate",
    "moving": r"moving average|rolling average|smooth",
    "change": r"change.?point|regime|structural break|drift|when did .{0,40}chang|trend chang",
    "season": r"autocorrel|seasonal|weekday|day of (?:the )?week|periodic|(?:weekly|monthly|daily|time) pattern|"
              r"patterns? (?:in|do you|you see)|pattern recognition",
    "sim": r"monte carlo|simulat|probabilit|likel(?:y|ihood)\b|chance|odds|best.?case|worst.?case|stress test|scenario|"
           r"bayes|ensemble",
    "pca": r"\bpca\b|principal component|dimension(?:ality)? reduc|fewer dimensions|few key dimensions|components? explain",
    "pareto": r"pareto|80/20|20 ?% of (?:the )?\w+ (?:cause|hold|make|account)",
    # 'when did the anomaly start, peak and return to normal' (17 Nov probe: the first / last date of the data was
    # called the start / recovery, and Diwali + a Sunday were 'the worst day').
    "episode": r"\bwhen\b.{0,80}\b(?:start(?:ed|s)?|beg[ai]n|peak(?:ed|s)?|recover(?:ed|s)?|return(?:ed|s)?|end(?:ed|s)?|"
               r"worst|settled?)\b|(?:back|return(?:ed)?|returning|came back) to normal|\bpeak(?:ed)?\b.{0,60}\b(?:normal|"
               r"recover)|how long did .{0,50}\blast|\bepisodes?\b|\bspike .{0,30}\b(?:start|peak|recover)",
    "tiers": r"\bcluster(?:ing| (?:zones|circles|divisions|feeders|dts|technicians|meters|consumers)| by)\b|\bclusters\b|"
             r"segment|\btiers?\b|high, medium and low|group .{0,30}(?:high|low) performers",
}
_ASK_RE = {k: re.compile(v, re.I) for k, v in _ASK.items()}

# Named techniques this module does not implement: the answer must say they were not run.
_NOT_RUN = [
    (r"\bshap\b|feature importance", "SHAP / model feature importance (no ML model is trained)"),
    (r"hidden markov|\bhmm\b", "hidden Markov model"),
    (r"granger", "Granger causality test"),
    (r"cointegrat", "cointegration test"),
    (r"bayesian network", "Bayesian network"),
    (r"reinforcement", "reinforcement learning"),
    (r"game theory|\bthe game between\b|strategic interaction", "game-theory model"),
    (r"survival analysis|kaplan|time.?to.?failure|hazard", "survival / time-to-failure model (needs per-meter install and failure dates)"),
    (r"neural|deep learning|xgboost|random forest|machine learning", "machine-learning model"),
    (r"markov chain|transition probabilit|state transition", "Markov transition model (needs per-meter stage history, which is not stored)"),
]
_NOT_RUN = [(re.compile(p, re.I), name) for p, name in _NOT_RUN]


# What-if with a stated % change ('if productivity rises by 20%', 'supply drops by 50%', '10% of technicians are
# absent', 'if SAT pace doubles'). 11 Nov pilot: the SQL hard-coded '(3.0 - 2.7) * tech-days' for a 20% rise and
# answered 9,933 extra meters a month instead of ~20,000.
_DOWN = r"drop|drops|dropped|fall|falls|fell|decrease[sd]?|reduce[sd]?|cut|cuts|down|absent|fewer|lower|less|slower|shrinks?|stops?"
_UP = (r"rise|rises|rose|raise[sd]?|raising|increase[sd]?|increasing|grow|grows|improve[sd]?|improving|"
       r"boost(?:ed|s|ing)?|up|more|higher|faster|add|adds|added|adding")
_SCEN_RE = re.compile(rf"\b(?P<verb>{_DOWN}|{_UP})\b[^%.?]{{0,45}}?(?P<p>\d+(?:\.\d+)?)\s*%"
                      rf"|(?P<p2>\d+(?:\.\d+)?)\s*%[^.?]{{0,40}}?\b(?P<verb2>{_DOWN}|{_UP})\b", re.I)
# 'after increasing capacity by 15%': the triage rewrite of 'if capacity increases by 15%' (17 Nov smoke test).
_WHATIF_RE = re.compile(r"\bif\b|what.?if|suppose|assum|scenario|stress|simulat|sensitiv|would\b|"
                        r"\b(?:after|with|by) (?:increasing|raising|improving|boosting|adding|reducing|cutting)\b", re.I)


def scenario(question: str) -> float | None:
    """The relative change a what-if states (+0.2 for 'rises by 20%', -0.5 for 'drops by 50%'), else None."""
    q = str(question or "")
    if not _WHATIF_RE.search(q):
        return None
    if re.search(r"\bdoubles?\b|\bdoubled\b|\btwice\b", q, re.I):
        return 1.0
    if re.search(r"\bhalves?\b|\bhalved\b", q, re.I):
        return -0.5
    m = _SCEN_RE.search(q)
    if not m:
        return None
    p = float(m.group("p") or m.group("p2")) / 100
    verb = (m.group("verb") or m.group("verb2") or "").lower()
    return -p if re.fullmatch(_DOWN, verb) else p


# 'add 50 technicians' / 'if 30 technicians leave': output = productivity x technicians. 11 Nov: 'what if we add
# 50 technicians in Kashi' subtracted a 30-day install total from a per-day figure -> 'would REDUCE installs by 40,430'.
_ADD_TECH_RE = re.compile(r"\b(?P<verb>add|adding|added|hire|hiring|deploy|deploying|put|bring|remove|removing|lose|losing|"
                          r"cut|cutting|drop|dropping)\s+(?:another\s+)?(?P<n>\d+)\s+(?:more\s+|extra\s+|new\s+|additional\s+)?"
                          r"(?:technicians?|installers?|field staff)\b"
                          r"|\b(?P<n2>\d+)\s+(?:more\s+|extra\s+|additional\s+)?technicians?\s+(?P<verb2>leave|quit|are removed|join|are added)\b",
                          re.I)


def technicians_added(question: str) -> int | None:
    m = _ADD_TECH_RE.search(str(question or ""))
    if not m:
        return None
    n = int(m.group("n") or m.group("n2"))
    verb = (m.group("verb") or m.group("verb2") or "").lower()
    return -n if re.match(r"remov|los|cut|drop|leave|quit|are removed", verb) else n


_NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
              "ten": 10, "twelve": 12}
_HORIZON_RE = re.compile(r"\bnext\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|twelve)\s+(day|week|month|quarter|year)s?\b",
                         re.I)


def horizon(question: str, unit: str) -> int | None:
    """Periods of `unit` the question asks to project ('next 6 months' on a monthly series = 6; on a daily one = 180).
    11 Nov: 'project installations for the next 6 months' got a fixed 3-month total, the answer's own 6-month figure
    was then 'unsupported' and cut."""
    m = _HORIZON_RE.search(str(question or ""))
    if not m:
        return None
    n = int(m.group(1)) if m.group(1).isdigit() else _NUM_WORDS[m.group(1).lower()]
    days = {"day": 1, "week": 7, "month": 30, "quarter": 91, "year": 365}[m.group(2).lower()] * n
    per = {"day": 1, "week": 7, "month": 30}.get(unit, 30)
    return max(1, min(round(days / per), 400))


_ANY_HORIZON_RE = re.compile(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|twelve|a)\s+(day|week|month|quarter|year)s?\b", re.I)


def horizons(question: str) -> list[tuple[str, int]]:
    """Every horizon a multi-horizon forecast names ('1 week, 1 month and 3 months ahead') as (label, days)."""
    q = str(question or "")
    if not re.search(r"\bahead\b|\bnext\b|forecast|project|horizon", q, re.I):
        return []
    out = []
    for n, u in _ANY_HORIZON_RE.findall(q):
        k = 1 if n.lower() == "a" else int(n) if n.isdigit() else _NUM_WORDS[n.lower()]
        days = {"day": 1, "week": 7, "month": 30, "quarter": 91, "year": 365}[u.lower()] * k
        label = f"{k} {u.lower()}{'s' if k != 1 else ''}"
        if (label, days) not in out:
            out.append((label, days))
    return out


def _multi_horizon_line(name: str, y: np.ndarray, unit: str, question: str) -> list[str]:
    hs = horizons(question)
    if len(hs) < 2 or len(y) < 3:
        return []
    per = {"day": 1, "week": 7, "month": 30}.get(unit, 30)
    pw = min(len(y), {"day": 30, "week": 12, "month": 6}.get(unit, 6))
    avg = float(np.mean(y[-pw:]))
    sd = float(np.std(y[-pw:], ddof=1)) if pw > 1 else 0.0
    parts = []
    for label, days in hs:
        k = max(1, round(days / per))
        # Spread of a k-period total if periods vary independently: sd * sqrt(k).
        parts.append(f"next {label} about {_fmt(avg * k)} (95% range {_fmt(max(0, avg * k - 1.96 * sd * k ** .5))}.."
                     f"{_fmt(avg * k + 1.96 * sd * k ** .5)})")
    return [f"- {name} for each horizon asked, at the recent average of {_fmt(avg)} per {unit} (last {pw} {unit}s): "
            + "; ".join(parts)]


def wanted(question: str) -> set[str]:
    """The analyses a question asks for (empty for ordinary questions)."""
    q = str(question or "")
    got = {k for k, rx in _ASK_RE.items() if rx.search(q)}
    if scenario(q) is not None or technicians_added(q) is not None:
        got.add("whatif")
    return got


def _technician_lines(df: pd.DataFrame, n: int) -> list[str]:
    """Extra output from n more (or fewer) technicians at the current productivity of the overall row."""
    labels = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    row = df.iloc[0]
    if len(df) > 1:
        tot = pd.Series(False, index=df.index)
        for c in labels:
            tot |= df[c].astype(str).str.strip().str.lower().str.match(r"(?:grand )?total\b")
        if tot.sum() != 1:
            return []
        row = df[tot].iloc[0]
    cols = {str(c).lower(): c for c in df.columns}
    prod = next((row[c] for k, c in cols.items() if "productivity" in k or "per_tech" in k), None)
    if prod is None or pd.isna(prod):
        inst = next((row[c] for k, c in cols.items() if re.search(r"install", k) and not re.search(r"per|pct|rate", k)), None)
        days = next((row[c] for k, c in cols.items() if "technician_day" in k or "tech_day" in k), None)
        if inst is None or days is None or not days:
            return []
        prod = float(inst) / float(days)
    prod = float(prod)
    return [f"- SCENARIO {'+' if n > 0 else ''}{n} technicians at the current productivity of {_fmt(prod)} installs per "
            f"technician-day (computed by code; assumes they work every day like the current team): "
            f"{'+' if n > 0 else ''}{_fmt(n * prod)} installs per day, {'+' if n > 0 else ''}{_fmt(n * prod * 30)} per 30 days. "
            "Adding technicians never reduces output in this estimate."]


def _scenario_lines(df: pd.DataFrame, change: float) -> list[str]:
    """Every figure of the overall row (the single row, or the Total row) with the change applied."""
    labels = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    row = df.iloc[0]
    if len(df) > 1:
        tot = pd.Series(False, index=df.index)
        for c in labels:
            tot |= df[c].astype(str).str.strip().str.lower().str.match(r"(?:grand )?total\b")
        if tot.sum() != 1:
            return []
        row = df[tot].iloc[0]
    nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and pd.notna(row[c])][:6]
    if not nums:
        return []
    sign = "+" if change >= 0 else ""
    return [f"- SCENARIO {sign}{_fmt(change * 100)}% applied by code to each base figure (everything else unchanged): "
            + "; ".join(f"{c} {_fmt(float(row[c]))} -> {_fmt(float(row[c]) * (1 + change))} "
                        f"({sign}{_fmt(float(row[c]) * change)})" for c in nums)
            + ". Use the line(s) that match what was asked; convert per-day to per-month only by the days in RESULT."]


_BACKLOG_ASK_RE = re.compile(r"\bpending|backlog|\bclear(?:ed|ing)?\b|\bleft\b|remaining|outstanding|non.?sat|"
                             r"\bwork ?off\b|\bdays? to (?:finish|complete|clear)", re.I)
_BACKLOG_COL_RE = re.compile(r"pending|backlog|remaining|outstanding|non_?sat|balance|\bleft\b|_left|stock", re.I)
_RATE_COL_RE = re.compile(r"per_?day|daily|_rate$|per_day|/ ?day", re.I)
_RATE_WINDOW = {"day": 30, "week": 8, "month": 6}
_PER_DAY = {"day": 1.0, "week": 7.0, "month": 30.44}


def _single_figures(frames: list) -> dict:
    """Name -> value of every numeric column of the one-row frames (the 'current level' extra queries)."""
    out = {}
    for f in frames:
        if isinstance(f, pd.DataFrame) and len(f) == 1:
            for c in f.columns:
                v = pd.to_numeric(pd.Series([f.iloc[0][c]]), errors="coerce").iloc[0]
                if pd.notna(v):
                    out.setdefault(str(c), float(v))
    return out


def _clearance_lines(question: str, frames: list, change: float | None, added: int | None,
                     today: pd.Timestamp) -> list[str]:
    """What-if on a backlog: the scenario changes the RATE (installs / SAT per day), never the backlog itself.
    Days to clear before vs after, and what is left after the horizon. 17 Nov probe: '+15% daily capacity' answered
    'would be reduced' with no figure; '+10% SAT rate' took 10% OF THE BACKLOG (26,325) as the reduction."""
    if not _BACKLOG_ASK_RE.search(str(question)):
        return []
    singles = _single_figures(frames)
    backlog = next(((k, v) for k, v in singles.items() if _BACKLOG_COL_RE.search(k) and v > 0), None)
    if backlog is None:
        return []
    rate, how, prod = None, "", None
    for f in frames:
        if not isinstance(f, pd.DataFrame) or len(f) < 1:
            continue
        rows, _label, dates, _nums = _split(f)
        if dates is None or dates.isna().any():
            continue
        # Batch-wise measures (SAT) have rows only for months with a batch: 1 or 2 rows must still give a rate.
        cols = [c for c in rows.columns if c != _label and pd.api.types.is_numeric_dtype(rows[c])
                and not _BACKLOG_COL_RE.search(str(c)) and not re.search(r"(^|_)(id|rank|year)$", str(c), re.I)]
        if not cols:
            continue
        monthly = bool((dates.dt.day == 1).all()) and (dates.nunique() < 2 or pd.Series(sorted(dates.unique())).diff().min() >= pd.Timedelta(days=27))
        if dates.nunique() < 3 and not monthly:
            continue
        rows, dates, _ = _drop_partial(rows, dates, today) if len(rows) >= 4 else (rows, dates, None)
        y = pd.to_numeric(rows[cols[0]], errors="coerce").groupby(dates.to_numpy()).sum().sort_index()
        note = ""
        if monthly:
            # Months with no row are months with nothing done (no SAT batch): count them as 0 over the last 6
            # complete months, the window the planner is asked for.
            last = (today.replace(day=1) - pd.DateOffset(months=1)).normalize()
            y = y[y.index <= last]
            idx = pd.date_range(min([last - pd.DateOffset(months=_RATE_WINDOW["month"] - 1), *y.index]), last, freq="MS")
            missing = len(idx.difference(y.index))
            y = y.reindex(idx, fill_value=0.0)
            unit = "month"
            if missing:
                note = f" (the {missing} month(s) with no row counted as 0 -- the work is done in batches)"
        else:
            step = pd.Series(y.index).diff().median()
            unit = "day" if step <= pd.Timedelta(days=1) else "week" if step <= pd.Timedelta(days=8) else "month"
        if not len(y):
            continue
        w = min(len(y), _RATE_WINDOW[unit])
        rate = float(y.iloc[-w:].mean()) / _PER_DAY[unit]
        how = (f"{cols[0]} averaged {_fmt(float(y.iloc[-w:].mean()))} per {unit} over the last {w} complete {unit}s{note}"
               + (f" = {_fmt(rate)} per day" if unit != "day" else ""))
        break
    low = {k.lower(): v for k, v in singles.items()}
    prod = next((v for k, v in low.items() if "productivity" in k or "per_tech" in k), None)
    techs = next((v for k, v in low.items() if re.search(r"technicians?(?:_per_day)?$|tech_per_day|technicians_per_day", k)), None)
    if rate is None:
        r1 = next((v for k, v in singles.items() if _RATE_COL_RE.search(k) and not _BACKLOG_COL_RE.search(k)), None)
        if r1 is not None:
            rate, how = r1, f"current rate {_fmt(r1)} per day"
        elif prod is not None and techs is not None:
            rate, how = prod * techs, (f"{_fmt(techs)} technicians a day x {_fmt(prod)} per technician-day = "
                                       f"{_fmt(prod * techs)} per day")
    if rate is None or rate < 0:
        return []
    if change is not None:
        new_rate, what = rate * (1 + change), f"rate {'+' if change >= 0 else ''}{_fmt(change * 100)}%"
    elif added is not None and prod is not None:
        new_rate, what = rate + added * prod, f"{'+' if added > 0 else ''}{added} technicians x {_fmt(prod)} per technician-day"
    else:
        return []
    name, b = backlog
    hz = horizon(question, "day") or 30
    days = lambda r: f"{_fmt(b / r)} days" if r > 0 else "never (no progress at this rate)"
    now_left, new_left = max(0.0, b - rate * hz), max(0.0, b - new_rate * hz)
    out = [f"- BACKLOG WHAT-IF (computed by code; the change applies to the RATE, never to the backlog itself): "
           f"{name} now {_fmt(b)}; {how}. Scenario {what} -> {_fmt(new_rate)} per day "
           f"({'+' if new_rate >= rate else ''}{_fmt(new_rate - rate)} per day).",
           f"- Time to clear {_fmt(b)}: {days(rate)} at the current rate vs {days(new_rate)} with the change"
           + (f" ({_fmt(b / rate - b / new_rate)} days sooner)" if rate > 0 and new_rate > rate else
              f" ({_fmt(b / new_rate - b / rate)} days later)" if new_rate > 0 and rate > new_rate else "") + ".",
           f"- Next {hz} days: {_fmt(min(b, rate * hz))} cleared at the current rate (left {_fmt(now_left)}) vs "
           f"{_fmt(min(b, new_rate * hz))} with the change (left {_fmt(new_left)}) -- the change alone clears "
           f"{_fmt(min(b, new_rate * hz) - min(b, rate * hz))} more.",
           "- Assumes no new work joins the backlog and the rate holds; the backlog is today's snapshot, so new arrivals "
           "make the real clearing time longer -- say so."]
    if rate == 0:
        out.append("- The recent rate is 0, so a % change of it is still 0: say that the rate itself must restart first.")
    return out


# A relationship between measures -- not 'because', 'what causes X' / 'root causes' (root-cause questions keep their
# own format) or a 'relationship graph' of the network.
_EXPLICIT_CORR_RE = re.compile(r"(?<!auto)correlat|\brelationships?\b(?! graph)|\brelate[sd]? (?:to|with)\b|\bcausation\b|"
                               r"\bcausal|\bcause[sd]? (?:more|less|fewer|higher|lower)\b|associat|go together|"
                               r"move together|depend(?:s|ent)? on|granger|cointegrat", re.I)


def _lag_lines(rows: pd.DataFrame, nums: list[str], unit: str) -> list[str]:
    """Correlation over time done properly: same-period r, the best lead/lag, and change-on-change r. Two programme
    KPIs that both grow over the months correlate strongly whether or not one affects the other; the period-to-
    period changes show whether they move together beyond that shared growth (17 Nov probe)."""
    m = rows[nums[:4]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(m) < 8:
        return []
    out = []
    y = m.iloc[:, 0]
    for c in m.columns[1:3]:
        x = m[c]
        if x.std() == 0 or y.std() == 0:
            continue
        r0 = float(np.corrcoef(x, y)[0, 1])
        best = (0, r0)
        for k in (1, 2, 3):
            if len(m) - k < 6:
                break
            for lead, (a, bb) in ((k, (x.iloc[:-k], y.iloc[k:])), (-k, (y.iloc[:-k], x.iloc[k:]))):
                if a.std() and bb.std():
                    rk = float(np.corrcoef(a, bb)[0, 1])
                    if abs(rk) > abs(best[1]) + .05:
                        best = (lead, rk)
        dx, dy = x.diff().dropna(), y.diff().dropna()
        rd = float(np.corrcoef(dx, dy)[0, 1]) if dx.std() and dy.std() else float("nan")
        pd_ = _corr_p(rd, len(dx)) if np.isfinite(rd) else 1.0
        lead_txt = ("same period fits best" if best[0] == 0 else
                    f"best fit when {c} LEADS {y.name} by {best[0]} {unit}(s): r {_fmt(best[1])}" if best[0] > 0 else
                    f"best fit when {y.name} LEADS {c} by {-best[0]} {unit}(s): r {_fmt(best[1])}")
        shared = np.isfinite(rd) and abs(r0) >= .4 and abs(rd) < .3
        out.append(f"- over time ({len(m)} {unit}s) {y.name} vs {c}: same-{unit} r {_fmt(r0)} ({_strength(r0)}); {lead_txt}; "
                   f"change-on-change r {_fmt(rd)} (p {_fmt(pd_)}, {'significant' if pd_ < .05 else 'not significant'})"
                   + (" -- the level correlation is mostly SHARED GROWTH of the programme over time, not the two moving "
                      "together from one period to the next" if shared else "") + ".")
    return out


def _holidays() -> set:
    try:
        from gmr_agents.resolvers import festivals as _f
        days = {pd.Timestamp(d) for v in _f._LUNAR.values() for d in v}
        days |= {pd.Timestamp(year=y, month=m, day=d) for (m, d) in _f._FIXED.values() for y in range(2024, 2029)}
        return days
    except Exception:
        return set()


_DROP_RE = re.compile(r"\b(?:drop|drops|dropped|dip|dips|fall|fell|falls|declin\w*|slump\w*|slowdown|slow(?:ed)? down|"
                      r"low(?:er)?|decreas\w*|sag\w*)\b", re.I)
_SPIKE_RE = re.compile(r"\b(?:spike\w*|surge\w*|jump\w*|ris(?:e|es|ing)|rose|increas\w*|high(?:er)?|peak in|burst)\b", re.I)


def episode_direction(question: str) -> int:
    """-1 when the question asks about a drop, +1 for a spike, 0 for either / both."""
    q = str(question or "")
    down, up = bool(_DROP_RE.search(q)), bool(_SPIKE_RE.search(q))
    return -1 if down and not up else 1 if up and not down else 0


def _episode_lines(name: str, y: np.ndarray, dates: pd.Series, unit: str, want: int = 0) -> list[str]:
    """Abnormal episodes in one series: start, peak, end and the first period back inside the normal range.

    Normal = what the series' own recent history predicts: for daily data the median of the same weekday over the
    previous 4 weeks (so every Sunday is not a 'drop'), else the median of the previous 3 periods. Points found
    abnormal are replaced by their expected value before they feed later baselines, so a long episode does not
    become the new normal. Abnormal = beyond 3 robust standard deviations (MAD) of the residuals AND at least 15% /
    25% (daily) from normal. Festival days are reported apart, not as episodes."""
    n = len(y)
    if n < 10 or dates is None:
        return []
    hol = _holidays() if unit == "day" else set()
    d = pd.to_datetime(dates).reset_index(drop=True)

    def expected(clean: list, i: int) -> float | None:
        if unit == "day":
            same = [clean[i - k] for k in (7, 14, 21, 28) if i - k >= 0]
            if len(same) >= 2:
                return float(np.median(same))
            return float(np.median(clean[max(0, i - 7):i])) if i >= 3 else None
        return float(np.median(clean[max(0, i - 3):i])) if i >= 3 else None

    raw = [expected(list(y), i) for i in range(n)]
    res = np.array([y[i] - e for i, e in enumerate(raw) if e is not None])
    if len(res) < 6:
        return []
    scale = 1.4826 * float(np.median(np.abs(res - np.median(res)))) or float(np.std(res)) or 1.0
    rel = .25 if unit == "day" else .15
    clean, flag, exp, holiday_hits = list(map(float, y)), [0] * n, [None] * n, []
    for i in range(n):
        e = expected(clean, i)
        exp[i] = e
        if e is None:
            continue
        dev = y[i] - e
        # A drop can be at most 100% of normal: when 3 robust std devs exceed normal itself (volatile daily series),
        # a fall to half of normal still counts -- else a day at ~0 could never be a drop.
        thr = max(rel * abs(e), 3 * scale if dev > 0 else min(3 * scale, .5 * abs(e)))
        if abs(dev) > thr:
            if any(abs((d[i] - h).days) <= 1 for h in hol):
                holiday_hits.append(i)
            else:
                flag[i] = 1 if dev > 0 else -1
            clean[i] = e
    lab = lambda i: str(d[i].date()) if unit == "day" else d[i].strftime("%b %Y") if unit == "month" else str(d[i].date())
    eps, i = [], 0
    while i < n:
        if not flag[i]:
            i += 1
            continue
        s, sign, j = i, flag[i], i
        # one normal period inside a run does not end it
        while j + 1 < n and (flag[j + 1] == sign or (j + 2 < n and not flag[j + 1] and flag[j + 2] == sign)):
            j += 1
        idx = [k for k in range(s, j + 1) if flag[k] == sign]
        pk = max(idx, key=lambda k: abs(y[k] - exp[k]))
        eps.append((len(idx) >= 2, abs(y[pk] - exp[pk]), s, pk, j, sign))
        i = j + 1
    other = []
    if want:
        # 'When did the DROP start': only below-normal episodes count. 17 Nov probe: with no abnormal drop found the
        # answer picked 15 Aug (Independence Day) as the start and Diwali as the worst day.
        other = [e for e in eps if e[-1] != want]
        eps = [e for e in eps if e[-1] == want]
    band = f"normal = {'the same weekday over the previous 4 weeks' if unit == 'day' else 'the previous 3 ' + unit + 's'}" \
           f" (median), abnormal = more than {_fmt(3 * scale)} above it or below it by {_fmt(3 * scale)} or half of normal " \
           f"(whichever is smaller), and always at least {int(rel * 100)}% away"
    if not eps:
        side = {-1: "BELOW", 1: "ABOVE"}.get(want, "outside")
        out = [f"- EPISODES {name}: no {unit} {side} the normal range in {lab(0)}..{lab(n - 1)} ({band})"
               + (f"; there were {len(other)} episode(s) on the other side, not what was asked" if other else "")
               + (". Low days were Sundays / festival days, which are normal for this series" if want == -1 else "") + "."]
    else:
        multi = sum(1 for e in eps if e[0])
        out = [f"- EPISODES {name} ({len(eps)} found -- {multi} lasting 2+ {unit}s, {len(eps) - multi} single-{unit} blips; "
               f"{band}; the {min(3, len(eps))} most important listed (longer ones first, then largest), oldest first):"]
        top = sorted(eps, key=lambda e: (e[0], e[1]), reverse=True)[:3]
        for _, _, s, pk, j, sign in sorted(top, key=lambda e: e[2]):
            back = next((k for k in range(j + 1, n) if not flag[k]), None)
            out.append(f"  * {'above' if sign > 0 else 'below'} normal: started {lab(s)}, peaked {lab(pk)} at {_fmt(y[pk])} "
                       f"vs normal {_fmt(exp[pk])} ({'+' if y[pk] >= exp[pk] else ''}{_fmt((y[pk] - exp[pk]) / exp[pk] * 100) if exp[pk] else 'n/a'}%), "
                       f"last abnormal {lab(j)}, "
                       + (f"back to normal {lab(back)} ({(j - s + 1)} {unit}(s) long)" if back is not None
                          else "NOT yet back to normal at the latest data"))
    if holiday_hits:
        out.append(f"- {name} festival / holiday days outside the range (expected, not anomalies): "
                   + ", ".join(f"{lab(k)} {_fmt(y[k])}" for k in holiday_hits[:5]))
    return out


def not_run(question: str) -> list[str]:
    return [name for rx, name in _NOT_RUN if rx.search(str(question or ""))]


# Shown to the SQL writer only when `wanted` is non-empty: the rows must be a series / item list
# the code can work on (a single pre-aggregated number gives it nothing).
PLAN_HINT = (
    "ANALYSIS ROWS: correlations, variance / spread, outliers, change points, moving averages, forecasts with a "
    "range, probabilities / simulations, PCA and tiers are computed BY CODE from the rows you return -- so return "
    "the raw series: one row per item (zone / circle / technician / feeder / OEM ...) or per period (day / week / "
    "month, complete periods, today excluded), each measure as its own numeric column, no Total row, no LIMIT under "
    "200 unless a top N is asked. 'A vs B relationship / does A drive B' = A and B side by side per item or per "
    "period, the measure asked about FIRST. Forecast / probability / Monte Carlo = the complete daily (last 60-90 days) or monthly history, plus "
    "the current level (stock, pending, target gap) in an extra query when the question needs it. What-if / "
    "scenario / stress test with a % change: return the BASE figures over a stated window (e.g. installs and "
    "technician-days in the last 30 days) -- the change is applied by code. Never put assumed or invented constants "
    "in the SQL (only numbers the question states).")


# --------------------------------------------------------------------------- #
# Shape helpers
# --------------------------------------------------------------------------- #

def _fmt(v: float) -> str:
    if v is None or not np.isfinite(v):
        return "n/a"
    a = abs(v)
    if a >= 1000:
        return f"{v:,.0f}"
    if a >= 10:
        return f"{v:,.1f}"
    if 0 < a < 0.01:
        return f"{v:.2g}"
    return f"{v:.3f}".rstrip("0").rstrip(".") if a < 1 else f"{v:.2f}"


def _phi(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _corr_p(r: float, n: int) -> float:
    """Two-sided p-value of a Pearson r via the Fisher z approximation."""
    if n < 4 or not np.isfinite(r) or abs(r) >= 1:
        return 0.0 if abs(r) >= 1 else 1.0
    z = math.atanh(r) * math.sqrt(n - 3)
    return 2 * (1 - _phi(abs(z)))


def _strength(r: float) -> str:
    a = abs(r)
    word = "very strong" if a >= .8 else "strong" if a >= .6 else "moderate" if a >= .4 else "weak" if a >= .2 else "no real"
    return f"{word} {'positive' if r > 0 else 'negative'}" if a >= .2 else word


def _dates(s: pd.Series) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = pd.to_datetime(s.astype(str), errors="coerce", format="mixed")
        if d.notna().mean() <= .8:
            d = pd.to_datetime(s.astype(str).str.slice(0, 10), errors="coerce", format="mixed")
    return d


def _split(df: pd.DataFrame):
    """(rows without Total, label column or None, date series or None, numeric columns)."""
    labels = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    is_total = pd.Series(False, index=df.index)
    for c in labels:
        low = df[c].astype(str).str.strip().str.lower()
        is_total |= low.isin({"overall", "all"}) | low.str.match(r"(?:grand )?total\b")
    rows = df[~is_total].reset_index(drop=True)
    dates, label = None, None
    for c in labels + [c for c in df.columns if c not in labels]:
        s = rows[c]
        iso = not pd.api.types.is_numeric_dtype(s) and s.astype(str).str.match(r"\d{4}-\d{2}(?:-\d{2})?\b").mean() > .8
        if pd.api.types.is_datetime64_any_dtype(s) or iso or re.search(r"date|day|week|month|period|year", str(c), re.I):
            d = _dates(s)
            if len(d) and d.notna().mean() > .8:
                dates, label = d, c
                break
    if label is None and labels:
        label = labels[0]
    nums = [c for c in rows.columns if c != label and pd.api.types.is_numeric_dtype(rows[c])
            and rows[c].notna().sum() >= 3 and rows[c].nunique() > 1
            and not re.search(r"(^|_)(id|rank|row_?num|year)$", str(c), re.I)]
    return rows, label, dates, nums


def _drop_partial(rows: pd.DataFrame, dates: pd.Series | None, today: pd.Timestamp):
    """Leave out today / the running month: a part-period is not a low point."""
    if dates is None or len(rows) < 4:
        return rows, dates, None
    last = dates.max()
    step = dates.sort_values().diff().median()
    partial = None
    if step is not None and pd.notna(step):
        if step >= pd.Timedelta(days=27) and last.year == today.year and last.month == today.month:
            partial = last
        elif step <= pd.Timedelta(days=1) and last.normalize() >= today.normalize():
            partial = last
        elif pd.Timedelta(days=1) < step <= pd.Timedelta(days=8) and last.normalize() > today.normalize() - pd.Timedelta(days=7):
            partial = last
    if partial is None:
        return rows, dates, None
    keep = dates != partial
    return rows[keep].reset_index(drop=True), dates[keep].reset_index(drop=True), partial


# --------------------------------------------------------------------------- #
# Analyses
# --------------------------------------------------------------------------- #

_RECENT = {"day": 30, "week": 12, "month": 6}


def _forecast(y: np.ndarray, unit: str, h: int) -> dict:
    """The projection _series_stats describes, as numbers: a straight line through the recent window when it
    fits (R^2 >= 0.3 and never negative), else the recent average. lo / hi = 95% prediction range per period."""
    n = len(y)
    pw = min(n, _RECENT.get(unit, 6))
    yw = y[-pw:]
    xw = np.arange(n - pw, n, dtype=float)
    avg = float(yw.mean())
    xs = np.arange(n, n + h, dtype=float)
    if pw >= 3:
        ps, pi = np.polyfit(xw, yw, 1)
        fit = ps * xw + pi
        sse = float(((yw - fit) ** 2).sum())
        r2 = 1 - sse / (float(((yw - avg) ** 2).sum()) or 1)
        fut = ps * xs + pi
        if r2 >= .3 and (fut >= 0).all():
            se = math.sqrt(sse / max(1, pw - 2))
            sxx = float(((xw - xw.mean()) ** 2).sum()) or 1.0
            # Prediction interval of a straight-line fit: wider the further ahead.
            band = 1.96 * se * np.sqrt(1 + 1 / pw + (xs - xw.mean()) ** 2 / sxx)
            return {"method": "trend", "pw": pw, "fut": fut, "lo": np.maximum(0, fut - band), "hi": fut + band,
                    "tot_band": 1.96 * se * math.sqrt(h)}
    sd = float(np.std(yw, ddof=1)) if pw > 1 else 0.0
    fut = np.full(h, avg)
    return {"method": "average", "pw": pw, "fut": fut, "lo": np.maximum(0, fut - 1.96 * sd), "hi": fut + 1.96 * sd,
            "tot_band": 1.96 * sd * math.sqrt(h)}


def _next_labels(dates: pd.Series | None, unit: str, h: int) -> list[str]:
    if dates is None or not len(dates):
        return [f"+{i + 1}" for i in range(h)]
    last = pd.Timestamp(dates.iloc[-1])
    if unit == "month":
        return [(last + pd.DateOffset(months=i + 1)).strftime("%b %Y") for i in range(h)]
    step = pd.Timedelta(days=7 if unit == "week" else 1)
    return [str((last + step * (i + 1)).date()) for i in range(h)]


def _interval_lines(name: str, y: np.ndarray, unit: str, h: int, dates: pd.Series | None) -> list[str]:
    """One forecast per future period with its 95% range (only for short horizons -- a 30-day list is noise),
    and the range of the h-period total."""
    if len(y) < 4:
        return []
    f = _forecast(y, unit, h)
    tot = float(f["fut"].sum())
    out = [f"- {name} forecast total for the next {h} {unit}s: {_fmt(tot)} (95% range {_fmt(max(0, tot - f['tot_band']))}"
           f"..{_fmt(tot + f['tot_band'])}; method: {'recent straight-line trend' if f['method'] == 'trend' else 'recent average'}"
           f" over the last {f['pw']} {unit}s)"]
    if h <= 6:
        labs = _next_labels(dates, unit, h)
        out.append(f"- {name} forecast per {unit} with 95% range: "
                   + "; ".join(f"{labs[i]} {_fmt(f['fut'][i])} ({_fmt(f['lo'][i])}..{_fmt(f['hi'][i])})" for i in range(h)))
    return out


def _backtest_lines(name: str, y: np.ndarray, unit: str, h: int, dates: pd.Series | None) -> list[str]:
    """Historical validation: the same method run as if standing k periods ago (up to 3 such starting points),
    its forecast compared with what actually happened. Says how far to trust the projection above."""
    n = len(y)
    k = max(1, min(h, n // 4, 30 if unit == "day" else 6))
    origins = [n - j * k for j in (1, 2, 3) if n - j * k >= 4]
    if not origins:
        return []
    errs, cover, tot_errs = [], [], []
    for o in origins:
        f = _forecast(y[:o], unit, k)
        act = y[o:o + k]
        m = act > 0
        if m.any():
            errs += list(np.abs(act[m] - f["fut"][m]) / act[m])
        cover += list((act >= f["lo"]) & (act <= f["hi"]))
        if act.sum() > 0:
            tot_errs.append((float(f["fut"].sum()) - float(act.sum())) / float(act.sum()))
    if not errs or not tot_errs:
        return []
    o = origins[0]
    f0 = _forecast(y[:o], unit, k)
    when = str(pd.Timestamp(dates.iloc[o - 1]).date()) if dates is not None and len(dates) >= o else f"period {o}"
    mape = float(np.mean(errs)) * 100
    grade = "good" if mape <= 10 else "fair" if mape <= 25 else "poor"
    return [f"- {name} historical check (backtest): the same method, run as if on {when}, forecast "
            f"{_fmt(float(f0['fut'].sum()))} for the next {k} {unit}s; actual was {_fmt(float(y[o:o + k].sum()))} "
            f"({'+' if tot_errs[0] >= 0 else ''}{_fmt(tot_errs[0] * 100)}%). Over {len(origins)} such test(s): typical "
            f"per-{unit} error {_fmt(mape)}% ({grade} accuracy), total off by {_fmt(float(np.mean(np.abs(tot_errs))) * 100)}% "
            f"on average, {int(np.sum(cover))} of {len(cover)} actual {unit}s fell inside the 95% range."]


def _series_stats(name: str, y: np.ndarray, dates: pd.Series | None, asks: set[str], unit: str,
                  horizon: int | None = None) -> list[str]:
    out = []
    n = len(y)
    mean, sd = float(np.mean(y)), float(np.std(y, ddof=1)) if n > 1 else 0.0
    med = float(np.median(y))
    cv = sd / mean * 100 if mean else float("nan")
    lab = (lambda i: str(dates.iloc[i].date())) if dates is not None else (lambda i: f"#{i + 1}")
    out.append(f"- {name}: {n} {unit}s, mean {_fmt(mean)}, median {_fmt(med)}, std dev {_fmt(sd)}, "
               f"coefficient of variation {_fmt(cv)}%, min {_fmt(y.min())} ({lab(int(y.argmin()))}), "
               f"max {_fmt(y.max())} ({lab(int(y.argmax()))})")
    x = np.arange(n, dtype=float)
    win = min(n, 30 if unit == "day" else 12)  # recent periods re-sampled by the simulation
    if n >= 4:
        slope, icpt = np.polyfit(x, y, 1)
        fit = slope * x + icpt
        resid = y - fit
        ss_tot = float(((y - mean) ** 2).sum())
        r2 = 1 - float((resid ** 2).sum()) / ss_tot if ss_tot else 0.0
        se = float(np.sqrt((resid ** 2).sum() / max(1, n - 2)))
        out.append(f"- {name} linear trend: {'+' if slope >= 0 else ''}{_fmt(slope)} per {unit} "
                   f"(R^2 {_fmt(r2)}; {'clear' if r2 >= .5 else 'weak' if r2 >= .2 else 'no clear'} trend), "
                   f"first-to-last change {_fmt(y[-1] - y[0])} ({_fmt((y[-1] - y[0]) / y[0] * 100) if y[0] else 'n/a'}%)")
        if asks & {"spread", "outlier"}:
            out.append(f"- {name} residuals around the trend: std {_fmt(se)}, largest above {_fmt(resid.max())} "
                       f"({lab(int(resid.argmax()))}), largest below {_fmt(resid.min())} ({lab(int(resid.argmin()))})")
            half = n // 2
            if "spread" in asks and half >= 3:
                s1, s2 = float(np.std(y[:half], ddof=1)), float(np.std(y[half:], ddof=1))
                out.append(f"- {name} variability first half vs second half: std {_fmt(s1)} vs {_fmt(s2)} "
                           f"({'rising' if s2 > 1.25 * s1 else 'falling' if s1 > 1.25 * s2 else 'similar'} spread)")
        if asks & {"forecast", "sim"}:
            h = horizon or {"day": 30, "week": 4, "month": 3}.get(unit, 3)
            # Project from the RECENT trend (last 30 days / 12 weeks / 6 months): a line through two years of
            # history projected 212k installs next month when the last 6 months ran at ~74k.
            pw = min(n, {"day": 30, "week": 12, "month": 6}.get(unit, 6))
            ps, pi = np.polyfit(np.arange(n - pw, n, dtype=float), y[-pw:], 1)
            fut = ps * np.arange(n, n + h) + pi
            yw = y[-pw:]
            fitw = ps * np.arange(n - pw, n) + pi
            se = float(np.sqrt(((yw - fitw) ** 2).sum() / max(1, pw - 2)))
            r2w = 1 - float(((yw - fitw) ** 2).sum()) / (float(((yw - yw.mean()) ** 2).sum()) or 1)
            avg = float(yw.mean())
            if r2w >= .3 and (fut >= 0).all():
                out.append(f"- {name} projection (straight-line trend over the last {pw} {unit}s, R^2 {_fmt(r2w)}, assumes it "
                           f"holds): next {unit} {_fmt(fut[0])} (about +/-{_fmt(1.96 * se)} at 95%); next {h} {unit}s total "
                           f"{_fmt(fut.sum())}; at the recent average ({_fmt(avg)} per {unit}) the next {h} {unit}s total {_fmt(avg * h)}")
            else:
                # A weak line (day-of-week swings, a few odd days) extrapolated 30 days says nothing: 11 Nov, a 30-day
                # daily fit projected 19k installs for the next month against a 100k Monte Carlo median.
                sd_w = float(np.std(yw, ddof=1)) if pw > 1 else 0.0
                out.append(f"- {name} projection: no reliable trend over the last {pw} {unit}s (R^2 {_fmt(r2w)}), so the best "
                           f"estimate is the recent average -- {_fmt(avg)} per {unit} (one {unit} usually within "
                           f"{_fmt(max(0, avg - 1.96 * sd_w))}..{_fmt(avg + 1.96 * sd_w)}); next {h} {unit}s total about {_fmt(avg * h)}")
            out += _interval_lines(name, y, unit, h, dates)
            out += _backtest_lines(name, y, unit, h, dates)
            if "sim" in asks:
                rng = np.random.default_rng(7)
                draws = rng.choice(y[-win:], size=(5000, h), replace=True).sum(axis=1)
                p10, p50, p90 = np.percentile(draws, [10, 50, 90])
                out.append(f"- {name} Monte Carlo (5,000 bootstrap runs re-sampling the last {win} {unit}s): "
                           f"next {h} {unit}s total P10 {_fmt(p10)}, median {_fmt(p50)}, P90 {_fmt(p90)} "
                           f"(worst case {_fmt(draws.min())}, best case {_fmt(draws.max())}); "
                           f"next {unit} alone P10 {_fmt(np.percentile(y[-win:], 10))} .. P90 {_fmt(np.percentile(y[-win:], 90))}")
        if "outlier" in asks or "season" not in asks:
            z = (y - mean) / sd if sd else np.zeros(n)
            odd = [i for i in np.argsort(-np.abs(z)) if abs(z[i]) >= 2][:5]
            out.append(f"- {name} unusual {unit}s (|z| >= 2): "
                       + (", ".join(f"{lab(i)} {_fmt(y[i])} (z {_fmt(z[i])})" for i in odd) if odd else "none"))
        if "moving" in asks and n >= 7:
            w = 7 if unit == "day" else 3
            ma = pd.Series(y).rolling(w).mean().to_numpy()
            out.append(f"- {name} {w}-{unit} moving average: latest {_fmt(ma[-1])}, {w} {unit}s earlier {_fmt(ma[-1 - w]) if n > 2 * w - 1 else 'n/a'}, "
                       f"highest {_fmt(np.nanmax(ma))}, lowest {_fmt(np.nanmin(ma))}")
        if "change" in asks and n >= 8:
            best = None
            for k in range(3, n - 2):
                sse = ((y[:k] - y[:k].mean()) ** 2).sum() + ((y[k:] - y[k:].mean()) ** 2).sum()
                if best is None or sse < best[0]:
                    best = (sse, k)
            k = best[1]
            a, b = y[:k].mean(), y[k:].mean()
            pooled = math.sqrt(((np.var(y[:k], ddof=1) / k) + (np.var(y[k:], ddof=1) / (n - k))) or 1e-9)
            t = (b - a) / pooled if pooled else 0
            out.append(f"- {name} change point (best single split by least squares): from {lab(k)} the level moves from "
                       f"{_fmt(a)} to {_fmt(b)} per {unit} ({_fmt((b - a) / a * 100) if a else 'n/a'}%; "
                       f"{'significant' if abs(t) >= 2 else 'not significant'}, t {_fmt(t)})")
        if "season" in asks and n >= 8:
            yc = y - mean
            den = float((yc ** 2).sum()) or 1
            acf = {lag: float((yc[lag:] * yc[:-lag]).sum()) / den for lag in (1, 7, 12) if lag < n - 2}
            out.append(f"- {name} autocorrelation: " + ", ".join(f"lag {k} {_fmt(v)}" for k, v in acf.items())
                       + f" (|value| above {_fmt(2 / math.sqrt(n))} is meaningful)")
            if dates is not None and unit == "day" and n >= 14:
                wd = pd.Series(y).groupby(dates.dt.day_name().to_numpy()).mean()
                order = wd.sort_values()
                out.append(f"- {name} by weekday (mean): " + ", ".join(f"{d} {_fmt(v)}" for d, v in wd.items())
                           + f" -- lowest {order.index[0]}, highest {order.index[-1]}")
    return out


def _cross_stats(rows: pd.DataFrame, label: str | None, nums: list[str], asks: set[str]) -> list[str]:
    out = []
    names = rows[label].astype(str) if label else pd.Series([f"#{i + 1}" for i in range(len(rows))])
    for c in nums[:3]:
        v = pd.to_numeric(rows[c], errors="coerce")
        ok = v.notna()
        y = v[ok].to_numpy(float)
        if len(y) < 3:
            continue
        mean, sd = y.mean(), y.std(ddof=1)
        q1, q3 = np.percentile(y, [25, 75])
        out.append(f"- {c} across {len(y)} items: mean {_fmt(mean)}, median {_fmt(np.median(y))}, std dev {_fmt(sd)}, "
                   f"CV {_fmt(sd / mean * 100) if mean else 'n/a'}%, IQR {_fmt(q1)}..{_fmt(q3)}")
        if "corr" in asks and len(y) < 5 and c == nums[0]:
            out.append(f"- only {len(y)} items ({', '.join(map(str, names[ok].tolist()))}) -- too few to measure a "
                       "correlation: never say these measures correlate / associate / track each other on this basis; "
                       "compare the items side by side and say a period-by-period series is needed to measure it.")
        if asks & {"outlier", "spread"} or not asks & {"corr", "pca"}:
            z = (y - mean) / sd if sd else np.zeros(len(y))
            iqr = q3 - q1
            nm = names[ok].to_numpy()
            odd = [i for i in np.argsort(-np.abs(z)) if abs(z[i]) >= 2 or y[i] > q3 + 1.5 * iqr or y[i] < q1 - 1.5 * iqr][:6]
            out.append(f"- {c} outliers (|z| >= 2 or beyond 1.5 x IQR): "
                       + (", ".join(f"{nm[i]} {_fmt(y[i])} (z {_fmt(z[i])})" for i in odd) if odd else "none"))
        if "pareto" in asks and (y >= 0).all() and y.sum() > 0:
            s = np.sort(y)[::-1]
            cum = np.cumsum(s) / s.sum()
            k20 = max(1, int(round(len(s) * .2)))
            n80 = int(np.searchsorted(cum, .8) + 1)
            out.append(f"- {c} concentration: top 20% of items ({k20}) hold {_fmt(cum[k20 - 1] * 100)}% of the total; "
                       f"{n80} of {len(s)} items make up 80%")
        if "tiers" in asks and len(y) >= 6:
            lo, hi = np.percentile(y, [33.3, 66.7])
            nm = names[ok].to_numpy()
            tiers = {"high": nm[y > hi], "medium": nm[(y >= lo) & (y <= hi)], "low": nm[y < lo]}
            out.append(f"- {c} tiers by tercile (low < {_fmt(lo)} <= medium <= {_fmt(hi)} < high): "
                       + "; ".join(f"{t} {len(v)} ({', '.join(map(str, v[:6]))}{' ...' if len(v) > 6 else ''})" for t, v in tiers.items()))
    if len(nums) >= 2 and asks & {"corr", "pca", "spread", "tiers"}:
        m = rows[nums[:6]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(m) >= 4:
            pairs = []
            for i in range(len(m.columns)):
                for j in range(i + 1, len(m.columns)):
                    a, b = m.iloc[:, i], m.iloc[:, j]
                    if a.std() == 0 or b.std() == 0:
                        continue
                    r = float(np.corrcoef(a, b)[0, 1])
                    rs = float(np.corrcoef(a.rank(), b.rank())[0, 1])
                    pairs.append((abs(r), m.columns[i], m.columns[j], r, rs))
            for _, c1, c2, r, rs in sorted(pairs, reverse=True)[:4]:
                p = _corr_p(r, len(m))
                # The earlier column is the outcome (the planner puts the measure asked about first): slope = how
                # much of it goes with one more unit of the other.
                slope = np.polyfit(m[c2], m[c1], 1)[0]
                out.append(f"- correlation {c1} vs {c2} over {len(m)} items: Pearson r {_fmt(r)} ({_strength(r)}), "
                           f"Spearman {_fmt(rs)}, p {'< 0.001' if p < .001 else _fmt(p)} "
                           f"({'statistically significant' if p < .05 else 'not significant'} at 5%); R^2 {_fmt(r * r)}; "
                           f"each +1 {c2} goes with {_fmt(slope)} {c1}. Correlation is not proof of cause.")
            if len(m.columns) >= 3 and len(m) > len(m.columns) + 1 and "corr" in asks:
                yv, X = m.iloc[:, 0].to_numpy(float), m.iloc[:, 1:].to_numpy(float)
                Xs = (X - X.mean(0)) / np.where(X.std(0) == 0, 1, X.std(0))
                A = np.column_stack([np.ones(len(Xs)), Xs])
                coef, *_ = np.linalg.lstsq(A, yv, rcond=None)
                pred = A @ coef
                r2 = 1 - ((yv - pred) ** 2).sum() / (((yv - yv.mean()) ** 2).sum() or 1)
                out.append(f"- multiple regression of {m.columns[0]} on {', '.join(m.columns[1:])} ({len(m)} items, "
                           f"standardised): R^2 {_fmt(r2)}; effect per 1 std dev: "
                           + ", ".join(f"{c} {_fmt(v)}" for c, v in zip(m.columns[1:], coef[1:]))
                           + " (few items -- indicative only)")
            if "pca" in asks and len(m.columns) >= 2:
                Z = (m - m.mean()) / m.std(ddof=0).replace(0, 1)
                _, s, vt = np.linalg.svd(Z.to_numpy(float), full_matrices=False)
                share = s ** 2 / (s ** 2).sum() * 100
                load = ", ".join(f"{c} {_fmt(v)}" for c, v in zip(m.columns, vt[0]))
                out.append(f"- PCA over {len(m.columns)} standardised measures, {len(m)} items: component 1 explains "
                           f"{_fmt(share[0])}%, component 2 {_fmt(share[1]) if len(share) > 1 else 'n/a'}%; "
                           f"component 1 loadings: {load}")
    return out


def _variance_split(df: pd.DataFrame | None) -> list[str]:
    """Two-way split of the variation of one measure over two labels (zone x month, OEM x project): the share
    explained by each label's averages, the rest is their interaction. 11 Nov: 'how much of the variation in
    installations comes from zones vs months' got a negative 'variance' from hand-written SQL."""
    if not isinstance(df, pd.DataFrame) or len(df) < 6:
        return []
    labels = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(labels) != 2 or not nums:
        return []
    d = df.copy()
    for c in labels:
        low = d[c].astype(str).str.strip().str.lower()
        d = d[~low.str.match(r"(?:grand )?total\b") & ~low.isin({"", "none", "nan"})]
    y = pd.to_numeric(d[nums[0]], errors="coerce")
    d = d.assign(_y=y).dropna(subset=["_y"])
    if len(d) < 6 or d[labels[0]].nunique() < 2 or d[labels[1]].nunique() < 2:
        return []
    grand = d["_y"].mean()
    ss_tot = float(((d["_y"] - grand) ** 2).sum())
    if not ss_tot:
        return []
    ss = {}
    for c in labels:
        g = d.groupby(c)["_y"].agg(["mean", "count"])
        ss[c] = float((g["count"] * (g["mean"] - grand) ** 2).sum())
    rest = max(0.0, ss_tot - sum(ss.values()))
    return [f"- variation of {nums[0]} across {len(d)} {labels[0]} x {labels[1]} cells (two-way split of the sum of squares, "
            "computed by code): " + ", ".join(f"differences between {c} explain {_fmt(v / ss_tot * 100)}%" for c, v in ss.items())
            + f", the rest ({_fmt(rest / ss_tot * 100)}%) is their interaction / noise"]


def _joined_corr(df: pd.DataFrame | None, extras: list) -> list[str]:
    """Correlation between the main result and an extra query's result lined up on the same place label -- the
    shape of a cross-system question (WFM tickets per zone vs MDM meters per zone: two engines, two queries)."""
    frames = [f for f in [df, *extras] if isinstance(f, pd.DataFrame) and len(f) >= 4]
    if len(frames) < 2:
        return []
    (ra, la, da, na), (rb, lb, db, nb) = _split(frames[0]), _split(frames[1])
    if not (la and lb and na and nb) or da is not None or db is not None:
        return []
    key = lambda s: s.astype(str).str.strip().str.upper().str.replace(r"\s+", " ", regex=True)
    a = ra.assign(_k=key(ra[la]))[["_k", na[0]]]
    b = rb.assign(_k=key(rb[lb]))[["_k", nb[0]]].rename(columns={nb[0]: f"{nb[0]} (result 2)" if nb[0] == na[0] else nb[0]})
    m = a.merge(b, on="_k").dropna()
    if len(m) < 4:
        return [f"- only {len(m)} {la} values appear in both results -- too few to correlate; compare them side by side"]
    x, y = m.iloc[:, 2].astype(float), m.iloc[:, 1].astype(float)
    if x.std() == 0 or y.std() == 0:
        return []
    r = float(np.corrcoef(x, y)[0, 1])
    rs = float(np.corrcoef(x.rank(), y.rank())[0, 1])
    p = _corr_p(r, len(m))
    return [f"- correlation {m.columns[1]} (result 1) vs {m.columns[2]} (result 2) across {len(m)} {la} values found in "
            f"both: Pearson r {_fmt(r)} ({_strength(r)}), Spearman {_fmt(rs)}, p {'< 0.001' if p < .001 else _fmt(p)} "
            f"({'statistically significant' if p < .05 else 'not significant'} at 5%). Correlation is not proof of cause."]


def analyse(question: str, df: pd.DataFrame | None, today: pd.Timestamp | None = None,
            extras: list[pd.DataFrame] | None = None) -> str:
    """The STATS block for the answer prompt ('' when the question asks for no analysis)."""
    asks = wanted(question)
    missing = not_run(question)
    if not asks and not missing:
        return ""
    today = today or pd.Timestamp.now().normalize()
    lines: list[str] = []
    change = scenario(question)
    added = technicians_added(question)
    clear = _clearance_lines(question, [df, *(extras or [])], change, added, today) \
        if (change is not None or added is not None) else []
    lines += clear
    if change is not None and isinstance(df, pd.DataFrame) and len(df) and not clear:
        lines += _scenario_lines(df, change)
    if added is not None and isinstance(df, pd.DataFrame) and len(df) and not clear:
        lines += _technician_lines(df, added)
    if asks & {"corr", "spread"}:
        lines += _joined_corr(df, extras or [])
    if "spread" in asks or re.search(r"\bvariation\b|decompos", str(question), re.I):
        lines += _variance_split(df)
    for frame in [df, *(extras or [])]:
        if not isinstance(frame, pd.DataFrame) or len(frame) < 3:
            continue
        rows, label, dates, nums = _split(frame)
        # Rows the query itself marks as not actual ('projected (June -> July trend)', 'actual (partial, 2-3 days)')
        # are not history. 19 Nov: 'Forecast next month' fitted a line through the SQL's own September projection
        # and the partial August, and quoted October (248,070) beside the table's September (242,710).
        texts = [c for c in rows.columns if not pd.api.types.is_numeric_dtype(rows[c])]
        if texts:
            marked = rows[texts].astype(str).apply(
                lambda s: s.str.contains(r"\b(?:projected|forecast\w*|estimated?|partial)\b", case=False, regex=True)
            ).any(axis=1).to_numpy()
            if marked.any() and (~marked).sum() >= 2:
                rows = rows[~marked].reset_index(drop=True)
                dates = dates[~marked].reset_index(drop=True) if dates is not None else None
        if dates is not None and dates.nunique() <= 1:
            # One date on every row (HES cards return MAX(data_date) AS data_month) labels the month -- the rows are
            # items, not a series. 16 Nov: 35 circles were collapsed into one 'period' and no correlation was computed.
            texts = [c for c in rows.columns if c != label and not pd.api.types.is_numeric_dtype(rows[c])]
            label, dates = (texts[0] if texts else None), None
        if not nums:
            continue
        if dates is not None:
            rows, dates, partial = _drop_partial(rows, dates, today)
            order = dates.sort_values().index
            rows, dates = rows.loc[order].reset_index(drop=True), dates.loc[order].reset_index(drop=True)
            # Long rows (period x group): one series per group would be noise -- use per-period totals.
            if label and rows[label].duplicated().any():
                grp = [c for c in rows.columns if c != label and not pd.api.types.is_numeric_dtype(rows[c])]
                agg = rows.groupby(dates)[nums].sum(min_count=1).reset_index()
                rows, dates = agg, pd.to_datetime(agg.iloc[:, 0])
                if grp:
                    lines.append(f"- (series below are per-period totals across {grp[0]})")
            step = dates.diff().median()
            unit = "day" if step <= pd.Timedelta(days=1) else "week" if step <= pd.Timedelta(days=8) else "month"
            if partial is not None:
                lines.append(f"- {str(partial.date())} is still in progress: left out of the statistics below")
            if len(rows) >= 3:
                for i, c in enumerate([c for c in nums if not re.search(r"projected|forecast", str(c), re.I)][:3]):
                    y = pd.to_numeric(rows[c], errors="coerce")
                    ok = y.notna().to_numpy()
                    if ok.sum() >= 3:
                        got = _series_stats(c, y.to_numpy(float)[ok], dates[ok].reset_index(drop=True), asks, unit,
                                            horizon(question, unit))
                        # Projection / simulation / change point for the headline measure only: the
                        # other columns get their summary line (keeps the prompt short).
                        if "episode" in asks:
                            # Start / peak / back-to-normal per measure: the summary line + its episodes only.
                            lines += got[:1] + _episode_lines(c, y.to_numpy(float)[ok], dates[ok].reset_index(drop=True),
                                                              unit, episode_direction(question))
                            continue
                        lines += got if i == 0 else got[:1]
                        if i == 0 and "forecast" in asks:
                            lines += _multi_horizon_line(c, y.to_numpy(float)[ok], unit, question)
                if len(nums) >= 2 and "corr" in asks:
                    lines += [l for l in _cross_stats(rows, None, nums, {"corr"}) if l.startswith("- correlation")
                              or l.startswith("- multiple")]
                    lines += _lag_lines(rows, nums, unit)
        else:
            lines += _cross_stats(rows, label, nums, asks)
        if len(lines) >= (22 if "episode" in asks else 14):
            break
    head = ("STATS (computed exactly by code from the rows -- quote these figures, never re-estimate them; name the "
            "method in plain words, e.g. 'a straight-line trend', 'correlation across 18 zones'; projections and "
            "simulations are estimates that assume the recent pattern holds -- say so; a forecast is quoted with its 95% "
            "range and its historical check (backtest error) when given; correlation is not proof of cause):")
    tail = []
    if missing:
        tail.append("NOT RUN: " + "; ".join(missing) + ". Say plainly that this was not run (no such model exists in "
                    "this system) -- never present a figure as its output -- and give the nearest figures that WERE computed.")
    if clear:
        tail.append("WHAT-IF ANSWER: lead with the BACKLOG WHAT-IF figures -- days to clear now vs with the change, and "
                    "what is left after the period -- never only say it 'would be reduced'; the % never applies to the "
                    "backlog itself.")
    # Instructions only when their figures exist and the question really asks (replay 17 Nov: 'low power FACTOR'
    # and 'autoCORRELATION' would otherwise pay ~250 tokens for a causation instruction with nothing to apply it to).
    if not _EXPLICIT_CORR_RE.search(str(question)):
        # The 17 Nov additions (lead / lag, too-few-items) only for a real relationship ask; older lines unchanged.
        lines = [l for l in lines if not re.match(r"- (?:over time \(|only \d+ items \()", l)]
    corr_lines = any(re.search(r"^- (?:correlation|over time|only \d+ items|multiple regression)", l) for l in lines)
    if "corr" in asks and corr_lines and _EXPLICIT_CORR_RE.search(str(question)):
        tail.append("CORRELATION vs CAUSATION: answer in two labelled parts. 'What the data shows (correlation)': only the "
                    "measured r / lag / change-on-change figures above, or -- with too few items -- a side-by-side "
                    "comparison and no correlation claim. 'What it does not prove (causation)': correlation can come from "
                    "both growing with the programme over time (see change-on-change), a third factor (project size, rollout "
                    "phase, batch timing) or the reverse direction; give a plausible operational mechanism only as a "
                    "hypothesis, with what would test it. Never state 'drives', 'causes', 'leads to' or 'results in' as a "
                    "finding -- nor 'does not cause': this data can neither prove nor rule out a cause. Answer the causation part separately -- never repeat the correlation sentence as its answer.")
    if any(l.startswith("- EPISODES") for l in lines):
        tail.append("EPISODES: for each episode give when it started, when it peaked (value vs normal), and the date it was "
                    "back to normal (or 'not yet back to normal'); say what 'normal' means in one short clause. Sundays and "
                    "festival days are not anomalies. Never call the first or last date of the data, or a plain highest / "
                    "lowest value, a start / peak / recovery unless an EPISODES line says so; if no episode was found, say "
                    "the measure stayed within its normal range.")
    if not lines and not tail:
        return ""
    if not lines and asks - {"forecast", "whatif"}:
        tail.append("No statistic could be computed from these rows (too few items or periods) -- say what the numbers "
                    "show directly, without claiming a statistical method.")
    return "\n".join([head, *lines[:26], *tail]) if lines else "\n".join(tail)


def line_up(df: pd.DataFrame | None, extras: list) -> str:
    """Two results (one per system / engine) side by side on their shared place / category label, with the gap
    computed by code. 11 Nov: 'compare installed meters in WFM with the meters present in HES for each project' got
    both results but the answer quoted only HES's unplaced row and skipped the per-project comparison."""
    frames = [f for f in [df, *extras] if isinstance(f, pd.DataFrame) and len(f) >= 2]
    if len(frames) < 2:
        return ""
    (ra, la, da, _), (rb, lb, db, _) = _split(frames[0]), _split(frames[1])
    if not (la and lb) or da is not None or db is not None:
        return ""
    # Any numeric column will do here (no statistic is computed): two projects are enough to line up.
    na = [c for c in ra.columns if c != la and pd.api.types.is_numeric_dtype(ra[c])]
    nb = [c for c in rb.columns if c != lb and pd.api.types.is_numeric_dtype(rb[c])]
    if not (na and nb):
        return ""
    key = lambda s: s.astype(str).str.strip().str.upper().str.replace(r"\s+", " ", regex=True)
    a = ra.assign(_k=key(ra[la]))[["_k", na[0]]]
    b = rb.assign(_k=key(rb[lb]))[["_k", nb[0]]]
    bname = f"{nb[0]}_2" if nb[0] == na[0] else nb[0]
    b = b.rename(columns={nb[0]: bname})
    m = a.merge(b, on="_k", how="outer")
    both = m.dropna()
    if len(both) < 1:
        return ""
    rows = []
    for _, r in m.iterrows():
        x, y = r[na[0]], r[bname]
        gap = f"; difference {_fmt(float(y) - float(x))} ({_fmt((float(y) - float(x)) / float(x) * 100) if x else 'n/a'}%)" \
            if pd.notna(x) and pd.notna(y) else " (only in one result)"
        rows.append(f"- {r['_k']}: {na[0]} {_fmt(float(x)) if pd.notna(x) else '-'}, {bname} "
                    f"{_fmt(float(y)) if pd.notna(y) else '-'}{gap}")
    return ("LINED UP BY " + str(la).upper() + " (computed by code: RESULT 1 and RESULT 2 matched on the same name; "
            "compare item by item from here, and name items found in only one result):\n" + "\n".join(rows[:25]))
