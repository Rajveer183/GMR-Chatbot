"""Daily / weekly / monthly programme reports, built by code from fixed, dashboard-checked queries.

'Generate the weekly report', 'daily report for yesterday', 'monthly report for October with exceptions',
'weekly report for Kashi': before 16 Nov these went to the SQL writer, which picked one KPI (or asked which KPI,
or refused). A report is a fixed shape -- KPIs this period vs the previous one, the position now, where the change
came from, exceptions and actions -- so it is assembled here: no LLM call, every figure from a query below
(the all-project figures reconcile with the dashboard strip: installed, SAT meters, SAT % of eligible, remaining
stock, open tickets -- checked 16 Nov 2026).

Periods (windows end yesterday -- today is still loading):
    daily    yesterday (or the date named) vs the day before
    weekly   the last 7 complete days vs the 7 before
    monthly  the month named, or the last complete month, vs the month before; the current month = month to
             date vs the same days of last month
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import pandas as pd

log = logging.getLogger("gmr_agents.report")

_PROJECTS = ("agra", "kashi", "triveni")
_MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}

# What counts as a report request. A KPI-specific 'report' ('feeder loss report', 'SAT report by division') is a
# normal question: only programme-level words may sit between the verb / period and 'report'.
_PERIOD_WORD = r"daily|day|today'?s|yesterday'?s|weekly|week'?s|monthly|month'?s|fortnightly|mis|eod|end of (?:the )?day"
_FILLER = (r"(?:the|a|an|my|our|full|complete|overall|programme|program|project|status|progress|performance|executive|"
           r"management|kpi|summary|cxo|mis|agra|kashi|triveni|for|of|this|last|previous|past|\d{1,2}|"
           + _PERIOD_WORD + r")")
_REPORT_RE = re.compile(
    rf"(?:\b(?:{_PERIOD_WORD})\b(?:\s+{_FILLER})*\s+(?:report|digest|briefing|summary|review|update)\b"
    rf"|\b(?:generate|prepare|create|make|give(?: me)?|send|share|build|produce|draft|run|show(?: me)?)\b(?:\s+{_FILLER})*\s+report\b"
    rf"|^\s*(?:{_FILLER}\s+)*report\s+(?:for|of)\b"
    rf"|\b(?:daily|weekly|monthly)\s+mis\b)", re.I)
_TOPIC_RE = re.compile(r"\b(?:feeder|loss|theft|pf|power factor|voltage|outage|dt\b|transformer|technician|ticket list|"
                       r"warehouse|oem|defect|mapping|journey|receivable|invoice|cable|escalation|hhu|reading|risk)\w*", re.I)


# 'three risks and three actions', 'top 5 risks', 'risks and mitigations': a programme-level ask for the watchlist
# (business_rules.yaml dashboard_stats). 16 Nov: 'Write a CXO weekly summary with three risks and three actions.'
# hit 'risk' in _TOPIC_RE, went to the LLM, and came back with four red flags, no actions and an invented week
# ('02 Nov 2026 to 30 Nov 2026'). The report builds them by code now.
_COUNT = r"(?:\d{1,2}|one|two|three|four|five|six|seven)"
_NUMS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7}
_RISKS_RE = re.compile(rf"\b(?:(?:top|key|main|major|biggest|{_COUNT})\s+)*(?:risks|red flags|concerns)\b", re.I)
_ACTIONS_RE = re.compile(rf"\b(?:(?:top|key|main|recommended|{_COUNT})\s+)*(?:actions|next steps|mitigations|recommendations)\b", re.I)
_RISK_ACTION_RE = re.compile(rf"{_RISKS_RE.pattern}.{{0,40}}{_ACTIONS_RE.pattern}", re.I)


def _count(m: re.Match | None, default: int = 3) -> int | None:
    if not m:
        return None
    n = re.search(_COUNT, m.group(0), re.I)
    v = (int(n.group(0)) if n.group(0).isdigit() else _NUMS[n.group(0).lower()]) if n else default
    return max(1, min(v, 7))


def risk_ask(question: str) -> tuple[int | None, int | None]:
    """(risks wanted, actions wanted) -- None when not asked; 3 when asked without a number."""
    q = str(question or "")
    return _count(_RISKS_RE.search(q)), _count(_ACTIONS_RE.search(q))


def is_report(question: str) -> bool:
    q = str(question or "")
    # The risks / actions wording is the report's own section, not a KPI topic ('theft risk' still is one).
    rest = _ACTIONS_RE.sub(" ", _RISKS_RE.sub(" ", q))
    return bool(_REPORT_RE.search(q) or _RISK_ACTION_RE.search(q)) and not _TOPIC_RE.search(rest)


# --------------------------------------------------------------------------- #
# Period and scope
# --------------------------------------------------------------------------- #

def _month_bounds(y: int, m: int) -> tuple[date, date]:
    start = date(y, m, 1)
    nxt = date(y + (m == 12), m % 12 + 1, 1)
    return start, nxt - timedelta(days=1)


def request(question: str, today: date | None = None) -> dict:
    """{'period', 'start', 'end', 'prev_start', 'prev_end', 'label', 'prev_label', 'project', 'assumed'}"""
    today = today or date.today()
    yday = today - timedelta(days=1)
    q = str(question or "").lower()
    project = next((p for p in _PROJECTS if re.search(rf"\b{p}\b", q)), None)
    named_month = re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b(?:\s+(20\d\d))?", q)
    day_named = re.search(r"\b(\d{1,2})\s*(?:st|nd|rd|th)?\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*(?:\s+(20\d\d))?", q)
    assumed = None
    if re.search(r"\b(?:daily|day|today'?s?|yesterday'?s?|eod|end of (?:the )?day)\b", q) or day_named:
        period = "daily"
        if day_named:
            y = int(day_named.group(3) or today.year)
            d = date(y, _MONTHS[day_named.group(2)[:3]], int(day_named.group(1)))
            if d > yday and not day_named.group(3):
                d = date(y - 1, d.month, d.day)
        else:
            d = yday
        start = end = d
        prev_start = prev_end = d - timedelta(days=1)
        label, prev_label = d.strftime("%d %b %Y"), prev_start.strftime("%d %b %Y")
    elif re.search(r"\b(?:monthly|month'?s?|mis)\b", q) or named_month:
        period = "monthly"
        if named_month:
            m = _MONTHS[named_month.group(1)[:3]]
            y = int(named_month.group(2) or today.year)
            if not named_month.group(2) and (y, m) > (today.year, today.month):
                y -= 1
        elif re.search(r"\bthis month|current month|month to date|mtd\b", q):
            y, m = today.year, today.month
        else:
            y, m = (today.year, today.month - 1) if today.month > 1 else (today.year - 1, 12)
        start, end = _month_bounds(y, m)
        ps, pe = _month_bounds(*((y, m - 1) if m > 1 else (y - 1, 12)))
        if end >= today:  # the running month: month to date vs the same days of last month
            end = yday
            days = (end - start).days
            prev_start, prev_end = ps, min(pe, ps + timedelta(days=days))
            label = f"{start.strftime('%B %Y')} to date ({start.strftime('%d %b')} - {end.strftime('%d %b')})"
            prev_label = f"{prev_start.strftime('%d %b')} - {prev_end.strftime('%d %b %Y')}"
        else:
            prev_start, prev_end = ps, pe
            label, prev_label = start.strftime("%B %Y"), ps.strftime("%B %Y")
    else:
        period = "weekly"
        if not re.search(r"\b(?:weekly|week'?s?|fortnightly)\b", q):
            assumed = "No period was named, so this is the weekly report (last 7 complete days)."
        end = yday
        start = end - timedelta(days=6)
        prev_end = start - timedelta(days=1)
        prev_start = prev_end - timedelta(days=6)
        label = f"{start.strftime('%d %b')} - {end.strftime('%d %b %Y')}"
        prev_label = f"{prev_start.strftime('%d %b')} - {prev_end.strftime('%d %b %Y')}"
    risks, actions = risk_ask(question)
    return {"period": period, "start": start, "end": end, "prev_start": prev_start, "prev_end": prev_end,
            "label": label, "prev_label": prev_label, "project": project, "assumed": assumed,
            "risks": risks, "actions": actions, "cxo": bool(re.search(r"\b(?:cxo|ceo|cfo|coo|executive|leadership)\b", q))}


# --------------------------------------------------------------------------- #
# Queries (Postgres gold tables; read-only, run through the normal executor)
# --------------------------------------------------------------------------- #

_MI = "has_meter_serial AND progress_category IS NOT NULL AND NOT is_replacement"


def _d(x: date) -> str:
    return f"DATE '{x.isoformat()}'"


def _sqls(r: dict) -> dict[str, str]:
    cs, ce, ps, pe = _d(r["start"]), _d(r["end"]), _d(r["prev_start"]), _d(r["prev_end"])
    pf = f" AND LOWER(project) = '{r['project']}'" if r["project"] else ""
    pfs = f" AND LOWER(s.project) = '{r['project']}'" if r["project"] else ""
    place = "zone" if r["project"] else "project"

    def cmp(label: str, col: str, table: str, date_col: str, where: str = "TRUE") -> str:
        return (f"SELECT '{label}' AS kpi, COALESCE(SUM({col}) FILTER (WHERE {date_col} BETWEEN {cs} AND {ce}), 0) AS this_period, "
                f"COALESCE(SUM({col}) FILTER (WHERE {date_col} BETWEEN {ps} AND {pe}), 0) AS previous_period "
                f"FROM public.{table} WHERE {where}{pf} AND {date_col} BETWEEN {ps} AND {ce}")

    kpis = " UNION ALL ".join([
        cmp("Meters installed", "meter_count", "gold_mi_installation_day_agg", "period_start", _MI),
        cmp("Meters supplied", "inventory_count", "gold_meter_inventory_event_day_agg", "period_start"),
        f"SELECT 'Meters SAT-accepted (batches dated in the period)', "
        f"COALESCE(SUM(s.meter_count) FILTER (WHERE m.sat_date BETWEEN {cs} AND {ce}), 0), "
        f"COALESCE(SUM(s.meter_count) FILTER (WHERE m.sat_date BETWEEN {ps} AND {pe}), 0) "
        f"FROM public.gold_mi_sat_invoice_snapshot s JOIN public.milestone_config m "
        f"ON LOWER(m.project) = LOWER(s.project) AND m.sat_number = 'SAT-' || s.sat_stage "
        f"WHERE s.sat_stage IS NOT NULL{pfs} AND m.sat_date BETWEEN {ps} AND {ce}",
        cmp("O&M tickets closed", "closed_tickets", "gold_om_closed_tickets_day_agg", "closed_date"),
    ])
    return {
        "kpis": (f"SELECT kpi, this_period, previous_period, this_period - previous_period AS change, "
                 f"ROUND((this_period - previous_period) * 100.0 / NULLIF(previous_period, 0), 1) AS change_pct "
                 f"FROM ({kpis}) k ORDER BY CASE kpi WHEN 'Meters installed' THEN 1 WHEN 'Meters supplied' THEN 2 "
                 f"WHEN 'O&M tickets closed' THEN 4 ELSE 3 END"),
        "position": (
            f"SELECT (SELECT SUM(meter_count) FROM public.gold_mi_installation_month_agg WHERE {_MI}{pf}) AS installed_so_far, "
            f"(SELECT SUM(meter_count) FILTER (WHERE sat_stage IS NOT NULL) FROM public.gold_mi_sat_invoice_snapshot WHERE TRUE{pf}) AS sat_meters, "
            f"(SELECT ROUND(SUM(meter_count) FILTER (WHERE sat_stage IS NOT NULL) * 100.0 / NULLIF(SUM(meter_count) FILTER (WHERE is_sat_eligible), 0), 1) "
            f"FROM public.gold_mi_sat_invoice_snapshot WHERE TRUE{pf}) AS sat_pct_of_eligible, "
            f"(SELECT SUM(inventory_count - stock_consumed_count) FROM public.gold_meter_inventory_event_day_agg WHERE TRUE{pf}) AS remaining_stock, "
            f"(SELECT SUM(ticket_count) FROM public.gold_om_open_ageing_snapshot WHERE TRUE{pf}) AS open_tickets, "
            f"(SELECT SUM(ticket_count) FILTER (WHERE age_key = 'age_60_plus') FROM public.gold_om_open_ageing_snapshot WHERE TRUE{pf}) AS open_over_60_days, "
            f"(SELECT ROUND(GREATEST(SUM(lumpsum_invoiced - lumpsum_realised - lumpsum_within_timeline), 0) "
            f"+ GREATEST(SUM(pmpm_invoiced - pmpm_realised - pmpm_within_timeline), 0) "
            f"+ GREATEST(SUM(cable_invoiced - cable_realised - cable_within_timeline), 0), 2) "
            f"FROM (SELECT r.*, ROW_NUMBER() OVER (PARTITION BY project ORDER BY created_at DESC, id DESC) AS rn "
            f"FROM public.revenue_receivables r) l WHERE rn = 1{pf}) AS overdue_cr"),
        "by_place": (
            f"SELECT UPPER(COALESCE(NULLIF(TRIM({place}), ''), 'Unmapped')) AS {place}, "
            f"COALESCE(SUM(meter_count) FILTER (WHERE period_start BETWEEN {cs} AND {ce}), 0) AS installed_this_period, "
            f"COALESCE(SUM(meter_count) FILTER (WHERE period_start BETWEEN {ps} AND {pe}), 0) AS installed_previous_period, "
            f"COALESCE(SUM(meter_count) FILTER (WHERE period_start BETWEEN {cs} AND {ce}), 0) "
            f"- COALESCE(SUM(meter_count) FILTER (WHERE period_start BETWEEN {ps} AND {pe}), 0) AS change "
            f"FROM public.gold_mi_installation_day_agg WHERE {_MI}{pf} AND period_start BETWEEN {ps} AND {ce} "
            f"GROUP BY 1 ORDER BY installed_this_period DESC"),
        "om_by_place": (
            f"WITH c AS (SELECT UPPER(COALESCE(NULLIF(TRIM({place}), ''), 'Unmapped')) AS k, "
            f"SUM(closed_tickets) FILTER (WHERE closed_date BETWEEN {cs} AND {ce}) AS closed_this_period, "
            f"SUM(closed_tickets) FILTER (WHERE closed_date BETWEEN {ps} AND {pe}) AS closed_previous_period "
            f"FROM public.gold_om_closed_tickets_day_agg WHERE closed_date BETWEEN {ps} AND {ce}{pf} GROUP BY 1), "
            f"o AS (SELECT UPPER(COALESCE(NULLIF(TRIM({place}), ''), 'Unmapped')) AS k, SUM(ticket_count) AS open_now, "
            f"SUM(ticket_count) FILTER (WHERE age_key = 'age_60_plus') AS open_over_60_days "
            f"FROM public.gold_om_open_ageing_snapshot WHERE TRUE{pf} GROUP BY 1) "
            f"SELECT COALESCE(o.k, c.k) AS {place}, COALESCE(c.closed_this_period, 0) AS closed_this_period, "
            f"COALESCE(c.closed_previous_period, 0) AS closed_previous_period, COALESCE(o.open_now, 0) AS open_now, "
            f"COALESCE(o.open_over_60_days, 0) AS open_over_60_days, "
            f"ROUND(COALESCE(o.open_over_60_days, 0) * 100.0 / NULLIF(o.open_now, 0), 1) AS pct_over_60_days "
            f"FROM o FULL OUTER JOIN c ON c.k = o.k ORDER BY open_now DESC"),
        "daily": (
            f"SELECT period_start AS install_date, SUM(meter_count) AS meters_installed "
            f"FROM public.gold_mi_installation_day_agg WHERE {_MI}{pf} "
            f"AND period_start BETWEEN {_d(min(r['start'], r['end'] - timedelta(days=13)))} AND {ce} "
            f"GROUP BY 1 ORDER BY 1"),
        # Watchlist inputs per project (business_rules.yaml dashboard_stats 'Watchlist'), only when risks / actions
        # were asked; scored in _watchlist. 'today' = the day after the last complete day (the DB session is UTC).
        **({"watch": _watch_sql(r["end"] + timedelta(days=1), pf)} if r.get("risks") or r.get("actions") else {}),
    }


def _watch_sql(today: date, pf: str) -> str:
    t = _d(today)
    age_60 = "age_key = 'age_60_plus'"
    age_90 = "age_key IN ('age_91_120', 'age_120_plus')"

    def per(expr: str, table: str, where: str = "TRUE") -> str:
        return f"(SELECT {expr} FROM public.{table} x WHERE LOWER(x.project) = p.project AND {where})"

    return (
        f"SELECT p.project, "
        f"{per('SUM(ticket_count)', 'gold_om_open_ageing_snapshot')} AS open_tickets, "
        f"{per('SUM(ticket_count)', 'gold_om_open_ageing_snapshot', age_60)} AS open_over_60_days, "
        f"{per('SUM(meter_count)', 'gold_mi_sat_invoice_snapshot')} AS installed_meters, "
        f"{per('SUM(meter_count)', 'gold_mi_sat_invoice_snapshot', 'sat_stage IS NOT NULL')} AS sat_meters, "
        f"{per('SUM(meter_count)', 'gold_non_sat_ageing_snapshot', age_90)} AS waiting_sat_over_90_days, "
        f"{per('MAX(sat_date)', 'milestone_config')} AS last_sat_batch, "
        f"{per('SUM(outstanding_count)', 'gold_revenue_ageing_snapshot', f'{t} - sat_date > 60')} AS not_realised_over_60_days, "
        f"{per('SUM(defect_rate_numerator_count)', 'gold_meter_defect_event_day_agg')} AS defective_meters, "
        f"{per('SUM(meter_count)', 'gold_mi_installation_month_agg', 'NOT is_replacement AND has_meter_serial AND oem IS NOT NULL')} AS defect_base_meters, "
        f"l.lumpsum_invoiced, l.lumpsum_realised "
        f"FROM (SELECT DISTINCT LOWER(project) AS project FROM public.gold_mi_sat_invoice_snapshot WHERE TRUE{pf}) p "
        f"LEFT JOIN (SELECT LOWER(project) AS project, lumpsum_invoiced, lumpsum_realised, "
        f"ROW_NUMBER() OVER (PARTITION BY project ORDER BY created_at DESC, id DESC) AS rn "
        f"FROM public.revenue_receivables) l ON l.project = p.project AND l.rn = 1 "
        f"ORDER BY p.project")


def _run_all(sqls: dict[str, str]) -> dict[str, pd.DataFrame]:
    from gmr_agents.execution import executor
    from gmr_agents.guards import sql_guard
    from gmr_agents.catalog.catalog import get_catalog

    catalog = get_catalog("wfm")

    def one(item):
        key, sql = item
        guard = sql_guard.check_sql_safety(sql, catalog)
        if not guard["ok"]:
            raise RuntimeError(f"report query {key} failed the SQL guard: {guard.get('reason')}")
        return key, executor.run_sql(guard["normalized_sql"], None, "postgres")

    with ThreadPoolExecutor(len(sqls)) as pool:
        return dict(pool.map(one, sqls.items()))


# --------------------------------------------------------------------------- #
# Narrative, exceptions and actions (code: every figure comes from the frames)
# --------------------------------------------------------------------------- #

def _n(v) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if pd.isna(f):
        return "n/a"
    return f"{f:,.0f}" if abs(f) >= 100 or f == int(f) else f"{f:,.1f}"


def _chg(this: float, prev: float) -> str:
    if not prev:
        return "no figure the period before" if this else "none in either period"
    p = (this - prev) / prev * 100
    return f"{'+' if p >= 0 else ''}{p:.1f}% vs {_n(prev)}"


def _exceptions(r: dict, f: dict[str, pd.DataFrame]) -> tuple[list[dict], list[str]]:
    """(exception rows for the table, recommended actions) -- thresholds stated in each row."""
    rows: list[dict] = []
    actions: list[str] = []
    place = "Zone" if r["project"] else "Project"
    bp = f["by_place"]
    bp = bp[bp.iloc[:, 0] != "UNMAPPED"]
    for _, x in bp.iterrows():
        this, prev = float(x["installed_this_period"]), float(x["installed_previous_period"])
        if prev >= (50 if r["period"] == "daily" else 200) and this < 0.75 * prev:
            rows.append({"area": f"{place} {x.iloc[0]}", "check": "Installations fell 25% or more",
                         "value": f"{_n(this)} vs {_n(prev)} ({(this - prev) / prev * 100:.1f}%)", "severity": "High" if this < 0.5 * prev else "Medium"})
    falls = sorted([x for x in rows if x["check"].startswith("Installations")], key=lambda x: x["severity"])
    if falls:
        worst = bp.assign(_c=bp["change"]).sort_values("_c").iloc[0]
        actions.append(f"Review installation capacity in {worst.iloc[0]}: {_n(worst['installed_this_period'])} meters this "
                       f"period vs {_n(worst['installed_previous_period'])} before -- check technician attendance and stock at its sites.")
    om = f["om_by_place"]
    om = om[om.iloc[:, 0] != "UNMAPPED"]
    for _, x in om.iterrows():
        if float(x["open_now"] or 0) >= 500 and float(x["pct_over_60_days"] or 0) > 20:
            rows.append({"area": f"{place} {x.iloc[0]}", "check": "Open O&M tickets older than 60 days above 20%",
                         "value": f"{_n(x['open_over_60_days'])} of {_n(x['open_now'])} ({x['pct_over_60_days']}%)",
                         "severity": "High" if float(x["pct_over_60_days"]) > 50 else "Medium"})
    if len(om):
        top = om.sort_values("open_over_60_days", ascending=False).iloc[0]
        if float(top["open_over_60_days"] or 0) > 0:
            actions.append(f"Clear the 60+ day O&M backlog in {top.iloc[0]} first: {_n(top['open_over_60_days'])} of its "
                           f"{_n(top['open_now'])} open tickets are older than 60 days.")
    pos = f["position"].iloc[0]
    k = f["kpis"].set_index("kpi")
    days = (r["end"] - r["start"]).days + 1
    pace = float(k.loc["Meters installed", "this_period"]) / days if days else 0
    if pace > 0 and pos["remaining_stock"] is not None and not pd.isna(pos["remaining_stock"]):
        cover = float(pos["remaining_stock"]) / pace
        if cover < 30:
            rows.append({"area": "Stock", "check": "Remaining stock covers under 30 days at this period's pace",
                         "value": f"{_n(pos['remaining_stock'])} meters / {_n(pace)} a day = {cover:.0f} days",
                         "severity": "High" if cover < 15 else "Medium"})
            actions.append(f"Raise meter supply: remaining stock of {_n(pos['remaining_stock'])} meters lasts about "
                           f"{cover:.0f} days at {_n(pace)} installs a day.")
    if r["period"] != "daily":
        d = f["daily"]
        d = d[(pd.to_datetime(d["install_date"]).dt.date >= r["start"])]
        if len(d) >= 3:
            avg = float(d["meters_installed"].mean())
            for _, x in d.iterrows():
                if float(x["meters_installed"]) < 0.5 * avg:
                    rows.append({"area": f"Day {pd.Timestamp(x['install_date']).strftime('%d %b %Y')}",
                                 "check": "Installations under half the period's daily average",
                                 "value": f"{_n(x['meters_installed'])} vs average {_n(avg)}", "severity": "Low"})
    sat = k.loc["Meters SAT-accepted (batches dated in the period)"]
    if float(sat["this_period"]) == 0:
        rows.append({"area": "SAT", "check": "No SAT batch dated in this period (SAT is accepted in batches)",
                     "value": f"SAT stands at {_n(pos['sat_meters'])} meters, {_n(pos['sat_pct_of_eligible'])}% of eligible",
                     "severity": "Info"})
    order = {"High": 0, "Medium": 1, "Low": 2, "Info": 3}
    rows.sort(key=lambda x: order.get(x["severity"], 9))
    return rows, actions[:3]


def _name(x) -> str:
    """'VARANASI II' -> 'Varanasi II' (title case, Roman numerals kept)."""
    return re.sub(r"\b(?:Ii|Iii|Iv|Vi|Vii)\b", lambda m: m.group(0).upper(), str(x).title())


def _sev(value: float, threshold: float, worse_low: bool = False) -> float:
    """Severity = value / threshold - 1 (threshold / value - 1 when lower is worse), between -1 and 3."""
    s = (threshold / value - 1 if value else 3.0) if worse_low else value / threshold - 1
    return round(max(-1.0, min(s, 3.0)), 2)


def _watchlist(r: dict, f: dict[str, pd.DataFrame], partial_day: bool) -> list[dict]:
    """Every watchlist check (business_rules.yaml dashboard_stats), worst first:
    {'risk', 'value', 'threshold', 'severity', 'status', 'evidence', 'action'} -- each action names where to start."""
    w = f["watch"].copy()
    for c in w.columns[1:]:
        if c != "last_sat_batch":
            w[c] = pd.to_numeric(w[c], errors="coerce").fillna(0.0)
    w["project"] = w["project"].map(_name)
    today = r["end"] + timedelta(days=1)
    period = {"daily": "the day before", "weekly": "the week before", "monthly": "the month before"}[r["period"]]
    place = "zone" if r["project"] else "project"
    checks: list[dict] = []
    # One project in scope: name it plainly; across projects: say which to start with.
    lead, first = ("for", "") if r["project"] else ("starting with", " first")

    def add(risk: str, value: float, threshold: float, shown: str, limit: str, evidence: str, action: str,
            worse_low: bool = False, sev: float | None = None):
        sev = _sev(value, threshold, worse_low) if sev is None else sev
        checks.append({"risk": risk, "value": shown, "threshold": limit, "severity": sev,
                       "status": "Red flag" if sev > 0 else "Within threshold", "evidence": evidence, "action": action})

    # SAT backlog: meters waiting > 90 days (% of installed, 3%) and the gap since the last SAT batch (30 days) are
    # one risk -- the same backlog seen twice -- scored by the worse of the two.
    installed, waiting = w["installed_meters"].sum(), w["waiting_sat_over_90_days"].sum()
    dates = pd.to_datetime(w["last_sat_batch"], errors="coerce")
    if installed:
        pct = waiting * 100 / installed
        last = dates.max()
        gap = (today - last.date()).days if pd.notna(last) else None
        sev = max(_sev(pct, 3.0), _sev(gap, 30.0) if gap is not None else -1.0)
        top = w.loc[w["waiting_sat_over_90_days"].idxmax()]
        top_last = pd.to_datetime(top["last_sat_batch"], errors="coerce")
        add("SAT backlog", pct, 3.0, f"{pct:.1f}% of installed" + (f"; {gap} days since last batch" if gap is not None else ""),
            "3% of installed; 30 days",
            f"{_n(waiting)} meters ({pct:.1f}% of installed) have waited over 90 days for SAT"
            + (f"; the last SAT batch was dated {last.strftime('%d %b %Y')}, {gap} days ago" if gap is not None else ""),
            f"Get the next SAT batch scheduled with the DISCOM{'' if r['project'] else ','} {lead} {top['project']}: {_n(top['waiting_sat_over_90_days'])} "
            f"of its meters have waited over 90 days"
            + (f" and its last batch was dated {top_last.strftime('%d %b %Y')}." if pd.notna(top_last) else "."), sev=sev)

    open_t, open_60 = w["open_tickets"].sum(), w["open_over_60_days"].sum()
    if open_t:
        pct = open_60 * 100 / open_t
        om = f["om_by_place"]
        om = om[om.iloc[:, 0] != "UNMAPPED"]
        where = ""
        if len(om):
            t = om.sort_values("open_over_60_days", ascending=False).iloc[0]
            where = (f" in {_name(t.iloc[0])} first: {_n(t['open_over_60_days'])} of its {_n(t['open_now'])} open tickets "
                     "are older than 60 days")
        add("O&M ticket backlog", pct, 20.0, f"{pct:.1f}% of open", "20% of open",
            f"{_n(open_60)} of {_n(open_t)} open O&M tickets ({pct:.1f}%) are older than 60 days",
            f"Clear the 60+ day O&M backlog{where}." if where else "Clear the 60+ day O&M backlog, oldest tickets first.")

    inv, real = w["lumpsum_invoiced"].sum(), w["lumpsum_realised"].sum()
    if inv:
        pct = (inv - real) * 100 / inv
        w["_due"] = w["lumpsum_invoiced"] - w["lumpsum_realised"]
        top = w.loc[w["_due"].idxmax()]
        add("Lumpsum collections", pct, 20.0, f"{pct:.1f}% outstanding", "20% of invoiced",
            f"Rs {inv - real:,.1f} crore of Rs {inv:,.1f} crore lumpsum invoiced ({pct:.1f}%) is not yet realised",
            f"Escalate lumpsum collection in {top['project']}{first}: Rs {top['_due']:,.1f} crore of its Rs "
            f"{top['lumpsum_invoiced']:,.1f} crore invoiced is still unpaid.")

    sat, late = w["sat_meters"].sum(), w["not_realised_over_60_days"].sum()
    if sat:
        pct = late * 100 / sat
        top = w.loc[w["not_realised_over_60_days"].idxmax()]
        add("Revenue realisation lag", pct, 5.0, f"{pct:.1f}% of SAT meters", "5% of SAT meters",
            f"{_n(late)} SAT-accepted meters ({pct:.1f}%) are unpaid more than 60 days after SAT",
            f"Chase payment for SAT-accepted meters in {top['project']}{first}: {_n(top['not_realised_over_60_days'])} are unpaid "
            "more than 60 days after SAT.")

    k = f["kpis"].set_index("kpi").loc["Meters installed"]
    this, prev = float(k["this_period"]), float(k["previous_period"])
    if prev:
        drop = (1 - this / prev) * 100
        bp = f["by_place"]
        bp = bp[bp.iloc[:, 0] != "UNMAPPED"]
        worst = bp.sort_values("change").iloc[0] if len(bp) else None
        load = (f"; {r['end'].strftime('%d %b')} may not be fully loaded yet" if partial_day else "")
        act = (f"Restore the installation pace in {_name(worst.iloc[0])} ({_n(worst['installed_this_period'])} vs "
               f"{_n(worst['installed_previous_period'])} {period}): check technician attendance and stock at its sites."
               if worst is not None and worst["change"] < 0 else "Restore the installation pace: check technician attendance and stock.")
        if partial_day:
            act = f"Confirm {r['end'].strftime('%d %b')} is fully loaded, then " + act[0].lower() + act[1:]
        add("Installation pace", drop, 20.0, f"{drop:.1f}% drop" if drop > 0 else f"{-drop:.1f}% rise", "20% drop",
            f"{_n(this)} meters installed vs {_n(prev)} {period} ({'-' if drop > 0 else '+'}{abs(drop):.1f}%){load}", act)

    pos = f["position"].iloc[0]
    days = (r["end"] - r["start"]).days + 1
    pace = this / days if days else 0
    if pace > 0 and pd.notna(pos["remaining_stock"]):
        cover = float(pos["remaining_stock"]) / pace
        add("Meter stock cover", cover, 30.0, f"{cover:.0f} days", "30 days",
            f"{_n(pos['remaining_stock'])} meters in stock cover about {cover:.0f} days at {_n(pace)} installs a day",
            f"Raise meter supply: remaining stock of {_n(pos['remaining_stock'])} meters lasts about {cover:.0f} days "
            f"at {_n(pace)} installs a day.", worse_low=True)

    base, bad = w["defect_base_meters"].sum(), w["defective_meters"].sum()
    if base:
        pct = bad * 100 / base
        w["_rate"] = w["defective_meters"] * 100 / w["defect_base_meters"].where(w["defect_base_meters"] > 0)
        top = w.loc[w["_rate"].fillna(-1).idxmax()]
        add("Meter defects", pct, 1.5, f"{pct:.2f}%", "1.5%",
            f"{_n(bad)} defective meters, {pct:.2f}% of installed",
            f"Take up defective meters with the OEMs, {lead} {top['project']} ({top['_rate']:.2f}% defective).")

    return sorted(checks, key=lambda c: -c["severity"])


def build(question: str, today: date | None = None) -> dict:
    """The report as an answer: summary, insights (actions), kpis, tables, charts, and the SQL behind each table
    (sql = table 0, extra_sql = tables 1..) so the table download / paging endpoints can re-run them."""
    r = request(question, today)
    sqls = _sqls(r)
    f = _run_all(sqls)
    k = f["kpis"].set_index("kpi")
    pos = f["position"].iloc[0]
    scope = r["project"].title() if r["project"] else "all projects"
    title = {"daily": "Daily", "weekly": "Weekly", "monthly": "Monthly"}[r["period"]] + (" CXO summary" if r["cxo"] else " report")
    inst = k.loc["Meters installed"]
    days = (r["end"] - r["start"]).days + 1
    exc, actions = _exceptions(r, f)

    # Where the installation change came from (contribution of each place to the total change).
    bp = f["by_place"].copy()
    total_change = float(bp["change"].sum())
    bp["share_of_change_pct"] = (bp["change"] * 100.0 / total_change).round(1) if total_change else 0.0
    driver = ""
    if total_change and len(bp):
        lead = bp.sort_values("change", ascending=total_change < 0).iloc[0]
        if (lead["change"] > 0) == (total_change > 0):
            driver = (f" {lead.iloc[0].title()} accounts for the largest part of the {'rise' if total_change > 0 else 'fall'}"
                      f" ({'+' if lead['change'] >= 0 else ''}{_n(lead['change'])} of {'+' if total_change >= 0 else ''}{_n(total_change)}).")

    lines = [
        f"**{title} -- {scope}, {r['label']}** (compared with {r['prev_label']}).",
        f"Installed {_n(inst['this_period'])} meters ({_chg(inst['this_period'], inst['previous_period'])}"
        + (f"; {_n(inst['this_period'] / days)} a day" if days > 1 else "") + f").{driver}",
        f"Supplied {_n(k.loc['Meters supplied', 'this_period'])} meters; closed {_n(k.loc['O&M tickets closed', 'this_period'])} "
        f"O&M tickets ({_chg(k.loc['O&M tickets closed', 'this_period'], k.loc['O&M tickets closed', 'previous_period'])}); "
        f"{_n(k.loc['Meters SAT-accepted (batches dated in the period)', 'this_period'])} meters SAT-accepted in batches dated in the period.",
        f"Position now: {_n(pos['installed_so_far'])} installed, SAT {_n(pos['sat_meters'])} meters ({_n(pos['sat_pct_of_eligible'])}% of eligible), "
        f"{_n(pos['remaining_stock'])} meters in stock, {_n(pos['open_tickets'])} open O&M tickets "
        f"({_n(pos['open_over_60_days'])} older than 60 days), Rs {pos['overdue_cr']} crore overdue.",
    ]
    watch = []
    if r["risks"] or r["actions"]:
        n_risks = r["risks"] or r["actions"]
        n_actions = r["actions"] or n_risks
    else:
        high = [e for e in exc if e["severity"] in ("High", "Medium")]
        lines.append(f"{len(high)} exception(s) need attention" + (f" -- top: {high[0]['area']}, {high[0]['check'].lower()} ({high[0]['value']})." if high else "."))
    # The last day often lands in pieces (refresh mid-day): a day far below the days before it is flagged, not read
    # as a collapse. 16 Nov: 15 Nov showed 376 installs against ~2,500 a day the week before.
    d = f["daily"]
    partial_day = False
    if len(d) >= 4:
        last, before = float(d["meters_installed"].iloc[-1]), float(d["meters_installed"].iloc[-8:-1].mean())
        if pd.Timestamp(d["install_date"].iloc[-1]).date() == r["end"] and before and last < 0.3 * before:
            partial_day = True
            lines.append(f"Note: {r['end'].strftime('%d %b')} shows only {_n(last)} installs against {_n(before)} a day "
                         "before it -- that day may not be fully loaded yet, so this period's figures may rise at the next refresh.")
    if r["risks"] or r["actions"]:
        # Exactly the number asked, ranked by severity; when fewer checks cross their threshold, the closest ones
        # fill the list and are labelled as within threshold -- never a red flag that is not one.
        watch = _watchlist(r, f, partial_day)
        red = sum(c["severity"] > 0 for c in watch)
        if r["risks"]:
            lines.append(f"\n**Top {n_risks} risks** ({red} of {len(watch)} watchlist checks are past their threshold):")
            lines += [f"{i}. **{c['risk']}** -- {c['evidence']}." + ("" if c["severity"] > 0 else " (Within threshold; on watch.)")
                      for i, c in enumerate(watch[:n_risks], 1)]
        lines.append(f"\n**{n_actions} actions:**")
        lines += [f"{i}. {c['action']}" for i, c in enumerate(watch[:n_actions], 1)]
        actions = []  # the actions are in the summary; insights would repeat them
    if r["assumed"]:
        lines.insert(1, r["assumed"])
    summary = "\n".join(lines)

    kpis = [
        {"label": "Installed (period)", "value": float(inst["this_period"]), "formatted_value": _n(inst["this_period"]), "unit": None},
        {"label": "Change vs previous", "value": float(inst["change_pct"]) if pd.notna(inst["change_pct"]) else 0.0,
         "formatted_value": f"{inst['change_pct']:+.1f}%" if pd.notna(inst["change_pct"]) else "n/a", "unit": None},
        {"label": "SAT % of eligible", "value": float(pos["sat_pct_of_eligible"]), "formatted_value": f"{pos['sat_pct_of_eligible']}%", "unit": None},
        {"label": "Open O&M tickets", "value": float(pos["open_tickets"]), "formatted_value": _n(pos["open_tickets"]), "unit": None},
    ]
    place = "Zone" if r["project"] else "Project"

    def table(tid: str, heading: str, df: pd.DataFrame, labels: dict[str, str]) -> dict:
        cols = [labels.get(c, c.replace("_", " ").title()) for c in df.columns]
        rows = [{cols[i]: (None if pd.isna(v) else (v.isoformat() if hasattr(v, "isoformat") else
                           (int(v) if isinstance(v, float) and v == int(v) else v))) for i, v in enumerate(rr)}
                for rr in df.itertuples(index=False)]
        return {"id": tid, "title": heading, "heading": heading, "columns": cols, "rows": rows, "page": 1,
                "page_size": max(len(rows), 1), "total_rows": len(rows), "has_next": False, "has_previous": False}

    tp, pp = f"This period ({r['label']})", f"Previous ({r['prev_label']})"
    tables = [
        table("report_kpis", "KPIs: this period vs previous", f["kpis"],
              {"kpi": "KPI", "this_period": tp, "previous_period": pp, "change": "Change", "change_pct": "Change %"}),
        table("report_position", "Position now (latest refresh)", f["position"],
              {"installed_so_far": "Installed so far", "sat_meters": "SAT meters", "sat_pct_of_eligible": "SAT % of eligible",
               "remaining_stock": "Remaining stock", "open_tickets": "Open O&M tickets", "open_over_60_days": "Open > 60 days",
               "overdue_cr": "Overdue (Rs crore)"}),
        table("report_by_place", f"Installations by {place.lower()} (where the change came from)", bp,
              {"installed_this_period": tp, "installed_previous_period": pp, "change": "Change",
               "share_of_change_pct": "Share of change %"}),
        table("report_om", f"O&M by {place.lower()}", f["om_by_place"],
              {"closed_this_period": f"Closed: {r['label']}", "closed_previous_period": f"Closed: {r['prev_label']}",
               "open_now": "Open now", "open_over_60_days": "Open > 60 days", "pct_over_60_days": "% over 60 days"}),
        table("report_exceptions", "Exceptions", pd.DataFrame(exc or [{"area": "-", "check": "No exception crossed its threshold",
                                                                           "value": "-", "severity": "-"}]),
              {"area": "Area", "check": "Check (threshold)", "value": "Evidence", "severity": "Severity"}),
    ]
    if watch:
        tables.append(table("report_watchlist", "Watchlist: every check vs its threshold",
                            pd.DataFrame(watch)[["risk", "value", "threshold", "severity", "status", "action"]],
                            {"risk": "Risk", "value": "Value", "threshold": "Threshold", "severity": "Severity",
                             "status": "Status", "action": "Action"}))
    daily = f["daily"]
    charts = []
    if len(daily) >= 2:
        charts.append({"id": "report_daily", "type": "bar", "title": "Meters installed per day",
                       "x": [pd.Timestamp(d).strftime("%d %b") for d in daily["install_date"]],
                       "y": [float(v) for v in daily["meters_installed"]]})
    insights = actions if (actions or watch) else ["No exception crossed its threshold this period."]
    order = ["kpis", "position", "by_place", "om_by_place"]
    return {"summary": summary, "insights": insights, "kpis": kpis, "tables": tables, "charts": charts,
            "sql": sqls[order[0]], "extra_sql": [sqls[x] for x in order[1:]], "request": r,
            "frames": {**f, "by_place": bp, "exceptions": pd.DataFrame(exc)}, "title": f"{title} - {scope} - {r['label']}"}
