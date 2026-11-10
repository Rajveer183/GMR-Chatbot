"""Verifier (no LLM): do the SQL's filters match what the question names?

Runs after the safety check (validation/sql_guard.py) and before execution.
Two error classes seen in testing, both cheap to catch:
  * MISSING  the question names a project / OEM / category / SAT stage / year /
             top-N that the SQL never mentions ("defects for SEIPL" without SEIPL)
  * LEAKED   the SQL filters on a project or OEM the question never named -- a
             filter copied from an example ("... = 'kashi'" for an all-projects ask)
A problem goes back to the query writer as one repair hint. It never rewrites SQL
and never blocks an answer on its own: it only flags clear mismatches.
`question` should be the resolved question (follow-ups already carry the
previous turn's entities), so carried filters are not reported as leaks.
"""

from __future__ import annotations

import re

_PROJECTS = {"agra": ["agra"], "kashi": ["kashi", "kasi", "varanasi"],
             "triveni": ["triveni", "trivei", "prayagraj", "mirzapur"]}
_OEMS = {"aew": ["aew"], "cap": ["cap"], "hpl": ["hpl"], "seipl": ["seipl"]}
_CATEGORIES = {"CONSUMER": ["consumer meter", "consumer meters", "consumer only", "consumers only"],
               "DT": ["dtr", "dt meter", "dt meters", "transformer"],
               "FEEDER": ["feeder meter", "feeder meters", "feeders only", "feeder only"]}
# Consumer meter types: Postgres value -> aliases in the question, and the spellings that
# count as "used" in SQL (Athena silver writes e.g. 'NSM1-PH', so '1-ph' counts for 1PH).
_METER_TYPES = {"1PH-Consumer_meter": (["1ph", "1 ph", "1 phase", "1phase", "single phase", "singlephase"],
                                       ["1ph", "1-ph", "1_ph"]),
                "3PH-Consumer_meter": (["3ph", "3 ph", "3 phase", "3phase", "three phase", "threephase"],
                                       ["3ph", "3-ph", "3_ph"]),
                "LTCT-Consumer_meter": (["ltct", "lt ct"], ["ltct"]),
                "HTCT-Consumer_meter": (["htct", "ht ct"], ["htct"])}
# (unit, how the question asks for it, what the SQL's GROUP BY must contain)
_TIME_BREAKDOWNS = [
    ("weekly", r"\b(?:weekly|week[- ]?wise|weekwise|each week|every week|by week|week by week|per week)\b",
     r"(?:week|/\s*7)"),
    ("daily", r"\b(?:daily|day[- ]?wise|daywise|each day|every day|by day|day by day|date[- ]?wise)\b",
     r"(?:period_start|date|day)"),
    ("monthly", r"\b(?:monthly|month[- ]?wise|monthwise|each month|every month|by month|month by month)\b",
     r"(?:month|period_start)"),
]
_PER = {"weekly": "week", "daily": "day", "monthly": "month"}
# 'weekly summary', 'monthly billing estimate', 'daily average' describe a figure, not one row per period.
_NOT_BREAKDOWN = (r"(?:summary|report|update|review|meeting|status|off|billing|bill|estimate|rate|run[- ]rate|target|"
                  r"pace|avg|average|mean|limit|cap)\b")


def _time_grouped(low: str, bucket: str) -> bool:
    """The SQL groups by the period: the bucket is in GROUP BY, or GROUP BY uses positions /
    aliases and the bucket is in the SELECT list ("date_trunc('month', period_start) AS month ... GROUP BY 1")."""
    if not re.search(r"\bgroup\s+by\b", low):
        return False
    if re.search(r"\bgroup\s+by\b[^;]*?" + bucket, low, re.DOTALL):
        return True
    select_list = re.search(r"\bselect\b(.*?)\bfrom\b", low, re.DOTALL)
    return bool(select_list and re.search(bucket, select_list.group(1)))
_NUM_WORDS = {"three": 3, "five": 5, "ten": 10, "twenty": 20, "fifteen": 15}


def _words(text: str) -> str:
    return " " + re.sub(r"[^a-z0-9]+", " ", str(text).lower()) + " "


def _named(text: str, table: dict[str, list[str]]) -> set[str]:
    w = _words(text)
    return {key for key, aliases in table.items() if any(f" {a} " in w for a in aliases)}


def meter_types(question: str) -> list[str]:
    """Consumer meter types the question names ('1ph', 'single phase', 'LTCT' ...), as Postgres values."""
    w = _words(question)
    return [mt for mt, (aliases, _) in _METER_TYPES.items() if any(f" {a} " in w for a in aliases)]


def _filtered(sql: str, values: list[str]) -> set[str]:
    """Values the SQL uses as an equality / IN filter (not just a breakdown label)."""
    low = sql.lower()
    hits = set()
    for v in values:
        if re.search(rf"(=\s*'{v}'|in\s*\([^)]*'{v}'[^)]*\)|like\s*'%{v}%')", low):
            hits.add(v)
    return hits


def _grouped_by(low: str, column: str) -> bool:
    """The SQL breaks the result down by this column (GROUP BY / PARTITION BY ... column)."""
    return bool(re.search(rf"\b(?:group|partition)\s+by\b[^;]*?\b{column}\b", low, re.DOTALL))


_SHARE_RE = re.compile(r"\b(?:share|percent|percentage)\b|%", re.I)
_COMPARE_RE = re.compile(r"\b(compare|compared|comparison|versus|vs|other|others|rank|ranking|"
                         r"highest|lowest|best|worst)\b")
_DATE_COLUMN_RE = re.compile(r"(date|time|_at$|^year|year$|month|period|^day|_day$|week)")


def _stock_filtered_by_place(sql: str) -> bool:
    """A query level that computes a stock balance (stock_consumed_count) and filters its OWN
    rows on zone / circle / division -- a zone used only inside a subquery that finds the
    zone's project is fine."""
    low = sql.lower()
    if "stock_consumed_count" not in low or "gold_meter_inventory_event_" not in low:
        return False
    try:
        import sqlglot
        from sqlglot import exp
        tree = sqlglot.parse_one(sql, read="postgres")
    except Exception:
        return False
    for select in tree.find_all(exp.Select):
        own = lambda node: node.find_ancestor(exp.Select) is select
        if not any(c.name.lower() == "stock_consumed_count" and own(c) for c in select.find_all(exp.Column)):
            continue
        where = select.args.get("where")
        if where and any(c.name.lower() in _PLACE_LEVELS and c.find_ancestor(exp.Select) is select
                         for c in where.find_all(exp.Column)):
            return True
    return False


# Zone is not here: a zone filter on stock is right (it reads the event rows stored with that zone, like the
# dashboard's one-zone tiles; zone-wise breakdowns get warehouse stock placed by sql/hierarchy.py); circle /
# division have no stock.
_WITHOUT_SAT_RE = re.compile(r"\b(without|pending|awaiting|no|not)\s+(a\s+)?sat\b|\bsat\s+(pending|not done|baaki)\b|"
                             r"\bnot\s+(yet\s+)?(cleared|passed)\s+sat\b", re.I)
_AGEING_RE = re.compile(r"\b(age|ageing|aging|aged|old|older|band|bands|days?|since|how long|stuck|why)\b", re.I)
_CONSUMER_RE = re.compile(r"\bconsumers?\b(?!\s*(number|no\b|id\b|name|account|code))", re.I)
_TYPE_WISE_RE = re.compile(r"\btype[- ]?wise\b|\bby (meter )?types?\b|\btypes? wise\b|\beach type\b", re.I)
_PLACE_LEVELS = {"circle", "division", "sub_division"}
_TREND_CUBE = "gold_mi_productivity_trend_month_agg"
# Words that name the Monthly Productivity trend card (KPI 3); the cube's own measure may also be asked by name.
# The card is monthly only: a 'daily / weekly productivity trend' is KPI 2's.
_KPI3_CARD_RE = re.compile(r"(?<!daily )(?<!weekly )\bproductivity trend\b|\b(?:monthly|month[- ]?wise) (?:mi )?productivity\b|"
                           r"\bkpi[- ]?3\b", re.I)
_KPI3_RE = re.compile(_KPI3_CARD_RE.pattern + r"|\b(?:average|avg\.?) active technicians\b", re.I)
_SUPPLY_CARD_RE = re.compile(r"\b(?:supply|inventory) utili[sz]ation\b|\butili[sz]ation rate\b|\bmi pace\b|"
                             r"\bpace vs\.? stock\b|\bstock availa?bi?li?ty\b", re.I)
_DISCOM_PROJECTS = {"puvvnl": ["kashi", "triveni"], "dvvnl": ["agra"]}
_DROPPED_TYPE_RE = re.compile(r"meter_type\s*(?:<>|!=)\s*'other-consumer_meter'|"
                              r"meter_type\s+not\s+in\s*\([^)]*'other-consumer_meter'|meter_type\s+is\s+not\s+null")
_REVENUE_AGEING_RE = re.compile(r"\brevenue\s+(?:not|un)[- ]?reali[sz]ed\b.*\b(?:ageing|aging|aged|age|old|days)\b|"
                                r"\brevenue\s+ageing\b", re.I)
_SERIAL_RE = re.compile(r"\b[A-Za-z]{1,4}\d{6,10}\b")
_STOCK_STAGE_RE = re.compile(r"\bwith (?:the )?(?:technicians?|supervisors?|agenc(?:y|ies))\b|"
                             r"\bin (?:the )?(?:gmr )?warehouses?\b", re.I)


_PLACE_LITERAL_RE = re.compile(r"\b(zone|circle|division|sub_division|filter_label)\b\s*\)?\s*"
                               r"(?:=\s*'([^']+)'|in\s*\(([^)]*)\))", re.I)


def _unknown_places(low: str) -> list[str]:
    """Place literals (zone / circle / division / sub-division / KPI 3 filter_label) that no stored place
    has, with the stored names that contain the asked words. Skipped until the place list has loaded."""
    from gmr_agents.execution import hierarchy

    places = hierarchy.place_names()
    if not places:
        return []
    known = set(places) | {"all", "unmapped", ""} | set(_PROJECTS) | {"puvvnl", "dvvnl"}
    out = []
    for m in _PLACE_LITERAL_RE.finditer(low):
        values = [m.group(2)] if m.group(2) is not None else re.findall(r"'([^']*)'", m.group(3) or "")
        for v in values:
            v = v.strip().lower()
            if v in known:
                continue
            words = [w for w in re.findall(r"[a-z]+", v) if len(w) > 2 and w not in {
                "edd", "edc", "eudc", "eudd", "edsd", "zone", "circle", "division", "sub"}]
            near = sorted(p for p in places if words and all(w in p for w in words))[:5]
            out.append(f"no stored {m.group(1)} is named '{v}'" + (f" -- the stored names are {near}" if near else "")
                       + " (place names are stored in full, e.g. 'edd chunar', 'edc agra')")
    return out


_PLACE_FILTER_RE = re.compile(r"(?:lower\(\s*)?\b(?:trimmed_)?(project|zone|circle|division|sub_division)\b\s*\)?\s*=\s*'([^']+)'")


def _without_subqueries(text: str) -> str:
    """`text` with every parenthesised SELECT removed (balanced parentheses)."""
    out, i = [], 0
    while i < len(text):
        m = re.compile(r"\(\s*select\b").search(text, i)
        if not m:
            out.append(text[i:])
            break
        out.append(text[i:m.start()])
        depth, j = 0, m.start()
        while j < len(text):
            depth += {"(": 1, ")": -1}.get(text[j], 0)
            j += 1
            if depth == 0:
                break
        i = j
    return "".join(out)


def _union_place_gaps(low: str) -> list[str]:
    if "gold_meter_defect_event_day_agg" not in low or "gold_mi_installation_" not in low or "union all" not in low:
        return []
    legs = []
    for chunk in re.split(r"\bunion all\b", low):
        # the first leg starts after the outer query's '(' ; the last ends at the ')' closing it
        opens = [m.start() for m in re.finditer(r"\(\s*select\b", chunk)]
        while opens and chunk[opens[-1]:].count("(") > chunk[opens[-1]:].count(")"):
            chunk = chunk[opens[-1] + 1:]
            opens = [m.start() for m in re.finditer(r"\(\s*select\b", chunk)]
        depth = 0
        for k, ch in enumerate(chunk):
            depth += {"(": 1, ")": -1}.get(ch, 0)
            if depth < 0:
                chunk = chunk[:k]
                break
        legs.append(chunk)
    defect = [_without_subqueries(l) for l in legs if "gold_meter_defect_event_day_agg" in _without_subqueries(l)]
    installed = [_without_subqueries(l) for l in legs if "gold_mi_installation_" in _without_subqueries(l)]
    if not defect or not installed:
        return []
    wanted = {(c, v) for l in defect for c, v in _PLACE_FILTER_RE.findall(l)}
    have = {(c, v) for l in installed for c, v in _PLACE_FILTER_RE.findall(l)}
    return [f"the installed-meters leg also needs {c} = '{v}' (the defect leg filters it; the OEM subquery alone does "
            "not narrow the installed base)" for c, v in sorted(wanted - have)]


def _stock_stage(words: str) -> str:
    return next(s for s in ("technician", "supervisor", "agenc", "warehouse") if s in words.lower()).replace("agenc", "agency")


_WAREHOUSE_RE = re.compile(r"\b(?!(?:a|an|the|each|every|per|by|any|one|this|that|which|all|some)\b)[a-z]+[ _-]?"
                           r"(?:hw|zw|swh|warehouse)\b", re.I)


_NON_COMM_TYPES = ("'never comm'", "'meter not in communication'")
_NEVER_COMM_RE = re.compile(r"\bnever[- ]?comm", re.I)
_NOT_IN_COMM_RE = re.compile(r"\bnot in communication\b", re.I)
_OM_AGEING_RE = re.compile(r"\b(ageing|aging|age bands?|age wise|by age)\b", re.I)


def _om_problems(question: str, low: str, asked: str | None = None) -> list[str]:
    """O&M card mistakes seen live on 26 Oct 2026 (each against the WFM dashboard's own SQL). `asked` is the
    user's own wording: triage once widened 'Never comm tickets ageing' to both non-comm types."""
    asked = asked or question
    problems: list[str] = []
    # Technician-days are distinct (technician, day) pairs: the per-category rows are split by meter type, so
    # SUM(technician_day_count) counted a technician twice (2.42 instead of the card's 2.77); the month table's
    # period_start is the month, so COUNT(DISTINCT period_start) there gave '1 active day' (20.36 per day).
    if "gold_om_technician_productivity_" in low:
        if re.search(r"\bsum\s*\(\s*(?:\w+\.)?technician_day_count\s*\)", low):
            problems.append("O&M technician-days = COUNT(DISTINCT (technician_name, period_start)) on "
                            "gold_om_technician_productivity_day_agg, never SUM(technician_day_count)")
        if "gold_om_technician_productivity_month_agg" in low and re.search(r"productiv|per day|per technician",
                                                                            question, re.I):
            problems.append("O&M productivity reads gold_om_technician_productivity_day_agg (day rows), not the "
                            "month table -- its period_start is the month, so days and technician-days are lost")
    # escalation_count counts escalation events; the card counts tickets (Agra pending 3,896 instead of 3,895).
    if "gold_om_discom_escalation_snapshot" in low and "escalation_count" in low:
        problems.append("escalated tickets = SUM(ticket_count) (escalation_count counts repeat escalations of one "
                        "ticket) -- the card counts tickets")
    # The escalations card's summary includes the discom's average days to decide (sum / sum).
    if ("gold_om_discom_escalation_snapshot" in low and not re.search(r"\bgroup\s+by\b", low)
            and "attend_days_sum" not in low):
        problems.append("the escalations summary also gives the discom's average days to attend: "
                        "ROUND(SUM(attend_days_sum)::numeric / NULLIF(SUM(attended_count), 0), 2)")
    # The detail snapshot holds every open ticket; the Never / Non-comm card is its two communication types.
    if "gold_om_open_ticket_detail_snapshot" in low:
        if not any(t in low for t in _NON_COMM_TYPES):
            problems.append("Never / Non-comm reads gold_om_open_ticket_detail_snapshot WHERE complaint_type IN "
                            "('NEVER COMM', 'METER NOT IN COMMUNICATION') -- the table holds every open ticket")
        elif not _grouped_by(low, "complaint_type"):
            one_type = not re.search(r"\bnon[- ]?comm|communication fail", asked, re.I)
            if one_type and _NEVER_COMM_RE.search(asked) and not _NOT_IN_COMM_RE.search(asked) and _NON_COMM_TYPES[1] in low:
                problems.append("the question asks about NEVER COMM tickets only -- complaint_type = 'NEVER COMM'")
            elif one_type and _NOT_IN_COMM_RE.search(asked) and not _NEVER_COMM_RE.search(asked) and _NON_COMM_TYPES[0] in low:
                problems.append("the question asks about METER NOT IN COMMUNICATION tickets only -- complaint_type = "
                                "'METER NOT IN COMMUNICATION'")
            elif (all(t in low for t in _NON_COMM_TYPES)
                  and not re.search(r"filter\s*\(\s*where\s+complaint_type", low)):
                # Live 26 Oct: 'communication failure tickets in Prayagraj II' gave 95,418 with no split.
                problems.append("a non-comm count also gives its split, like the card: SUM(ticket_count) FILTER "
                                "(WHERE complaint_type = 'NEVER COMM') AS never_comm and ... 'METER NOT IN "
                                "COMMUNICATION' AS not_in_communication")
    # HHU rows exist for today (partial uploads) but the card's window ends yesterday. Live 26 Oct: 'HHU readings
    # circle wise in Azamgarh zone in 2026' ran to '2026-12-31' -- EDC-MAU 68 instead of the card's 55.
    if "gold_om_hhu_reading_day_agg" in low:
        import datetime as _dt
        today = _dt.date.today().isoformat()
        late = [d for d in re.findall(r"'(\d{4}-\d{2}-\d{2})'", low) if d >= today]
        if late or not re.search(r"current_date\s*-\s*1\b|current_date\s*-\s*interval\s*'1 day'|'\d{4}-\d{2}-\d{2}'",
                                 low):
            problems.append("HHU windows end yesterday (period_start <= CURRENT_DATE - 1): the table holds today's "
                            "partial uploads, which the card never shows")
    # 'Tickets closed per technician per day' for a place is the team figure. Live 26 Oct: Triveni came back as
    # one row per technician-day, averaged to 1.67 instead of the card's 1.94.
    if ("gold_om_technician_productivity_" in low and _grouped_by(low, "technician_name")
            and not re.search(r"\b(top|best|worst|lowest|highest|bottom|most|least|which technician|each technician|"
                              r"technician[- ]?wise|"
                              r"by technician|per technician (?:list|wise)|technicians|named|support|performer)\b",
                              asked, re.I)):
        problems.append("O&M productivity for a place is the team figure: one row (or one per place) of SUM(complaint_count) "
                        "/ COUNT(DISTINCT (technician_name, period_start)) -- not one row per technician")
    # Complaint meters by SAT status = the tab's tiles (counts), not a meter list. Live 26 Oct: 'Complaint meters SAT
    # status for Triveni in September 2026' listed meters.
    if ("gold_om_meter_complaints_agg" in low and re.search(r"\bsat (?:status|done)\b|\bnon[- ]?sat\b", asked, re.I)
            and not re.search(r"\b(top|most|which meters|list|worst)\b", asked, re.I) and "count(" not in low):
        problems.append("complaint meters by SAT status = the tab's tiles: COUNT(*) meters, SAT Done / Non-SAT meters and "
                        "%, total complaints and per SAT status -- not a list of meters")
    # Active ticket ageing card: every answer carries SAT Done / SAT Pending beside the bands.
    if ("gold_om_open_ageing_snapshot" in low and "age_key" in low and "sat_status" not in low
            and _OM_AGEING_RE.search(question)):
        problems.append("the O&M Active Ticket Ageing card shows SAT Done and SAT Pending beside the bands: add "
                        "SUM(ticket_count) FILTER (WHERE sat_status = 'SAT DONE') AS sat_done and ... 'SAT PENDING' "
                        "AS sat_pending")
    # The productivity table holds roll-up rows ('ALL' + one per category x meter type). Live 27 Oct: ARYENDRA's
    # September closures came back as 26 (13 counted twice) with no category filter.
    if "gold_om_technician_productivity_day_agg" in low and not re.search(r"\bcategory\s*(?:=|<>|!=|in\s*\()", low):
        problems.append("gold_om_technician_productivity_day_agg holds roll-up rows: filter category = 'ALL' (every "
                        "ticket once) or one category -- with no category filter every ticket counts twice")
    # Month wise on the day table needs a month bucket. Live 27 Oct: 'O&M productivity month wise for DT complaints
    # in 2026' grouped by period_start and returned 105 day rows labelled by month.
    if ("gold_om_technician_productivity_day_agg" in low
            and re.search(r"\bmonth(?:ly|[- ]?wise)\b|\beach month\b|\bby month\b|\btrend\b", question, re.I)
            and not re.search(r"\bdaily\b|\bday[- ]?wise\b|\bweekly\b|\bweek[- ]?wise\b", question, re.I)
            and re.search(r"\bgroup\s+by\b[^;]*\bperiod_start\b", low)
            and not re.search(r"date_trunc\s*\(\s*'month'", low)):
        problems.append("month wise = one row per month: GROUP BY date_trunc('month', period_start) (period_start is "
                        "the day here), technician-days still COUNT(DISTINCT (technician_name, period_start))")
    # Tickets closed by a technician are ticket rows by work completion: QC tickets have no closed_date and most
    # non-QC ones none either. Live 27 Oct: ARYENDRA's tickets 'not found' via closed_date (11 non-QC + 2 QC exist).
    if (re.search(r"\b(?:list|show|which)\b.*\btickets?\b.*\bclosed by\b|\btickets? closed by technician\b", asked, re.I)
            and "work_completed_at" not in low):
        problems.append("tickets a technician closed = ticket rows by work completion: unified_non_qc_complaints UNION "
                        "ALL unified_complaints WHERE work_completed_at in the window AND technician_name ILIKE "
                        "'%<name>%' (closed_date is mostly empty)")
    return problems


_MAPPING_TABLES = ("mapping_audit_summary", "mapping_audit_feeder", "mapping_audit", "mapping_audit_candidates",
                   "mapping_audit_evidence")
_MONTHS = {m: i for i, m in enumerate(("january", "february", "march", "april", "may", "june", "july", "august",
                                       "september", "october", "november", "december"), 1)}


# Technical dashboard cards (power quality / DT load / load violation): one row per meter (or DT) per month.
_TECHNICAL_TABLES = ("voltage_deviation_index", "voltage_fluctuation_index", "voltage_unbalance_index", "voltage_drop",
                     "current_unbalance", "pct_time_out_of_band", "pct_time_current_imbalance", "low_voltage_pockets",
                     "low_voltage_pockets_meters", "dt_loading", "dt_peak_loading", "dt_peak_loading_feeder",
                     "dt_load_kva", "dt_load_rise", "load_violation", "load_assessed_consumers",
                     # 10 Nov 2026: Operation Analytics / Parameters, power factor, theft, energy audit cards
                     "voltage_imbalance", "current_imbalance", "voltage_imbalance_alerts", "current_imbalance_alerts",
                     "voltage_imbalance_reduction", "current_imbalance_reduction", "reverse_flow", "data_quality_issues",
                     "data_quality_issues_daily", "communication_failures", "hidden_outage_summary",
                     "hidden_outage_events", "consumption_spike_drop", "consumption_spike_drop_detail",
                     "lpf2_low_power_factor", "lpf2_low_power_factor_meters", "lpf2_low_power_factor_daily",
                     "power_factor_deterioration", "power_factor_deterioration_meters", "reliability_indices",
                     "outage_ranking", "reliability_improvement_trends", "overload_md_breach", "load_diversion",
                     "energy_theft_suspect", "theft_risk_by_area", "theft_reduction_trend", "total_anomalies_detected",
                     "repeat_anomaly_tracking", "feeder_loss", "feeder_loss_consumers", "dt_loss", "dt_loss_consumers",
                     "feeder_loss_ai", "feeder_loss_ai_consumers")
_THREE_PHASE_TABLES = ("voltage_unbalance_index", "current_unbalance", "pct_time_current_imbalance")
TECHNICAL_DEFAULT_MONTH = ("(SELECT MAX(month_date) FROM (SELECT month_date FROM hes_gold.dt_load_profile WHERE "
                           "month_date >= DATE '2026-06-01' GROUP BY month_date HAVING COUNT(DISTINCT data_date) >= 10) c)")


_PROJECT_ZONES = {"agra": ("agra", "aligarh", "etah", "firozabad", "mathura"),
                  "kashi": ("azamgarh", "varanasi i", "varanasi ii"),
                  "triveni": ("mirzapur", "prayagraj i", "prayagraj ii")}


_SILVER_READINGS = re.compile(r"\bhes_silver\.(block_usage|daily_usage)\b", re.I)


def _meter_reading_problems(question: str, sql_texts: list[str]) -> list[str]:
    """hes_silver.block_usage (~3.1 billion rows, partitioned only by discom_code / data_date: one meter-day scans
    0.4-0.8 GB) and daily_usage: one meter and a short literal date range, always (measured 10 Nov 2026)."""
    from datetime import date
    problems: list[str] = []
    for s in sql_texts:
        low = (s or "").lower()
        for t in sorted(set(_SILVER_READINGS.findall(low))):
            if not re.search(r"\bmeter_no\s*(?:=\s*'|in\s*\(\s*'|in\s*\(\s*select\s+meter_no\s+from\s+hes_gold\.dim_meter\b)", low):
                problems.append(f"hes_silver.{t} is huge: filter meter_no = '<serial>' (or meter_no IN (SELECT meter_no "
                                "FROM hes_gold.dim_meter WHERE consumer_id = ...)) -- never a place filter instead")
            days = re.findall(r"date\s*'(\d{4}-\d{2}-\d{2})'", low)
            if not re.search(r"\bdata_date\s*(?:=|between\b|>=|>|<=|<|in\s*\()", low) or not days:
                problems.append(f"hes_silver.{t}: pin data_date with DATE literals (one day for half-hour readings; no "
                                "date asked = DATE '2026-08-02' on block_usage, July 2026 on daily_usage)")
            elif len(days) >= 2 and (date.fromisoformat(max(days)) - date.fromisoformat(min(days))).days > \
                    (3 if t == "block_usage" else 32):
                problems.append("block_usage: at most 3 days per query (~0.5 GB a day) -- use daily_usage for consumption "
                                "over a longer range" if t == "block_usage" else "daily_usage: at most one month per query")
            if re.search(r"select\s+\*", low):
                problems.append(f"hes_silver.{t}: name the columns (no SELECT *): every extra column is scanned")
            if t == "block_usage" and re.search(r"\bmax\s*\(\s*data_date\s*\)", low):
                problems.append("block_usage: never MAX(data_date) over the table (full scan) -- the latest full day is "
                                "DATE '2026-08-02'")
            if t == "block_usage" and re.search(r"\bpower_factor_total\b", low):
                problems.append("power_factor_total is always NULL: PF = wh_import / NULLIF(vah_import, 0)")
            if re.search(r"\bkwh_import\b|\bkvah_import\b", low) and not re.search(r"\bas\s+kwh_import\b|\bas\s+kvah_import\b", low):
                problems.append("hes_silver energy columns are wh_import / wh_export / vah_import (Wh): kWh = "
                                "wh_import / 1000.0 (kwh_import does not exist)")
        # Live 10 Nov: 'total units used by AL1159584 last month' read October on daily_usage (no data after August).
        if _SILVER_READINGS.search(low) and re.search(r"date\s*'2026-(?:09|1[0-2])-\d{2}'", low):
            problems.append("HES readings run June - early August 2026: 'last month' / 'this month' = July 2026 "
                            "(daily_usage) or the latest full day DATE '2026-08-02' (block_usage), never the calendar month")
        # Live 10 Nov: replaced meter AL1228741 'not found' -- swapped-out meters live only in dim_meter_replaced.
        if re.search(r"\bhes_gold\.dim_meter\b", low) and re.search(r"\b(?=[A-Za-z]{1,4}\d)[A-Za-z]{1,4}\d{6,10}\b", question) \
                and "dim_meter_replaced" not in low and not _SILVER_READINGS.search(low):
            problems.append("one meter's master row: dim_meter UNION ALL dim_meter_replaced (a replaced meter is only in "
                            "hes_gold.dim_meter_replaced) -- say which one it was found in")
        if re.search(r"\bhes_gold\.dim_meter_(?:energy|asof_\w+)\b", low):
            problems.append("use hes_gold.dim_meter (current) + dim_meter_replaced; dim_meter_energy is retired and "
                            "dim_meter_asof_* is a frozen 8 Sep copy")
    return problems


def _card_problems_nov10(question: str, sql_texts: list[str], asked: str | None = None) -> list[str]:
    """Cards added 10 Nov 2026. Each rule names the wrong figure it prevents, measured on live July data."""
    problems: list[str] = []
    q = question.lower()
    everyone = re.search(r"\b(?:all|every|fleet|assessed|checked|scored|compared|total meters)\b", q)
    for s in sql_texts:
        low = (s or "").lower()
        has = lambda t: re.search(rf"\bhes_gold\.{t}\b", low)  # noqa: E731
        counts = re.search(r"\bcount\s*\(|\bcount_if\s*\(|\bsum\s*\(", low)
        # Reduction tables hold every meter compared: unfiltered = ~168,000 'improved' (card 1,522 / 735).
        if (has("voltage_imbalance_reduction") or has("current_imbalance_reduction")) and counts \
                and not re.search(r"\bqualifies(?:_live|_relative)?\b", low) and not everyone:
            problems.append("the reduction tables hold every meter compared: count improved meters WHERE qualifies "
                            "(imbalance fell 30+ points)")
        # Low PF card counts Borderline-or-worse only (796,038 unfiltered vs 711,642); max ageing is a MAX (17, not 173,732).
        if has("lpf2_low_power_factor") and re.search(r"\bunique_meter_count\b", low) and "severity" not in low:
            problems.append("the Low PF card counts graded meters only: add severity NOT IN ('Normal', 'NoData')")
        # Live 10 Nov (PF hard set): flagged meters on the per-meter table without the 3-day ageing rule ->
        # 505 DTs for 484, 704,059 'both months' for 483,550 (the card only holds meters aged 3+ days).
        if has("lpf2_low_power_factor_meters") and re.search(r"\bseverity\s*(?:not\s+in\b|in\s*\(|=)", low) \
                and not re.search(r"\bageing_days\s*>=?\s*[1-9]|\bageing_bucket\b", low) \
                and not re.search(r"\bmeter_no\s*=\s*'", low):
            problems.append("a Low PF finding = severity NOT IN ('Normal', 'NoData') AND ageing_days >= 3 on "
                            "lpf2_low_power_factor_meters (the card's rule; 868,698 without the ageing filter vs 711,642)")
        # Live 10 Nov: 'median low PF % of flagged DT meters' took a percentile of rollup group means (40.19 vs 62.9).
        if has("lpf2_low_power_factor") and re.search(r"\bapprox_percentile\s*\(\s*(?:\w+\.)?avg_low_pf_pct\b", low):
            problems.append("a median / percentile of low PF % is over meters: APPROX_PERCENTILE(low_pf_pct, p) FROM "
                            "hes_gold.lpf2_low_power_factor_meters WHERE severity NOT IN ('Normal', 'NoData') AND "
                            "ageing_days >= 3")
        if re.search(r"\bsum\s*\(\s*max_ageing_days\b", low):
            problems.append("max_ageing_days is a maximum: combine groups with MAX(max_ageing_days), never SUM")
        # Area reliability indices are re-derived from sums (AVG(saidi) is a mean of feeders, not the card's 3,004.44).
        if has("reliability_indices") and re.search(r"\bavg\s*\(\s*(?:\w+\.)?(?:saidi|saifi|caidi|caifi|maifi)\b", low):
            problems.append("an area's index is re-derived from sums: SAIDI = SUM(dur) / SUM(total_consumers), SAIFI = "
                            "SUM(events) / SUM(total_consumers), CAIDI = SUM(dur) / SUM(events), CAIFI = SUM(events) / "
                            "SUM(affected), MAIFI = SUM(momentary) / SUM(total_consumers) -- never AVG of feeder ratios")
        # Energy audit: AVG(loss_pct) = -187.82% (feeders) vs the card's 61.42%; bare current_date is today in Athena.
        loss = any(has(t) for t in ("feeder_loss", "dt_loss", "feeder_loss_ai"))
        if loss and re.search(r"\bavg\s*\(\s*(?:\w+\.)?loss_pct\b", low):
            problems.append("area loss is energy-weighted: SUM(loss_kwh) * 100.0 / SUM(net_input), never AVG(loss_pct)")
        if loss and re.search(r'(?<!")\bcurrent_date\b(?!")', s or ""):
            problems.append('the window-end column is "current_date" in double quotes (bare current_date is today)')
        if loss and re.search(r"\bdata_date\s*=\s*date\s*'2026-(?:0[89]|1[0-2])-01'", low):
            problems.append("energy-audit tables hold June and July 2026 only (August was not computed): say so "
                            "instead of pinning August")
        # Slot / reading tables are 300M+ / 92M rows a month: one-meter lookups only.
        for big, small in (("hidden_outage_events", "hidden_outage_summary"),
                           ("consumption_spike_drop_detail", "consumption_spike_drop")):
            if has(big) and not re.search(r"\bmeter_no\s*(?:=|in\s*\()", low):
                problems.append(f"hes_gold.{big} is a per-slot table (hundreds of millions of rows): counts come from "
                                f"hes_gold.{small}; use {big} only WHERE meter_no = 'X'")
        # Flag filters: reverse_flow / consumption_spike_drop / energy_theft_suspect hold every meter checked.
        if has("reverse_flow") and counts and not re.search(r"\bis_anomaly\b|\banomaly_type\b|\bmeter_no\s*=", low) \
                and not everyone:
            problems.append("reverse_flow holds every consumer checked: flagged = COUNT_IF(is_anomaly) (133,844 in July, "
                            "not 3.75M)")
        if has("consumption_spike_drop") and counts and not re.search(r"\bis_spike_or_drop_suspect\b|\bmeter_no\s*=", low) \
                and not everyone:
            problems.append("consumption_spike_drop holds every meter checked: flagged = WHERE is_spike_or_drop_suspect")
        if has("energy_theft_suspect"):
            if re.search(r"\bdata_date\s*=", low) and "month_date" not in low:
                problems.append("energy_theft_suspect has one row per meter per DAY: filter month_date = <month>, not "
                                "data_date (that is a single day)")
            if counts and not re.search(r"\bis_theft_suspect\b|\bmeter_no\s*=", low) and not everyone:
                problems.append("energy_theft_suspect holds every meter: suspects = COUNT(DISTINCT meter_no) ... AND "
                                "is_theft_suspect")
            if re.search(r"\bcount\s*\(\s*\*\s*\)", low):
                problems.append("energy_theft_suspect rows are meter-days: count COUNT(DISTINCT meter_no)")
        if re.search(r"\bsum\s*\(\s*(?:\w+\.)?assessed_meters\b(?:(?!\bfrom\b).)*\bfrom\s+hes_gold\.overload_md_breach\b",
                     low, re.S):
            problems.append("assessed_meters repeats on every overload_md_breach row: never SUM it -- the denominator is "
                            "SUM(assessed_meters) FROM hes_gold.load_assessed_consumers with the same filters")
        if re.search(r"\bavg\s*\(\s*(?:\w+\.)?reduction_pct\b", low):
            problems.append("reduction % = (SUM(prev_suspects) - SUM(curr_suspects)) * 100.0 / NULLIF(SUM(prev_suspects), 0)")
        if re.search(r"\bavg\s*\(\s*(?:\w+\.)?pct_blocks_flagged\b", low):
            problems.append("% slots flagged = SUM(total_block_count_flagged) * 100.0 / SUM(total_block_count), never "
                            "AVG(pct_blocks_flagged)")
        hes = [t for t in _TECHNICAL_TABLES if has(t)]
        # Live 10 Nov: 'meter data in Varanasi zones last month' pinned DATE '2026-10-01' (no HES data after August).
        if hes and re.search(r"\b(?:data_date|month_date)\s*=\s*date\s*'2026-(?:09|1[0-2])-01'", low) \
                and not re.search(r"\b(?:predict|forecast|project(?:ed|ion)?|next month|if the trend|will be)\b", q):
            problems.append("HES data runs June-August 2026: 'last month' / 'this month' / 'latest' = the dashboard "
                            f"default month {TECHNICAL_DEFAULT_MONTH}, never the calendar month")
        # Live 10 Nov (PF hard set): 'how many meters will have low PF in September 2026?' pinned September -> '0 meters'.
        if hes and re.search(r"\b(?:data_date|month_date)\s*(?:=|>=?)\s*date\s*'2026-(?:09|1[0-2])-01'", low) \
                and re.search(r"\b(?:predict\w*|forecast\w*|project(?:ed|ion)?|next month|if the trend|will be|"
                              r"will have|likely)\b", q):
            problems.append("HES data runs June-August 2026, so a later month has no rows: a forecast queries the "
                            "month-wise series (data_date BETWEEN DATE '2026-06-01' AND DATE '2026-08-01' GROUP BY "
                            "data_date) and projects from June and July -- never pin the future month")
        # Live 10 Nov: category = 'CONSUMER' (WFM's value) on current_imbalance -> 0 meters (card 290 of 52,720).
        # Live 10 Nov: 'Power factor deterioration in Agra' filtered LOWER(zone) = 'agra' (38,393; the project's
        # five zones hold 291,868). 'Agra' / 'Kashi' / 'Triveni' without the word zone = the project.
        for proj, zones in _PROJECT_ZONES.items():
            if hes and re.search(rf"\b{proj}\b", q) and not re.search(r"\bzones?\b", q) \
                    and re.search(rf"\bzone\s*\)?\s*(?:=|like)\s*'%?{proj}%?'", low) and "cluster" not in low:
                problems.append(f"'{proj.title()}' is the project: its zones UPPER(zone) IN "
                                f"({', '.join(repr(z.upper()) for z in zones)}) (or LOWER(cluster) = '{proj}' where the "
                                "table has cluster) -- not the single zone")
        # Live 10 Nov (PF hard set): 'consumers in Varanasi zones' read as the Kashi project and added AZAMGARH
        # (185,581 vs 117,449). '<stem> zone(s)' = the zones named after that stem only.
        # Read the user's own words: triage rewrote the question as 'Varanasi (Kashi) project'.
        user = (asked or question).lower()
        for stem, zones, others in (("varanasi", ("VARANASI I", "VARANASI II"), ("azamgarh",)),
                                    ("prayagraj", ("PRAYAGRAJ I", "PRAYAGRAJ II"), ("mirzapur",))):
            if hes and re.search(rf"\b{stem}\s+(?:i+\s+(?:and|&|,)\s+(?:{stem}\s+)?i+\s+)?zones?\b", user) \
                    and not re.search(r"\b(?:kashi|triveni|project)\b", user) \
                    and any(f"'{o}'" in low for o in others if not re.search(rf"\b{o}\b", user)):
                problems.append(f"'{stem.title()} zones' = UPPER(zone) IN ({', '.join(repr(z) for z in zones)}) only -- "
                                "not the whole project")
        # Live 10 Nov: category = 'DT' on total_anomalies_detected -> 0 rows (stored 'dt_meter').
        if hes and re.search(r"\bcategory\s*(?:=|in\s*\([^)]*?)\s*'(?:dt|dtr|dt meter|ht|1ph|3ph|lt|feeder meter)'", low):
            problems.append("HES category values are 'dt_meter', 'feeder', 'consumer_1_phase', 'consumer_3_phase', "
                            "'consumer_ht' -- never 'DT' / 'DTR' / 'HT'")
        # Live 10 Nov: 'Power factor deterioration in Agra' added the bands (361,738; a meter sits in several bands)
        # instead of the distinct total 291,868.
        # Live 10 Nov (PF hard set): 'for each project ... PF deterioration meters' summed every band too (361,738).
        if has("power_factor_deterioration") and not has("power_factor_deterioration_meters") \
                and (not re.search(r"\bbands?\b|-?\bwise\b|\bby\b|\beach\b|\bsplit\b|\bbreak", q)
                     or "pf_band" not in low):
            problems.append("a meter can sit in several PF bands: the total = COUNT(DISTINCT meter_no) FROM "
                            "hes_gold.power_factor_deterioration_meters (same month / place); give the band rows beside it, "
                            "never their sum")
        # Live 10 Nov: 'Did theft go down compared to June?' pinned June (the June row compares May -> June).
        if has("theft_reduction_trend") and re.search(r"\b(?:compared? (?:to|with)|vs\.?|versus|than|since|from) june\b", q) \
                and re.search(r"\bdata_date\s*=\s*date\s*'2026-06-01'", low):
            problems.append("theft_reduction_trend is dated by the CURRENT month: June -> July is data_date = "
                            "DATE '2026-07-01' (prev_period '2026-06')")
        if "unified_discom_escalations" in low:
            # Live 10 Nov: technician 'VI_ADITYA PATEL_8063894 PROBE' -> exact match on a shortened name, 0 rows.
            if re.search(r"\btechnician_name\s*\)?\s*\)?\s*=", low):
                problems.append("technician names carry suffixes (e.g. '... PROBE'): match UPPER(technician_name) "
                                "LIKE '%<NAME>%', never =")
            # The log holds consumer name / mobile / address -- never returned (G23 selected consumer_name).
            if re.search(r"\b(?:consumer_name|mobile_no|mobile_number|address|technician_number|"
                         r"escalation_supervisor_number|latitude|longitude)\b", low):
                problems.append("never return consumer_name / mobile_no / address / phone numbers / coordinates from "
                                "the escalation log -- ticket, meter, place, status and dates only")
        # Live 10 Nov: 'Pending escalations raised by technician X' read the gold card (no technician column).
        if re.search(r"\bescalat", q) and re.search(r"\btechnician\s+(?!remark|wise|-wise)[a-z_]+", q) \
                and "gold_om_discom_escalation_snapshot" in low and "technician_name" not in low:
            problems.append("one technician's escalations are in gmr_wfm_silver.unified_discom_escalations "
                            "(technician_name LIKE '%NAME%', latest escalation per ticket) -- the gold card has no technician")
        if hes and re.search(r"\bcategory\s*(?:=|in\s*\()\s*'consumer'", low):
            problems.append("HES meter types: category 'consumer_1_phase' / 'consumer_3_phase' / 'consumer_ht' / "
                            "'dt_meter' / 'feeder' (never 'CONSUMER'); consumers = category LIKE 'consumer%'")
        # Live 10 Nov: 'Low power factor for DT and feeder meters' filtered category = 'feeder' only.
        cat_filter = re.findall(r"\bcategory\s*(?:=\s*'([^']*)'|in\s*\(([^)]*)\))", low)
        if hes and cat_filter:
            vals = " ".join(a or b for a, b in cat_filter)
            for words, val in ((r"\bdts?\b|\bdtrs?\b|\bdt meters?\b|\btransformers?\b", "dt_meter"),
                               (r"\bfeeders?\b(?! loss)", "feeder")):
                if re.search(rf"(?:{words})[^?]*\b(?:and|&|,)\b|\b(?:and|&|,)\s*(?:{words})", q) and val not in vals:
                    problems.append(f"the question names both meter types: keep category '{val}' in the filter "
                                    "(category IN ('dt_meter', 'feeder') ... GROUP BY category)")
        # Live 10 Nov: 'feeders out of balance ... last month' took DATE_ADD('month', -1, <default month>) = June.
        if hes and re.search(r"\b(?:last|previous|this|current|latest) month\b", q) \
                and not re.search(r"\b(?:vs|versus|compare\w*|change\w*|rise|rose|fell|reduc\w*|improv\w*|than)\b", q) \
                and re.search(r"date_add\s*\(\s*'month'\s*,\s*-\s*1\s*,\s*(?:\(?\s*select\s+)?max\s*\(\s*month_date", low):
            problems.append("'last month' / 'this month' on an HES card = the dashboard default month itself "
                            f"({TECHNICAL_DEFAULT_MONTH}), not the month before it")
        if has("communication_failures") and re.search(r"\bnc_bucket\s*(?:=|in)\s*\(?\s*'\d", low):
            problems.append("nc_bucket is stored 'Non_Communicating_3_30_Days' style values, not '3-30 Days'")
    return problems


def _technical_problems(question: str, sql_texts: list[str]) -> list[str]:
    """Technical-card tables (6 Nov 2026 test): answers summed June-August (no month pin) or read August
    (MAX(data_date), 2-3 days of readings) instead of the dashboard's default month."""
    problems: list[str] = []
    month_wise = re.search(r"\b(month[- ]?wise|monthly(?! peak)|trend|each month|every month|by month|over (?:the )?months|"
                           r"predict\w*|forecast\w*|project(?:ed|ion)|next month|will be|likely|"
                           r"june to|from june|month[- ]on[- ]month|compare(?:d)? (?:june|july|august))\b", question, re.I)
    for s in sql_texts:
        low = (s or "").lower()
        used = [t for t in _TECHNICAL_TABLES if re.search(rf"\bhes_gold\.{t}\b", low)]
        if not used:
            continue
        # Live 10 Nov: 'Month-wise load diversion suspects' joined the 10-day months only -> July alone.
        if month_wise and re.search(r"count\s*\(\s*distinct\s+data_date\s*\)\s*>=\s*10", low) \
                and not re.search(r"select\s+max\s*\(\s*month_date\s*\)", low):
            problems.append("month-wise = every month with data: data_date BETWEEN DATE '2026-06-01' AND DATE "
                            "'2026-08-01' GROUP BY data_date (label August partial) -- the 10-day rule only picks the "
                            "default month, it never drops months from a series")
        # reliability_improvement_trends: July holds two comparisons (months on 07-01, halves on 07-16) -> a range.
        if not month_wise and not re.search(r"\b(?:data_date|month_date)\s*(?:=|between\b|in\s*\()", low) \
                and not re.search(r"\bmeter_no\s*=\s*'", low) \
                and not (used == ["reliability_improvement_trends"] and re.search(r"\bdata_date\s*>=", low)):
            problems.append("these tables hold one row per meter per MONTH: pin one month -- data_date (month_date on "
                            f"dt_load_kva) = {TECHNICAL_DEFAULT_MONTH} when no month is asked (the dashboard's default "
                            "month), or DATE '2026-07-01' for a named month. Never add months together")
        # Meter type on the power-quality tables is `role`. Live 6 Nov: 'average voltage unbalance for DT meters'
        # filtered meter_type = '3PLTCTSM' (WFM's DT code) -> no rows.
        pq_tables = {"voltage_deviation_index", "voltage_fluctuation_index", "voltage_unbalance_index", "voltage_drop",
                     "current_unbalance", "pct_time_out_of_band", "pct_time_current_imbalance"}
        if pq_tables & set(used) and re.search(r"\bmeter_type\s*(?:=|in\b|like\b)", low):
            problems.append("meter type on these tables is role: '1ph', '3ph', 'ht', 'dtr' (DT meters), 'feeder' -- "
                            "filter role = 'dtr', never meter_type")
        # A meter can sit in several low-voltage bands: a Total over band rows counts it twice. Live 6 Nov:
        # ROLLUP (band, ageing_bucket) over SUM(unique_meter_count) said 891,805 meters (card: 621,312 distinct).
        if "low_voltage_pockets" in used and re.search(r"sum\s*\(\s*unique_meter_count\s*\)", low) \
                and re.search(r"\brollup\b", low):
            problems.append("low-voltage meters in several bands are counted once per band: no ROLLUP Total over "
                            "SUM(unique_meter_count) -- give the per-band rows, and the total as COUNT(DISTINCT "
                            "meter_no) FROM hes_gold.low_voltage_pockets_meters")
        # DT loading bands include their lower edge ('>120%' = 120 and above). Live 6 Nov: 'DT-days with peak loading
        # above 120%' filtered peak_pct_loading > 120 -> 758 (card 779; 21 DT-days sit at exactly 120%).
        if {"dt_load_kva", "dt_loading"} & set(used) and re.search(
                r"\b(?:peak_pct_loading|max_pct_loading|pct_loading)\s*>\s*(?:20|35|55|70|85|100|120)(?:\.0+)?\b(?!\s*=)", low):
            problems.append("DT loading bands include their lower edge: 'above 120%' = peak_band = '>120%' "
                            "(peak_block_band on dt_loading), i.e. >= 120 -- use the band column, not a strict >")
        # A breakdown nobody asked for. Live 6 Nov: 'Load violation for 3-phase consumers in August' was split by
        # project and the headline % (violators / assessed of the whole) was never given.
        broke = re.findall(r"\bgroup\s+by\s+(?:rollup\s*\()?\s*(?:\w+\.)?(cluster|zone|circle|division|sub_division|"
                           r"category|role|feeder_code|dtr_code)\b", low)
        if broke and not re.search(r"-?\bwise\b|\bby\b|\beach\b|\bper\b|\bwhich\b|\btop\b|\bcompare|\bvs\b|\bversus\b|"
                                   r"\bsplit\b|\bbreak|\bdistribution\b|\brank|\bhighest\b|\blowest\b|\bmost\b|\bleast\b|"
                                   r"\bworst\b|\bbest\b|\bacross\b|\blist\b|\bshow\b|\bwhere\b|\bbetween\b"
                                   r"|,[^?]*\band\b|\b\w+ and \w+ meters\b|\b(?:zones|circles|divisions|projects)\b"
                                   r"|\bi and ii\b", question, re.I):   # a listed set ('Agra, Kashi and Triveni') = a split
            problems.append(f"the question asks for one figure, not a {broke[0].replace('_', ' ')}-wise breakdown: "
                            "remove that GROUP BY and give the whole scope (e.g. violators, assessed and the % overall)")
        # Athena has no Postgres casts. Live 6 Nov: ROUND(COUNT(DISTINCT meter_no)::numeric, 2) -> 'mismatched input'.
        if re.search(r"::\s*\w+", low):
            problems.append("Athena SQL: no '::' casts -- use CAST(x AS DOUBLE) (or nothing for a count)")
        # AND ... OR ... without brackets: the OR side escapes the month / band filters. Live 6 Nov: 'critical low
        # voltage meters in Kashi' wrote band = 'Critical' AND zone IN (Agra) OR zone IN (Kashi) -> 63,481.
        flat = re.sub(r"--[^\n]*", "", low)
        while re.search(r"\([^()]*\)", flat):              # brackets (and the subqueries in them) -> x
            flat = re.sub(r"\([^()]*\)", "x", flat)
        for where in re.findall(r"\bwhere\b(.*?)(?=\bgroup\s+by\b|\border\s+by\b|\bhaving\b|\blimit\b|$)", flat, re.S):
            if re.search(r"\band\b", where) and re.search(r"\bor\b", where):
                problems.append("WHERE mixes AND with OR without brackets -- the OR side drops the month and other "
                                "filters: put the alternatives in brackets, e.g. AND (zone IN (...) OR ...)")
                break
        # One project asked: only its zones. Same query added Agra's zones to a Kashi question.
        asked_projects = [p for p in _PROJECT_ZONES if re.search(rf"\b{p}\b", question, re.I)]
        if len(asked_projects) == 1 and not _COMPARE_RE.search(question.lower()):
            for other, zones in _PROJECT_ZONES.items():
                if other != asked_projects[0] and re.search(rf"'{zones[0]}'", low):
                    problems.append(f"the question is about {asked_projects[0].title()} only: filter its zones "
                                    f"({', '.join(_PROJECT_ZONES[asked_projects[0]]).upper()}), not {other.title()}'s")
                    break
        # Live 6 Nov: 'DTs by monthly peak loading band' was answered as a June-August trend.
        # MAX(...) over a GROUP BY returns one row per qualifying month -- right today only because one month has
        # 10+ days (live 6 Nov, '% time beyond voltage band in Varanasi I zone').
        if re.search(r"max\s*\(\s*month_date\s*\)(?:\s+as\s+\w+)?\s+from\s+hes_gold\.dt_load_profile\s+"
                     r"(?:where\s[^)]*?)?group\s+by", low):
            problems.append(f"the default month must be one value: use exactly {TECHNICAL_DEFAULT_MONTH}")
        outer = re.sub(r"select\s+(?:max\s*\(\s*)?month_date\s*\)?(?:\s+as\s+\w+)?\s+from\s+hes_gold\.dt_load_profile[^)]*\)",
                       "", low)
        grouped = " ".join(re.findall(r"\bgroup\s+by\s+(?:rollup\s*\()?([\w\s,.]+)", outer))   # the GROUP BY lists only
        pinned = re.search(r"\b(?:data_date|month_date)\s*=", outer)   # grouping one pinned month is harmless
        if not month_wise and not pinned and re.search(r"\b(?:data_date|month_date)\b", grouped) \
                and not re.search(r"\b(?:june|july|august|months|vs|versus|compare)\b", question, re.I):
            problems.append("one month was asked (no month-wise / trend wording): pin data_date to the default month "
                            "and do not group by month")
        # Also 'SELECT MAX(data_date) AS m FROM hes_gold.dt_load_rise' (live 6 Nov: load-change bands for August).
        if re.search(rf"max\(\s*(?:data_date|month_date)\s*\)(?:\s+as\s+\w+)?\s+from\s+(?:hes_gold\.)?"
                     rf"(?:{'|'.join(_TECHNICAL_TABLES)})\b", low):
            problems.append("the default month is not MAX(data_date) (August 2026 holds only 2-3 days): use "
                            f"{TECHNICAL_DEFAULT_MONTH}")
        # Only months with data (June-August 2026): a later month is a forecast from the month-wise series, and
        # pinning it returns no rows (live 6 Nov: 'predict the average VDI for September 2026' -> '0 meters').
        named = [m for m in ("june", "july", "august") if re.search(rf"\b{m}\b", question, re.I)]
        if re.search(r"\b(?:predict|forecast|project(?:ed|ion)?|next month|if the trend|will be)\b", question, re.I):
            named = []
        # 'compared to June' names the baseline, not the month to pin (theft reduction / reliability comparisons).
        named = [m for m in named if not re.search(rf"\b(?:compared? (?:to|with)|vs\.?|versus|than|since|from) {m}\b",
                                                   question, re.I)]
        # The default-month subquery holds DATE '2026-06-01' itself -- it never pins June (live 6 Nov: 'load
        # violation percentage in June' ran on the default month and said June had no data).
        no_default = re.sub(r"\(\s*select\s+(?:max\s*\(\s*)?month_date\s*\)?(?:\s+as\s+\w+)?\s+from\s*\(?\s*(?:select\s+month_date\s+from\s+)?"
                            r"hes_gold\.dt_load_profile.*?count\s*\(\s*distinct\s+data_date\s*\)\s*>=\s*10\s*\)?\s*\w*\s*\)", "x", low, flags=re.S)
        if len(named) == 1 and not month_wise and f"-{_MONTHS[named[0]]:02d}-01" not in no_default:
            problems.append(f"the question names {named[0].title()}: pin DATE '2026-{_MONTHS[named[0]]:02d}-01' "
                            "(data_date = DATE '...', not the default month)")
        # Live 6 Nov: 'Load violation in EDC Ballia' took MAX(month_date) FROM dt_load_profile without the
        # 10-day condition and answered for August.
        if "dt_load_profile" in low and not re.search(r"count\s*\(\s*distinct\s+data_date\s*\)\s*>=\s*10", low):
            problems.append(f"the default month needs the 10-day condition -- use exactly {TECHNICAL_DEFAULT_MONTH}")
        # Athena: ROLLUP takes column references only. Live 6 Nov: 'Voltage unbalance by circle' grouped
        # ROLLUP (CASE WHEN circle IS NULL ... UPPER(circle) END) -> 'Queries of this type are not supported'.
        for inner in re.findall(r"rollup\s*\(([^)]*\([^)]*\)[^)]*|[^)]*\bcase\b[^)]*)\)", low):
            problems.append("Athena: GROUP BY ROLLUP (circle) on the bare column -- never on an expression "
                            f"({inner.strip()[:40]}...); label the Total row in SELECT with CASE WHEN GROUPING(circle) = 1")
            break
        for t in _THREE_PHASE_TABLES:
            if t in used and "blocks_3ph" not in low:
                problems.append(f"{t} is a 3-phase card: add blocks_3ph > 0 (the dashboard always applies it)")
        # Athena: a CASE label aliased with the grouped column's own name breaks GROUPING(col) in ORDER BY
        # (live 6 Nov: '% time current imbalance above 10% by circle').
        # (live 6 Nov again with peak_band: 'DTs by peak loading band' -- any grouped column, not only places)
        for m in re.finditer(r"\border\s+by\b[^;]*?\bgrouping\s*\(\s*(\w+)\s*\)", low):   # Athena fails only here
            col, head = m.group(1), low[:m.start()]
            # Subqueries out first: the default-month subquery's own SELECT is the last 'select' before ORDER BY
            # (live 10 Nov: B20 / D10 / D18 failed in Athena while this check looked only at that subquery).
            while re.search(r"\([^()]*\)", head):            # every bracket (the label sits outside them)
                head = re.sub(r"\([^()]*\)", "x", head)
            if re.search(rf"\bend\s+as\s+{col}\b", head[head.rfind("select"):]):     # ... in that same SELECT
                problems.append(f"Athena: the ROLLUP label must not be named {col} (alias it {col}_name)")
        # The global 'Unmapped' CASE label does not work here (live 6 Nov: 'which division has the highest % time
        # out of band' -> EXPRESSION_NOT_AGGREGATE): select and group the bare column.
        if "'unmapped'" in low and re.search(r"group\s+by\s+(?:rollup\s*\(\s*)?(?:upper|lower|case)\b", low):
            problems.append("on these tables select and GROUP BY the bare place column (circle / zone / division) -- "
                            "no 'Unmapped' CASE label; rankings add <col> IS NOT NULL")
        # The asked measure. Live 6 Nov: '% time under voltage by circle' answered the out-of-band total.
        if "pct_time_out_of_band" in used:
            for words, col in ((r"under[- ]?voltage|\buv\b|low voltage time", "uv_blocks"),
                               (r"over[- ]?voltage|\bov\b|high voltage time", "ov_blocks")):
                if re.search(words, question, re.I) and col not in low:
                    problems.append(f"the question asks for {col[:2].upper()} time: SUM({col}) * 100.0 / SUM(live_blocks)")
        # 'more than N points' = the card's band (lower bound included). Live 6 Nov: '> 20' gave 1,150, card 1,152.
        if "dt_load_rise" in used and re.search(r"load_rise_pp\s*>\s*\d", low) \
                and re.search(r"\b(?:more than|over|above)\s+\d+\s*(?:points|pp|percentage points)", question, re.I):
            problems.append("load-rise bands include their lower bound: 'more than 20 points' = load_rise_pp >= 20 "
                            "(the card's 'Rose >20 pp')")
        if (re.search(r"\b(?:which|top|most|highest|lowest|worst|best)\b", question, re.I)
                and re.search(r"\border\s+by\b", low)):
            col = next((c for c in ("sub_division", "division", "circle", "zone")
                        if re.search(rf"\bgroup\s+by\b[^;]*\b{c}\b", low)), None)
            if col and not re.search(rf"\b{col}\s+is\s+not\s+null", low):
                problems.append(f"a {col.replace('_', ' ')} ranking must skip the blank area: add {col} IS NOT NULL")
    return problems


def _mapping_problems(question: str, sql_texts: list[str]) -> list[str]:
    """Consumer-to-feeder mapping audit (the dashboard's mapping_audit / _feeder / _summary tables)."""
    problems: list[str] = []
    for s in sql_texts:
        low = (s or "").lower()
        if not any(f"hes_gold.{t}" in low or re.search(rf"\b{t}\b", low) for t in _MAPPING_TABLES):
            continue
        # One full copy per month: without a data_date pin, June + July + August are added together (3x).
        month_wise = re.search(r"\b(month[- ]?wise|monthly|trend|change|over time|each month|june to|from june)\b",
                               question, re.I)
        if "data_date" not in low and not month_wise:
            problems.append("the mapping audit tables hold one full copy per month: pin data_date = (SELECT "
                            "MAX(data_date) FROM hes_gold.mapping_audit_summary) for the latest month, or the "
                            "asked month (DATE '2026-07-01') -- never add months together")
        named = [m for m in _MONTHS if re.search(rf"\b{m}\b", question, re.I)]
        if len(named) == 1 and not month_wise and f"-{_MONTHS[named[0]]:02d}-01" not in low:
            problems.append(f"the question names {named[0].title()}: data_date = DATE '2026-{_MONTHS[named[0]]:02d}-01' "
                            "(the audit month), not the latest month")
        # Feeder names carry their code ('METHAPAR_41638210383MET'): an exact match finds nothing.
        if re.search(r"\bfeeder_name\s*=\s*'", low):
            problems.append("feeder_name holds the name plus its code ('METHAPAR_41638210383MET'): match with "
                            "UPPER(feeder_name) LIKE '%METHAPAR%' and list one row per feeder_code")
        # Athena rejects ORDER BY GROUPING(x) when a SELECT alias is also named x. Live 27 Oct: 'auto indexing
        # concluded percentage zone wise in Kashi' failed with 'Invalid reference to output of SELECT clause'.
        for col in set(re.findall(r"grouping\s*\(\s*(\w+)\s*\)", low)):
            if re.search(rf"grouping\s*\(\s*{col}\s*\)[^,]*?\bend\s+as\s+{col}\b", low):
                problems.append(f"Athena: the ROLLUP label must not be named {col} (alias it {col}_name) -- an alias "
                                f"with the grouped column's name breaks GROUPING({col}) in ORDER BY")
        # A named project must filter: live 27 Oct 'concluded % zone wise in Kashi' labelled zones Kashi / Other in
        # a CASE and returned every zone.
        for proj, zones in (("agra", "'agra'"), ("kashi", "'azamgarh'"), ("triveni", "'mirzapur'")):
            if (re.search(rf"\b{proj}\b", question, re.I) and not _COMPARE_RE.search(question.lower())
                    and not re.search(rf"(?:where|and)\s+(?:upper\(\s*(?:\w+\.)?zone\s*\)\s+in\s*\([^)]*{zones}"
                                      rf"|lower\(\s*(?:\w+\.)?cluster\s*\)\s*(?:=|in\s*\()\s*'{proj}'"
                                      rf"|(?:\w+\.)?cluster\s*=\s*'{proj}'|\w*proj\w*\s*=\s*'{proj}')", low)):
                problems.append(f"the question is about {proj.title()} only: filter it in WHERE -- LOWER(cluster) = "
                                f"'{proj}' on mapping_audit, or UPPER(zone) IN (its zones) on the summary / feeder tables")
        # A feeder NAME in feeder_code finds nothing. Live 27 Oct: feeder_code = 'LATGHAT' -> '0 review and no data'.
        if re.search(r"\bfeeder_code\s*=\s*'[a-z ]+'", low):
            problems.append("feeder_code values look like '42601220265LAT': a feeder named in the question is "
                            "UPPER(feeder_name) LIKE '%LATGHAT%'")
        # Joining the feeder table needs the feeder key. Live 27 Oct: 'consumers to move to feeder BARDEEHA PTW'
        # joined on data_date only and listed every reassigned meter.
        if (re.search(r"\bmapping_audit\b(?!_)", low) and "mapping_audit_feeder" in low
                # Only a JOIN of the feeder table itself -- a mapping_audit join next to a feeder_code IN
                # (SELECT ... FROM mapping_audit_feeder) subquery (the proof pack) is fine.
                and re.search(r"\bjoin\s+(?:hes_gold\.)?mapping_audit_feeder\b", low)
                and not re.search(r"\bon\b[^;]*feeder_code\s*=|=\s*\w+\.feeder_code\b", low)):
            problems.append("join mapping_audit_feeder on its feeder_code (= declared_feeder or auto_feeder) AND "
                            "data_date -- a feeder named in the question: auto_feeder IN (SELECT feeder_code FROM "
                            "hes_gold.mapping_audit_feeder WHERE ... UPPER(feeder_name) LIKE '%NAME%')")
        ranking = re.search(r"\b(?:which|top|most|highest|lowest|largest|least|best|worst)\b", question, re.I)
        if ranking and re.search(r"\border\s+by\b", low):
            # Rankings skip the blank place / feeder row (the 51,102 consumers with no hierarchy). Live 27 Oct:
            # 'which substation has the most consumers' answered with that blank row.
            if re.search(r"\bgroup\s+by\b", low) and not re.search(r"\bgroup\s+by\b[^;]*\bfeeder_code\b", low):
                # Only the finest grouped level is ranked (a substation row carries its zone).
                col = next((c for c in ("sub_station", "sub_division", "division", "circle", "zone")
                            if re.search(rf"\bgroup\s+by\b[^;]*\b{c}\b", low)), None)
                if col and not re.search(rf"\b{col}\s+is\s+not\s+null", low):
                    problems.append(f"a {col.replace('_', '')} ranking must skip the blank row of consumers with no "
                                    f"hierarchy: add {col} IS NOT NULL")
            # Only a ranking read FROM the feeder table; a LEFT JOIN that just adds feeder names (candidate
            # feeders of one meter) has no no-feeder row to skip.
            if (re.search(r"\bfrom\s+(?:hes_gold\.)?mapping_audit_feeder\b", low)
                    and "feeder_code is not null" not in low and "like" not in low
                    and re.search(r"\bfeeder", question, re.I)):
                problems.append("a feeder ranking must skip the no-feeder row (51,103 consumers): add feeder_code IS NOT NULL")
            # 'most / highest X' is a count unless a share is asked. Live 27 Oct: 'substation with the most pending
            # verification' was ranked by share (a 2-consumer substation at 100%).
            if (not re.search(r"\b(share|percent\w*|%|pct|rate|ratio|accuracy|confidence)\b", question, re.I)
                    and re.search(r"\b(?:most|highest|largest)\b", question, re.I)
                    and re.search(r"\border\s+by\s+\w*(?:_pct|share)\w*", low)):
                problems.append("'most / highest' asks for a COUNT: order by the count (e.g. pending consumers), "
                                "not by a share / percentage")
        # Totals over the feeder table include the no-feeder row, like the dashboard. Live 27 Oct: 'total MI across
        # all feeders' dropped it (4,111,811 instead of 4,162,914).
        if ("mapping_audit_feeder" in low and "feeder_code is not null" in low and re.search(r"\bsum\s*\(", low)
                and not re.search(r"\bgroup\s+by\b|\border\s+by\b|\blike\b|\bfeeder_code\s*(?:=|in)\b", low)):
            problems.append("feeder-table totals include every row (the no-feeder row too), like the dashboard: drop "
                            "feeder_code IS NOT NULL; feeders audited = COUNT(*) FILTER (WHERE mi > 0 OR "
                            "correct_tagging > 0)")
        # The evidence tables hold a row per consumer x candidate feeder (x 30-minute slot): ~15M / ~90-500M rows a
        # month. Without a meter or feeder the query scans the month and times out.
        for t in ("mapping_audit_candidates", "mapping_audit_evidence"):
            if f"hes_gold.{t}" in low and not re.search(
                    r"\bmeter_no\s*(?:=|in\b)|\b(?:declared|candidate)_feeder\s*(?:=|in\b)", low):
                problems.append(f"hes_gold.{t} is huge (a row per consumer x candidate feeder): filter meter_no = 'X' or "
                                "a feeder (declared_feeder / candidate_feeder = its feeder_code) as well as data_date")
        # The outage log has one row per 30-minute slot: outages are its event_ids. Live 28 Oct: 'outage windows
        # for meter AL1157377' counted slots per feeder and answered '5 outages, all dark' (17 outages, 12 dark).
        if ("hes_gold.mapping_audit_evidence" in low and re.search(r"\bgroup\s+by\b", low)
                and "event_id" not in low):
            problems.append("mapping_audit_evidence rows are 30-minute slots: count outages as COUNT(DISTINCT "
                            "event_id) and the ones the meter went dark in as COUNT(DISTINCT CASE WHEN consumer_dark "
                            "THEN event_id END) -- never COUNT(*)")
        # A per-feeder evidence summary must say which feeder is the meter's current one. Live 28 Oct: without it
        # the answer called the recommended feeder SAHAWAR PTW 'the declared feeder'.
        if (re.search(r"hes_gold\.mapping_audit_(?:evidence|candidates)\b", low)
                and re.search(r"\bgroup\s+by\b[^;]*\bcandidate_feeder\b|\bselect\b[^;]*\bcandidate_feeder\b", low)
                and "declared_feeder" not in low and "is_declared" not in low):
            problems.append("show which candidate is the meter's current (index-based) feeder: is_declared on "
                            "mapping_audit_candidates, or CASE WHEN candidate_feeder = MAX(declared_feeder) THEN "
                            "'Yes' ELSE 'No' END AS is_current_feeder on mapping_audit_evidence")
        # The summary and the meter table share only the month: joining them multiplies every row. Live 28 Oct:
        # 'correction queue in Triveni' summed consumer_count over the join -> 508,125,282,840.
        # (The month pin '(SELECT MAX(data_date) FROM hes_gold.mapping_audit_summary)' is not a use of the table.)
        tables_part = re.sub(r"select\s+max\s*\(\s*data_date\s*\)(?:\s+as\s+\w+)?\s+from\s+(?:hes_gold\.)?mapping_audit_summary\b",
                             "", low)
        if ("mapping_audit_summary" in tables_part
                and re.search(r"\b(?:from|join)\s+(?:hes_gold\.)?mapping_audit\b(?!_)", tables_part)):
            problems.append("never JOIN mapping_audit_summary with mapping_audit (they share only the month: every "
                            "row multiplies). Counts come from one table: SUM(consumer_count) on the summary, or "
                            "COUNT(*) on mapping_audit (verdict = 'REASSIGN' AND LOWER(cluster) = 'triveni')")
        # Reassign in and out are the same consumers counted at each end.
        if re.search(r"\b(out|inc)\s*\)?\s*\+\s*(sum\s*\()?\s*(inc|out)\b", low):
            problems.append("reassign out and reassign in count the SAME reassigned consumers at either end -- "
                            "never add them; total reassignments = SUM(out) (= SUM(inc))")
    return problems


_CR_SCALE_RE = re.compile(r"/\s*(?:10{5,}(?:\.0)?|1e[5-9])\b|\*\s*1e-?[5-9]\b")
_MONEY_RE = re.compile(r"\b(?:amount|crore|cr|rs|rupees?|money|invoiced amount|realised amount|realized amount)\b", re.I)


def _revenue_journey_problems(question: str, low: str, sql_texts: list[str], asked: str | None = None) -> list[str]:
    """Meter journey / current stage / receivables / revenue ageing mistakes seen live on 27 Oct 2026, each
    against the WFM dashboard's own SQL."""
    asked = asked or question
    words = f"{question} {asked}"
    problems: list[str] = []
    journey = "gold_meter_journey_stage_agg" in low
    # The journey card counts replacements; only the home tile drops them. Live 27 Oct: 'journey avg time category
    # wise' copied the tile's NOT is_replacement (Consumer 181.22 instead of the card's 181.38).
    if (journey and re.search(r"\bis_replacement\b", low)
            and not re.search(r"\breplace|\bheadline|\bnew install|\bhome\b", words, re.I)):
        problems.append("the meter journey card counts replacements: remove the is_replacement filter (only "
                        "'excluding replacements' or the home-page headline drops them)")
    # Consumer vs non-consumer is the home tile only. Live 27 Oct: 'journey avg time for meters installed in 2025'
    # came back as two columns, Consumer and Non-consumer.
    if (journey and re.search(r"upper\(category\)\s*<>\s*'consumer'|non[_ -]?consumer", low)
            and not re.search(r"\bnon[- ]?consumer|\bheadline|\bhome\b", words, re.I)):
        problems.append("journey answers are Total (or per category Consumer / DT / Feeder via GROUP BY ROLLUP "
                        "(category)) -- the Consumer vs Non-consumer split is the home tile only")
    # The year is the installation year. Live 27 Oct: 'total journey by category for 2026' filtered supply_month.
    if journey and re.search(r"\b20\d\d\b", question) and "installation_year" not in low:
        problems.append("the journey card's only year filter is installation_year (integer): WHERE installation_year = "
                        "<year> -- not supply_month")
    # Each stage keeps its own denominator. Live 27 Oct: 'slowest SAT to revenue circle in agra' divided by
    # meter_count (Agra 50.87 instead of the card's 64.35).
    if journey:
        for col, den in (("installation_to_sat_days_sum", "sat_meter_count"),
                         ("sat_to_revenue_realized_days_sum", "revenue_realized_meter_count"),
                         ("total_journey_days_sum", "revenue_realized_meter_count")):
            if re.search(rf"{col}\s*\)\s*(?:::\w+\s*)?/\s*nullif\s*\(\s*sum\s*\(\s*meter_count\s*\)", low):
                problems.append(f"{col} is averaged over {den}: ROUND(SUM({col}) / NULLIF(SUM({den}), 0), 2) -- "
                                "never over meter_count")
    # Several meters asked: every one is a row. Live 27 Oct: 'journey of meters AL1151970 and AL3188371' used
    # LIMIT 1 in each lookup and showed one meter (then described the second from the first's dates).
    serials = set(_SERIAL_RE.findall(question))
    if len(serials) > 1 and re.search(r"\blimit\s+1\b", low):
        problems.append(f"the question names {len(serials)} meters: return one row per meter (ROW_NUMBER() OVER "
                        "(PARTITION BY serial ORDER BY updated_at DESC) = 1), never LIMIT 1")
    # One meter lives in the silver tables; the gold snapshots hold counts only. Live 27 Oct: 'where is meter
    # SC10196043 stuck' filtered meter_serial_number on the stage snapshots and was rejected.
    if serials and re.search(r"gold_meter_(?:installation|revenue)_pending_stage_snapshot|gold_meter_journey_stage_agg|"
                             r"gold_revenue_ageing_snapshot", low):
        problems.append("one meter is looked up in the silver tables (unified_smart_meter_inventory, "
                        "unified_installations, sat_meter_mappings + sat_milestones) -- the gold stage / journey "
                        "tables hold counts only, with no meter serial")
    # Current stage: 'not installed' = the first four stages. Live 27 Oct: 'how many meters are not yet installed'
    # counted blank serials in unified_installations (0 instead of 109,837).
    if (re.search(r"\bnot (?:yet )?installed\b|\buninstalled\b", question, re.I)
            and "unified_installations" in low and "gold_meter_installation_pending_stage_snapshot" not in low):
        problems.append("meters not yet installed = SUM(meter_count) FROM public.gold_meter_installation_pending_stage_"
                        "snapshot (stages inventory, agency, supervisor, technician) -- the current-stage card")
    # Stage wise = all six stages. Live 27 Oct: both tables were read but filtered to the last two stages.
    if (re.search(r"\bstages?[- ]?wise\b|\bby stage\b|\beach stage\b", question, re.I)
            and "gold_meter_installation_pending_stage_snapshot" in low
            and re.search(r"current_stage\s+in\s*\(\s*'installation_done_sat_pending'\s*,\s*'sat_done_revenue_pending'\s*\)",
                          low)
            and not re.search(r"\binstalled\b|\bsat done\b|\bsat pending\b", question, re.I)):
        problems.append("stage wise = all six stages (inventory, agency, supervisor, technician, "
                        "installation_done_sat_pending, sat_done_revenue_pending) -- remove the current_stage filter")
    # Receivables are stored in crore already. Live 27 Oct: 'total invoiced amount in crore' divided by 1e8 (0 cr).
    if "revenue_receivables" in low and _CR_SCALE_RE.search(low):
        problems.append("revenue_receivables amounts are already in crore: never divide (or multiply) them by "
                        "10^5 / 10^7 -- SUM(lumpsum_invoiced) is the crore figure")
    # Outstanding = invoiced - realised (within timeline is unpaid too). Live 27 Oct: 'PMPM revenue outstanding'
    # subtracted within_timeline as well.
    if ("revenue_receivables" in low and re.search(r"\boutstanding\b|\bunpaid\b|\bpending amount\b", question, re.I)
            and not re.search(r"\boverdue\b", question, re.I)
            and re.search(r"_invoiced\s*-\s*\w*_realised\s*-\s*\w*_within_timeline", low)):
        problems.append("outstanding = invoiced - realised (Within Timeline + Overdue, all unpaid); "
                        "invoiced - realised - within_timeline is only the Overdue part")
    # Money questions are receivables (crore), never meter-count tables. Live 27 Oct: 'which project has the highest
    # outstanding amount' read the revenue ageing meter counts and said 'Rs 82,496 crore'.
    if (_MONEY_RE.search(question) and "revenue_receivables" not in low
            and re.search(r"gold_revenue_ageing_snapshot|gold_meter_revenue_pending_stage_snapshot", low)):
        problems.append("an amount (Rs / crore) is the receivables topic: public.revenue_receivables, latest row per "
                        "project, outstanding = invoiced - realised per stream -- the ageing table counts meters")
    # 'Outstanding' / 'revenue' meters are the revenue ageing snapshot, never the non-SAT ageing one. Live 27 Oct:
    # 'meters outstanding older than 90 days in triveni' read gold_non_sat_ageing_snapshot (91,805 vs 82,496).
    if (re.search(r"\boutstanding\b|\brevenue\b", question, re.I) and not re.search(r"\bnon[- ]?sat\b|\bsat pending\b",
                                                                                     question, re.I)
            and "gold_non_sat_ageing_snapshot" in low):
        problems.append("revenue outstanding meters = public.gold_revenue_ageing_snapshot (SUM(outstanding_count), age "
                        "from sat_date) -- gold_non_sat_ageing_snapshot is installed meters waiting for SAT")
    # GROUP BY LOWER(x) while the label shows x itself: Postgres rejects it, and grouping x and LOWER(x) together
    # (ROLLUP / GROUPING SETS) prints blank-label rows. Live 27 Oct: 'which project has the most meters pending with
    # agency' showed '' projects; 'revenue not realized ageing zone wise for kashi' failed twice.
    for col in set(re.findall(r"grouping\s*\(\s*lower\s*\(\s*(\w+)\s*\)\s*\)", low)):
        if re.search(rf"(?:upper|initcap)\s*\(\s*{col}\s*\)", low) or re.search(
                rf"(?:rollup|grouping\s+sets)\s*\([^;]*?\(\s*{col}\s*\)", low):
            problems.append(f"group by the column itself: GROUP BY ROLLUP ({col}) with CASE WHEN GROUPING({col}) = 1 "
                            f"THEN 'Total' ELSE UPPER({col}) END -- not LOWER({col}) beside {col}")
    # Every query carries the place the question names. Live 27 Oct: 'closures in triveni in September 2026' gave
    # the extra work-completed query for every project (17,475 called Triveni's).
    named = _named(words, _PROJECTS)
    if named and len(named) < len(_PROJECTS) and not _COMPARE_RE.search(question.lower()):
        for s in [t for t in sql_texts if t][1:]:
            ls = s.lower()
            if "gold_" in ls and not any(a in ls for p in named for a in _PROJECTS[p] + [p]):
                problems.append(f"every query needs the same project filter as the question "
                                f"({', '.join(sorted(named))}) -- an extra query has none")
                break
    return problems


def check(question: str, sql_texts: list[str], columns: set[str] | None = None, asked: str | None = None) -> list[str]:
    """Problems found (empty = consistent). `columns` = the column names of the SQL's
    tables (None = unknown: every check runs). A check those tables cannot satisfy is
    skipped -- a year on a snapshot table with no date column ('as of 20 Sep 2026'),
    a meter category on a table with no category column -- since asking the query
    writer to fix it would only cost a repair round."""
    sql = "\n".join(s for s in sql_texts if s)
    low = sql.lower()
    cols = {c.lower() for c in columns} if columns is not None else None
    problems: list[str] = []
    comparing = bool(_COMPARE_RE.search(question.lower()))
    # The mapping audit's summary / feeder tables have no project column: a project is its zone list
    # (UPPER(zone) IN ('AGRA', ...)), and every meter in them is a consumer meter.
    mapping = any(m in low for m in _MAPPING_TABLES)
    # The Technical-card tables have their own place / category columns (zones for a project, role / category
    # instead of WFM's CONSUMER / DT / FEEDER and meter_type): the WFM category and meter-type checks don't apply
    # (live 6 Nov: 'top 10 consumers by excess load' was sent back for a CONSUMER filter load_violation lacks).
    technical = any(re.search(rf"\bhes_gold\.{t}\b", low) for t in _TECHNICAL_TABLES) \
        or bool(re.search(r"\bhes_silver\.(?:block_usage|daily_usage)\b|\bhes_gold\.dim_meter(?:_replaced)?\b", low))
    mapping = mapping or technical

    for label, table, column in (("project", _PROJECTS, "project"), ("OEM", _OEMS, "oem")):
        # The user's own words count too: live 27 Oct the rewrite turned 'seipl' into 'SEPL' and the OEM filter
        # was dropped (17,869 Kashi installs instead of 273 SEIPL).
        named = _named(question, table) | (_named(asked, table) if asked else set())
        # Every value named ('all projects (Agra, Kashi, Triveni)') = no filter needed. The SQL
        # breaking down by that dimension, for several named values, a comparison ('Agra vs
        # other projects', 'why does HPL have the highest ...') or a share ('what share of all
        # installed meters is from Triveni') = each named one is a row.
        if len(named) == len(table) or ((len(named) > 1 or comparing or _SHARE_RE.search(question))
                                         and _grouped_by(low, column)):
            named_ok = named
        else:
            named_ok = set()
        for key in named - named_ok:
            if not any(a in low for a in table[key]) and key not in low:
                problems.append(f"the question names {label} '{key}' but the SQL never uses it")
        # Leak: an equality / IN filter on a value the question never named (a named discom names its projects).
        if column == "project":
            named = named | {p for d, ps in _DISCOM_PROJECTS.items() if f" {d} " in _words(question) for p in ps}
        leaked = _filtered(sql, list(table)) - named
        # A filter on every value of the dimension (IN ('agra','kashi','triveni')) is a breakdown, not a leak.
        if leaked and len(leaked) < len(table) and not named and not mapping:
            problems.append(f"the SQL filters on {label} {sorted(leaked)} that the question does not mention -- "
                            f"remove that filter (it was probably copied from an example)")

    has_category = cols is None or any("category" in c for c in cols)
    words = _words(f"{question} {asked or ''}")
    named_cats = set()
    asked_cats = {cat for cat, aliases in _CATEGORIES.items()
                  if any(f" {a} " in _words(question) for a in aliases) or (cat == "CONSUMER" and _CONSUMER_RE.search(question))}
    # Several categories named and the SQL breaks down by category ('Split SAT 10 by DT, Consumer and
    # Feeder'): each one is a row, like the project / OEM breakdowns above. Bare 'DT' / 'feeder' count
    # here -- grouping by category returns every category whatever the word meant.
    mentioned = asked_cats | {cat for cat in _CATEGORIES if re.search(rf"\b{cat.lower()}r?s?\b", _words(question))}
    split = len(mentioned) > 1 and _grouped_by(low, r"\w*category")
    for cat, aliases in _CATEGORIES.items() if has_category and not mapping else ():
        named = cat in asked_cats
        if named and cat.lower() not in low and not split:
            problems.append(f"the question asks about {cat} meters but the SQL never filters/selects category '{cat}'")
        if named or any(f" {a} " in words for a in aliases) or (cat == "CONSUMER" and _CONSUMER_RE.search(words)) \
                or re.search(rf"\b{cat.lower()}s?\b", words):
            named_cats.add(cat)
    # Leak: a category filter nobody asked for on the defect-rate union (defects + installed base). Live 27 Oct:
    # 'defect rate for seipl meters in varanasi ii zone' kept only DT installs (0 installed, a 0.00% rate).
    # Split columns (FILTER (WHERE category = 'DT')) and CASE labels are not filters.
    where_only = re.sub(r"\bcase\b.*?\bend\b", " ", re.sub(r"filter\s*\(\s*where[^)]*\)", " ", low, flags=re.S), flags=re.S)
    if ("gold_meter_defect_event_day_agg" in low and "gold_mi_installation_" in low and not comparing
            and not meter_types(f"{question} {asked or ''}") and not re.search(r"\bcategor|\btype", words)):
        for cat in ("DT", "FEEDER", "CONSUMER"):
            if cat not in named_cats and re.search(rf"category\s*(?:=\s*'{cat.lower()}'|in\s*\([^)]*'{cat.lower()}')",
                                                   where_only):
                problems.append(f"the SQL filters category '{cat}' that the question does not ask for -- remove it "
                                "(every category counts)")

    # 'category wise' = every category. Live 27 Oct: 'productivity category wise in august 2026' ran as
    # category = 'DT' only (one row).
    if (re.search(r"\bcategor(?:y|ies)[- ]?wise\b|\bby categor|\beach categor", question, re.I)
            and re.search(r"\bcategory\s*=\s*'(?:dt|feeder|consumer)'", where_only)):
        problems.append("a category-wise answer has one row per category (Consumer / DT / Feeder): GROUP BY category, "
                        "with no filter to a single category")

    # A place name that is not stored is a 0-row answer. Live 27 Oct: 'Chunar division' ran as division = 'chunar'
    # (stored 'edd chunar') and the answer said 0 active days.
    problems += _unknown_places(low)
    # The KPI 3 cube has no zone / circle columns: a place scope is filter_level + filter_label. Live 27 Oct:
    # 'Month-wise productivity in Agra zone' ran as project = 'agra' AND filter_level = 'zone' (every Agra zone
    # summed: 8.14 instead of 4.73).
    if (_TREND_CUBE in low and re.search(r"filter_level\s*=\s*'(?:zone|circle|division|sub_division)'", low)
            and "filter_label" not in low):
        problems.append("on gold_mi_productivity_trend_month_agg a place is filter_level = 'zone' AND "
                        "LOWER(filter_label) = 'agra' (the place's own name) -- filter_level alone sums every place")
    # filter_label is stored upper-case. Live 27 Oct: filter_label = 'agra' returned 0 rows.
    if _TREND_CUBE in low and re.search(r"(?<![\w(])filter_label\s*(?:=\s*'[^']*[a-z]|in\s*\([^)]*'[^']*[a-z])", sql):
        problems.append("filter_label is stored in UPPER case: filter_label = 'AGRA' (or LOWER(filter_label) = 'agra')")

    # The defect-rate union: every place filter of the defect leg also on the installed leg itself (not only inside
    # its OEM subquery). Live 27 Oct: SEIPL in Varanasi II divided 1,430 faulty by all 754,142 SEIPL installs (0.19%
    # instead of the card's 1.54%).
    problems += _union_place_gaps(low)

    # 'type wise' = Consumer / DTR / Feeder (category); 'consumer type wise' = the 4 consumer
    # types (1PH / 3PH / LTCT / HTCT) of CONSUMER meters only. Seen live: 'consumer stock ageing
    # type wise' came back as every meter type, DTR and Feeder included.
    if _TYPE_WISE_RE.search(question) and has_category and not technical:   # HES: category / role, no meter_type
        if _CONSUMER_RE.search(question) or re.search(r"\bmeter[- ]?types?\b", question, re.I):
            if "meter_type" not in low:
                problems.append("'consumer type wise' means one row per consumer meter type (1PH / 3PH / LTCT / HTCT) "
                                "-- GROUP BY meter_type with category = 'CONSUMER'")
        elif not re.search(r"\bcategor", low):
            problems.append("'type wise' means one row per category (Consumer / DTR / Feeder) -- GROUP BY category, "
                            "not meter_type")

    # 'without SAT' / 'pending SAT' counts: the total AND its split. The same question once
    # came back as 705 (all installed-not-SAT) and once as 364 (eligible only, the ageing card).
    # (O&M ticket questions filter tickets by their meter's SAT status -- 'open tickets of SAT pending meters'.)
    # Eligible pending = the Non-SAT Ageing card's total (gold_non_sat_ageing_snapshot): live 27 Oct, Kashi 75,066
    # there vs 75,070 from the MI-vs-SAT snapshot's flags.
    if (_WITHOUT_SAT_RE.search(question) and not _AGEING_RE.search(question)
            and not re.search(r"\b(tickets?|complaints?|escalat\w*|o&m|non[- ]?comm)\b", question, re.I)
            and not _SERIAL_RE.search(question)
            and not ("sat_stage is null" in low and (
                "gold_non_sat_ageing_snapshot" in low
                # a ranking / breakdown / what-if splits the snapshot's own flags per row
                or ("is_sat_eligible" in low and re.search(r"\b(worst|best|which|most|if|wise|by)\b", question, re.I))))):
        problems.append("a count of meters without / pending SAT needs the total AND its split: installed_not_sat = "
                        "SUM(meter_count) FILTER (WHERE sat_stage IS NULL) from public.gold_mi_sat_invoice_snapshot, "
                        "eligible_pending_sat = the Non-SAT card's (SELECT SUM(meter_count) FROM "
                        "public.gold_non_sat_ageing_snapshot ...same place filter), not_yet_due = the difference")

    # % SAT vs eligible MI = ALL SAT / eligible MI. Live 27 Oct: 'eligible sat % in kashi' came back as
    # SAT(eligible) / eligible = 94.18% instead of the card's 95.33%.
    if (re.search(r"\beligible\b", question, re.I) and re.search(r"%|percent|pct|\brate\b|\bratio\b", question, re.I)
            and re.search(r"filter\s*\(\s*where\s+(?:sat_stage\s+is\s+not\s+null\s+and\s+is_sat_eligible|"
                          r"is_sat_eligible\s+and\s+sat_stage\s+is\s+not\s+null)\s*\)(?:\s*,\s*0\s*\))?\s*\*\s*100", low)):
        problems.append("% SAT vs eligible MI = ALL SAT meters (FILTER (WHERE sat_stage IS NOT NULL), eligible or not) "
                        "* 100 / eligible MI -- the dashboard formula; never SAT(eligible) / eligible MI")

    # Un-utilised stock: its own snapshot, and its stage words. Live 27 Oct: 'consumer stock ageing type wise' read
    # the non-SAT ageing table; '90+ days stock still in the warehouse in triveni' dropped the stage (25,705 not 11,597).
    if (re.search(r"\b(stock|unused|un-?utili[sz]ed)\b", question, re.I) and "gold_non_sat_ageing_snapshot" in low
            and not re.search(r"\bsat\b", question, re.I)):
        problems.append("stock ageing reads public.gold_inventory_unutilized_stock_snapshot (SUM(stock_count), age = "
                        "CURRENT_DATE - inventory_date) -- gold_non_sat_ageing_snapshot is installed meters waiting for SAT")
    user_words = asked or question
    stage = _STOCK_STAGE_RE.search(question) or _STOCK_STAGE_RE.search(user_words)
    if "gold_inventory_unutilized_stock_snapshot" in low and "current_stage" not in low and stage:
        problems.append(f"'{stage.group(0)}' is a stock stage: filter current_stage = '{_stock_stage(stage.group(0))}' "
                        "(stages: warehouse, agency, supervisor, technician)")
    # Live 27 Oct: 'stock ageing warehouse wise in kashi' kept only current_stage = 'warehouse' (5,260 90+ instead
    # of the card's every-stage figures) -- 'warehouse wise' is the holding warehouse, not the stage. The user's
    # own words decide: the rewrite once turned 'warehouse wise' into 'in each warehouse'.
    if ("gold_inventory_unutilized_stock_snapshot" in low and not _STOCK_STAGE_RE.search(user_words)
            and not re.search(r"\bstages?\b|\btechnicians?\b|\bsupervisors?\b|\bagenc", user_words, re.I)
            and re.search(r"\bcurrent_stage\s*(?:=|in\s*\()", low)):
        problems.append("the question names no stock stage: count every stage (no current_stage filter) -- "
                        "'warehouse wise' means GROUP BY warehouse, the warehouse holding the stock")

    # inventory_date is each meter's receipt date, not a balance date. Live 27 Oct: 'remaining stock project wise' kept
    # inventory_date = MAX(inventory_date) (12,930 instead of 110,310) and 'remaining stock as of june 2026' kept
    # inventory_date <= '2026-06-30' (today's unused meters received by June, not June's stock).
    if (re.search(r"\b(?:remaining|left|balance)\b", question, re.I)
            and re.search(r"\binventory_date\s*(?:=|<=?|between)", low) and "gold_inventory_unutilized_stock_snapshot" in low
            and not _AGEING_RE.search(user_words)):
        problems.append("remaining stock = public.gold_meter_inventory_event_day_agg: SUM(inventory_count) - "
                        "SUM(stock_consumed_count) WHERE period_start <= the as-of date -- the stock snapshot's "
                        "inventory_date is each meter's receipt date, not a balance date")

    # Installs are MI progress (gold_mi_installation_*), never the inventory table's installed_count. Live 27 Oct:
    # 'how many dtr installed in chunar division in june 2026' read the inventory table (12 instead of 74).
    if ("gold_meter_inventory_event_" in low and "gold_mi_installation_" not in low
            and re.search(r"\binstall", question, re.I)
            and not _SUPPLY_CARD_RE.search(question)
            and not re.search(r"\b(stock|suppl\w*|receiv\w*|inventory|utili[sz]\w*|pace|remaining|warehouse|sku)\b",
                              f"{question} {asked or ''}", re.I)):
        problems.append("meters installed = MI progress: public.gold_mi_installation_day_agg / _month_agg (SUM(meter_count), "
                        "has_meter_serial AND progress_category IS NOT NULL AND NOT is_replacement) -- not the inventory "
                        "table's installed_count")

    # One meter: the query must look that meter up. Live 27 Oct: 'is meter CA8081038 eligible for sat' read only the
    # cutoff table, and 'is meter HP9000628 invoiced' answered 'not invoiced' without reading an invoice date.
    for serial in set(_SERIAL_RE.findall(question)):
        if serial.lower() not in low:
            problems.append(f"the question is about meter {serial} but no query looks it up -- filter its serial "
                            f"(new_meter_serial_number / meter_serial_number = '{serial}') in the main query")
        elif re.search(r"\binvoic", question, re.I) and "invoice_date" not in low:
            problems.append(f"whether meter {serial} is invoiced = its SAT batch's lumpsum_invoice_date / "
                            "pmpm_invoice_date (sat_meter_mappings LEFT JOIN sat_milestones on project AND sat_number)")

    # 'stage wise' on the current-stage card = all six stages; the revenue-pending table alone
    # is only the last two (seen live: 'revenue not realized stage wise' showed 2 stages).
    # 'revenue not realized meters' (KPI 16) counts not-installed meters too -- seen live: project
    # wise came back from the installed-only table (Agra 333,049 instead of 385,966).
    if ((re.search(r"\bstages?[- ]?wise\b|\beach stage\b|\bby stage\b|\bcurrent stage\b|\bwhich stage\b", question, re.I)
         or (re.search(r"\brevenue (not|un)[- ]?reali[sz]ed\b|\brevenue pending meters\b", question, re.I)
             and not re.search(r"\b(sat done|sat pending|installed|ageing|aging|age|days)\b", question, re.I)))
            and "gold_meter_revenue_pending_stage_snapshot" in low
            and "gold_meter_installation_pending_stage_snapshot" not in low):
        problems.append("a stage-wise answer needs all six stages: UNION ALL gold_meter_installation_pending_stage_snapshot "
                        "(inventory, agency, supervisor, technician) with gold_meter_revenue_pending_stage_snapshot "
                        "(installation_done_sat_pending, sat_done_revenue_pending)")
    # Live 27 Oct: 'revenue not realized meters project wise' (no age words) read the revenue ageing table
    # (157,979) instead of the current-stage card's six stages (1,132,826).
    elif (re.search(r"\brevenue (not|un)[- ]?reali[sz]ed\b", question, re.I)
            and not re.search(r"\b(sat done|sat pending|installed|ageing|aging|age|aged|days|old|older|outstanding|band)\b",
                              question, re.I)
            and "gold_revenue_ageing_snapshot" in low
            and "gold_meter_installation_pending_stage_snapshot" not in low):
        problems.append("'revenue not realized meters' (no age asked) = the current-stage card: all six stages from "
                        "gold_meter_installation_pending_stage_snapshot UNION ALL gold_meter_revenue_pending_stage_snapshot "
                        "-- the revenue ageing table is only for ageing / days questions")

    # '<level> wise' / 'by <level>' = one row per value of that column. Seen live: 'each step of the
    # meter journey in agra zone wise' ran as one row for zone = 'agra' (the breakdown was dropped).
    for lvl in ("zone", "circle", "division", "sub_division", "discom", "project", "warehouse", "oem"):
        # 'sub division wise' is the sub_division breakdown, not a division-wise one.
        word = lvl.replace("_", "[ _-]?") if lvl != "division" else r"(?<!sub )(?<!sub-)(?<!sub_)division"
        if (re.search(rf"\b{word}s?[- ]?wise\b|\bby {word}s?\b|\b(each|every|per) {word}\b|\b{word}s? breakdown\b",
                      question, re.I)
                and (cols is None or lvl in cols)
                and not (mapping and lvl == "project" and re.search(r"\bgroup\s+by\b[^;]*\b(?:proj|cluster)", low))
                and not _grouped_by(low, lvl) and len(re.findall(rf"\b{lvl}\s*=", low)) < 2
                # rows of a pre-aggregated table listed per value (no aggregate needed)
                and not (re.search(rf"^\s*select\b[^;]*?\b{lvl}\b[^;]*?\bfrom\b", low)
                         and not re.search(r"\b(sum|count|avg|min|max)\s*\(", low))):
            problems.append(f"the question asks for a {lvl.replace('_', '-')}-wise breakdown but the SQL never groups by "
                            f"{lvl} -- one row per {lvl} (GROUP BY {lvl}, usually with ROLLUP for the Total)")

    # Journey 'levels' / 'stages' / 'steps' = all six stages + Total; seen live: 'journey avg time
    # across the levels' came back as Total Journey only.
    if ("gold_meter_journey_stage_agg" in low
            and re.search(r"\b(levels?|stages?|steps?|each stage|stage[- ]?wise|per stage|breakdown)\b", question, re.I)
            and not all(c in low for c in ("inventory_to_agency_days_sum", "agency_to_supervisor_days_sum",
                                           "supervisor_to_technician_days_sum", "technician_to_installation_days_sum",
                                           "installation_to_sat_days_sum", "sat_to_revenue_realized_days_sum"))):
        problems.append("the journey 'levels' / stages are all six: DI to Agency, Agency to Supervisor, Supervisor to "
                        "Technician, Technician to Installation (/ meter_count), Installation to SAT (/ sat_meter_count), "
                        "SAT to Revenue (/ revenue_realized_meter_count) plus Total Journey -- one row (or column) each")

    # Journey stages BY a dimension = one row per zone / category ..., stages as columns. Seen
    # live: 'each step ... agra zone wise' came back as 42 zone x stage rows (LATERAL unpivot)
    # and the written answer covered only the first zone.
    if ("gold_meter_journey_stage_agg" in low and re.search(r"\blateral\b|\(\s*values\b", low)
            and re.search(r"\b\w+[- ]?wise\b|\bby (zone|circle|division|category|project|discom|meter type)", question, re.I)
            and not re.search(r"\b(stage|level|step)s?[- ]?wise\b", question, re.I)):
        problems.append("journey stages BY a dimension: return ONE ROW per that dimension (ROLLUP Total) with "
                        "di_to_agency, agency_to_supervisor, supervisor_to_technician, technician_to_installation, "
                        "installation_to_sat, sat_to_revenue, total_journey as COLUMNS -- no LATERAL/VALUES unpivot")

    # Invoice % (lumpsum / PMPM) is of Total SAT, never of all installed meters -- seen live: the
    # same 'PMPM invoice percentage' came back as 87.82% (of SAT) and 79.56% (of all meters).
    if (re.search(r"has_(pmpm|lumpsum)_invoice", low) and re.search(r"percent|pct|%|\bshare\b|\brate\b|\bratio\b", question, re.I)
            and re.search(r"\*\s*100|100(\.0)?\s*\*", low) and "sat_stage is not null" not in low):
        problems.append("invoice % = invoiced meters / Total SAT meters x 100 -- the denominator is SUM(meter_count) "
                        "FILTER (WHERE sat_stage IS NOT NULL), never all installed meters")

    # Meter type: '1ph consumer meters' once ran as meter_type = 'LTCT-Consumer_meter' (0 rows).
    has_type = (cols is None or any("meter_type" in c for c in cols)) and not technical
    named_types = meter_types(question) if has_type else []
    for mt in named_types:
        if not any(s in low for s in _METER_TYPES[mt][1]):
            problems.append(f"the question asks about {mt.split('-')[0]} meters but the SQL never uses that type "
                            f"-- filter meter_type = '{mt}'")
    if named_types and not (len(named_types) > 1 or comparing):
        for mt, (_, spellings) in _METER_TYPES.items():
            if mt not in named_types and re.search(rf"=\s*'[^']*({'|'.join(map(re.escape, spellings))})", low):
                problems.append(f"the SQL filters meter type {mt} that the question does not ask for -- "
                                f"use meter_type = '{named_types[0]}'")

    # active_days is 1 per technician-day row: SUM(active_days) counts rows (Chunar, Sep 2026:
    # 270 'active days' in a 30-day month). Active days = distinct calendar days with an install.
    # The KPI 3 cube is the exception: its active_days is already a distinct day count per month row.
    if re.search(r"\bsum\s*\(\s*(?:\w+\.)?active_days\s*\)", low) and _TREND_CUBE not in low:
        problems.append("SUM(active_days) counts technician rows, not days -- active days = COUNT(DISTINCT period_start) "
                        "on the *_day_agg table (calendar days with an install); technician-days = "
                        "SUM(technician_day_count)")

    # KPI 2 vs KPI 3: the cards count technician-days differently (Jan 2026: 5.56 vs 5.80), so the table follows the
    # question's words. Live 26 Oct: 'productivity circle wise in Kashi' was answered from the KPI 3 cube (3.23 vs 2.91).
    if _TREND_CUBE in low and not _KPI3_RE.search(question):
        problems.append("the question does not name the Monthly Productivity trend card, so MI productivity reads "
                        "gold_mi_technician_productivity_day_agg / _month_agg (the MI-Productivity per team card) -- "
                        "not gold_mi_productivity_trend_month_agg")
    elif _KPI3_CARD_RE.search(question) and "gold_mi_technician_productivity" in low and _TREND_CUBE not in low:
        problems.append("the question names the Monthly Productivity trend card (KPI 3): read "
                        "gold_mi_productivity_trend_month_agg (its own technician-day count), not the technician table")

    # Stock has no zone / circle / division (a meter gets its place when installed): a stock
    # balance filtered to one zone is always 0 ('stock left in Prayagraj II' -> '0 meters').
    if any(_stock_filtered_by_place(t) for t in sql_texts if t):
        problems.append("remaining stock has no circle / division (stock is held per warehouse, counted in its zone), so "
                        "a circle/division filter always gives 0 -- filter the ZONE (or project) for the stock "
                        "(extra query), and keep the circle/division filter only for supplied / installed")

    # The supply utilization / MI pace vs stock cards show supplies, installed, rate and stock together. Live 26 Oct:
    # 'supply utilization rate for OEM AEW' came back as 97.84% alone.
    if (_SUPPLY_CARD_RE.search(question) and "gold_meter_inventory_event_" in low
            and not any(re.search(ask, question.lower()) for _, ask, _ in _TIME_BREAKDOWNS)
            and not re.search(r"\btrend\b", question, re.I)
            and not (re.search(r"\bsum\s*\(\s*inventory_count\s*\)\s+as\b", low)
                     and re.search(r"\bsum\s*\(\s*installed_count\s*\)\s+as\b", low)
                     and "stock_consumed_count" in low)):
        problems.append("a supply utilization / MI pace vs stock answer gives the card's four figures as columns: "
                        "SUM(inventory_count) AS total_supplies, SUM(installed_count) AS total_installed, the "
                        "utilization %, and SUM(inventory_count) - SUM(stock_consumed_count) AS remaining_stock")

    # 'Agra HW' is the warehouse Agra_HW, not the Agra project. Live 26 Oct: 'stock ageing for Agra HW' was answered for
    # project = 'agra' (49,307 instead of 37,139).
    if _WAREHOUSE_RE.search(question) and (cols is None or "warehouse" in cols) and "warehouse" not in low:
        problems.append("the question names a warehouse (a name ending HW / ZW / SWH / Warehouse) -- filter "
                        "LOWER(warehouse) = 'agra_hw' (etc.), not the project")

    # revenue_receivables has no discom column: a discom is its projects. Live 26 Oct: 'within timeline for PUVVNL'
    # summed all three projects (Agra is DVVNL).
    if "revenue_receivables" in low:
        for discom, projects in _DISCOM_PROJECTS.items():
            if f" {discom} " in _words(question) and not all(re.search(rf"'{p}'", low) for p in projects):
                problems.append(f"{discom.upper()} = project(s) {', '.join(projects)} -- revenue_receivables has no "
                                f"discom column, so filter LOWER(project) IN ({', '.join(repr(p) for p in projects)})")

    problems += _om_problems(question, low, asked)
    problems += _revenue_journey_problems(question, low, sql_texts, asked)
    problems += _mapping_problems(question, sql_texts)
    problems += _technical_problems(question, sql_texts)
    problems += _card_problems_nov10(question, sql_texts, asked)
    problems += _meter_reading_problems(question, sql_texts)
    # A ROLLUP / GROUPING SETS total row has NULL in the grouped column: an 'Unmapped' label that does not test
    # GROUPING() first turns the total into an area and it is added again. Live 10 Nov: 'kal kitne meter lage the,
    # category ke hisaab se?' said 1,208 (Unmapped 604 + CONSUMER 604) for 604; Agra escalations 33,222 for 16,611.
    for s in sql_texts:
        sl = (s or "").lower()
        if re.search(r"\brollup\b|\bgrouping\s+sets\b|\bcube\b", sl):
            for col in set(re.findall(r"case\s+when\s+(?:\w+\.)?(\w+)\s+is\s+null(?:(?!\bend\b).)*?'unmapped'", sl, re.S)):
                if not re.search(rf"case\s+when\s+grouping\s*\(\s*(?:\w+\.)?{col}\s*\)\s*=\s*1\s+then\s+'total'", sl):
                    problems.append(f"the ROLLUP total row has {col} = NULL: label it first -- CASE WHEN GROUPING({col}) = 1 "
                                    f"THEN 'Total' WHEN {col} IS NULL OR {col} = '' THEN 'Unmapped' ELSE ... END, or the "
                                    "total is shown as 'Unmapped' and counted twice")
    # dt_loading holds one row per DT x loading band: counting rows (COUNT_IF / COUNT(*)) counts bands, not DTs.
    # Live 10 Nov: 'DTs that peaked above rating per project' said Kashi 648 for 85.
    for s in sql_texts:
        sl = (s or "").lower()
        if re.search(r"\bhes_gold\.dt_loading\b(?!_)", sl) and re.search(r"count_if\s*\(\s*peak_block_band|count\s*\(\s*\*\s*\)", sl) \
                and "count(distinct meter_no)" not in re.sub(r"\s+", "", sl).replace("count(distinctmeter_no)", "count(distinct meter_no)"):
            problems.append("dt_loading has one row per DT x loading band: count DTs with COUNT(DISTINCT meter_no) "
                            "(FILTER (WHERE peak_block_band IN ('100-120%', '>120%'))), never COUNT_IF / COUNT(*) of rows")
    # Every figure comes from the data. Live 6 Nov: 'Predict the average VDI for September 2026' ran
    # SELECT DATE '2026-09-01', 0 AS eligible_meters, NULL AS avg_vdi_pct ... -- no table at all.
    if sql_texts and not any(re.search(r"\b(?:from|join)\s+(?!\(|unnest\b)\"?[a-z_]", (s or ""), re.I) for s in sql_texts):
        problems.append("the SQL reads no table -- every figure must come from the data (for a forecast, query the "
                        "month-wise series and project from the complete months); never SELECT made-up values")

    # A meter-type exclusion nobody asked for. Live 26 Oct: 'invoice funnel zone wise in Triveni' added
    # meter_type <> 'OTHER-Consumer_meter' AND meter_type IS NOT NULL and lost 44 meters.
    # (On the productivity cubes meter_type IS NOT NULL picks the roll-up level, not an exclusion.)
    if (_DROPPED_TYPE_RE.search(low) and "productivity" not in low
            and not re.search(r"\b(other|unknown|unmapped|blank|null|excluding|exclude|without|except)\b", question, re.I)):
        problems.append("the SQL drops meter types the question never excluded (OTHER-Consumer_meter / NULL meter_type) -- "
                        "remove that filter; every meter counts")

    # 'Revenue not realized ageing' is the SAT-done-PMPM-uncollected snapshot. Live 26 Oct: 'revenue not realized
    # ageing project wise' read the current-stage table (1,019,165 instead of 157,979).
    if _REVENUE_AGEING_RE.search(question) and "gold_revenue_ageing_snapshot" not in low:
        problems.append("revenue not realized ageing reads public.gold_revenue_ageing_snapshot (SUM(outstanding_count), "
                        "age from sat_date) -- not the current-stage tables")

    # Time breakdown asked but dropped: 'productivity weekly in august' once came back as one
    # monthly figure. An average ('daily average', 'avg monthly') is not a breakdown.
    ql = question.lower()
    for unit, ask, bucket in _TIME_BREAKDOWNS:
        if (re.search(ask, ql) and not re.search(rf"\b(?:avg|average|mean)\s+{unit}|\b{unit}\s+{_NOT_BREAKDOWN}", ql)
                and not _time_grouped(low, bucket)
                and not (technical and re.search(r"\bmeter_no\s*=", low))):   # one meter's rows on a per-day HES table
            problems.append(f"the question asks for a {unit} breakdown but the SQL does not GROUP BY {_PER[unit]} -- "
                            f"return one row per {_PER[unit]} (plus the Total row)")

    for stage in re.findall(r"\bsat[\s-]?(\d{1,2})\b", question.lower()):
        if comparing and _grouped_by(low, "sat_stage"):
            continue  # 'why is SAT-13 lower than other batches': every stage is a row
        if not re.search(rf"(sat_stage\s*(=|in)\s*\(?\s*[^)]*\b{stage}\b|'sat-{stage}'|sat_stage\s*>=?\s*\d)", low):
            problems.append(f"the question is about SAT-{stage} but the SQL never selects sat_stage {stage}")

    has_dates = cols is None or any(_DATE_COLUMN_RE.search(c) for c in cols)
    for year in set(re.findall(r"\b(20\d\d)\b", question)) if has_dates else ():
        if year not in sql:
            problems.append(f"the question names the year {year} but the SQL has no {year} date")

    top = re.search(r"\b(?:top|best|worst|first|bottom|lowest|highest|oldest|newest|latest)\s+"
                    r"(\d{1,3}|three|five|ten|fifteen|twenty)\b",
                    question.lower())
    if top:
        n = _NUM_WORDS.get(top.group(1)) or int(top.group(1))
        if not re.search(rf"(limit\s+{n}\b|<=\s*{n}\b|<\s*{n + 1}\b)", low):
            problems.append(f"the question asks for {n} items but the SQL has no LIMIT {n}")

    # 'Last month' is the previous calendar month. Live 27 Oct: 'which technician closed the most complaints in agra
    # last month' ran for 1-31 October (the current month) while the answer said September.
    if re.search(r"\blast month\b|\bprevious month\b|\bpichhle mahine\b", question, re.I):
        import datetime as _dt
        this_month = _dt.date.today().replace(day=1).isoformat()
        if f"'{this_month}'" in low and "interval '1 month'" not in low:
            problems.append(f"'last month' is the previous calendar month -- {this_month} is the current month's first "
                            "day; use period_start >= (date_trunc('month', CURRENT_DATE) - INTERVAL '1 month')::date "
                            "AND period_start < date_trunc('month', CURRENT_DATE)::date")

    # Leak: a specific calendar date in the SQL when the question names no period at all
    # (no year / month / date / 'yesterday' / 'last N days' ...) -- 'No date = all time'.
    from gmr_agents.resolvers.query_encoder import encode

    asked = encode(question)
    # Technical-card tables: no period = the dashboard's default month (pinned by _technical_problems), never
    # all time. Live 6 Nov: this check called the default month's DATE '2026-06-01' bound a leak and the
    # repairs ended in SQL Athena could not run.
    technical = any(re.search(rf"\bhes_gold\.{t}\b", low) for t in _TECHNICAL_TABLES) \
        or bool(re.search(r"\bhes_silver\.(?:block_usage|daily_usage)\b|\bhes_gold\.dim_meter(?:_replaced)?\b", low))
    if not technical and not any(asked.get(k) for k in ("years", "months", "dates", "relative_period", "as_of")):
        dated = re.findall(r"(?:period_start|date|_at|_date|day|_month)\w*\s*(?:=|>=|<=|>|<|between)\s*(?:date\s*)?'(\d{4}-\d{2}-\d{2})'", low)
        if dated:
            problems.append(f"the SQL filters on the date {dated[0]} but the question names no period -- "
                            "no date means all time; remove that date filter")
    return problems
