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

_PROJECTS = {"agra": ["agra"], "kashi": ["kashi", "kasi", "varanasi"], "triveni": ["triveni", "trivei"]}
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
    from gmr_agents.sql import hierarchy

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

    for label, table, column in (("project", _PROJECTS, "project"), ("OEM", _OEMS, "oem")):
        # The user's own words count too: live 27 Oct the rewrite turned 'seipl' into 'SEPL' and the OEM filter
        # was dropped (17,869 Kashi installs instead of 273 SEIPL).
        named = _named(question, table) | (_named(asked, table) if asked else set())
        # Every value named ('all projects (Agra, Kashi, Triveni)') = no filter needed. The SQL
        # breaking down by that dimension, for several named values or a comparison ('Agra vs
        # other projects', 'why does HPL have the highest ...') = each named one is a row.
        if len(named) == len(table) or ((len(named) > 1 or comparing) and _grouped_by(low, column)):
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
        if leaked and len(leaked) < len(table) and not named:
            problems.append(f"the SQL filters on {label} {sorted(leaked)} that the question does not mention -- "
                            f"remove that filter (it was probably copied from an example)")

    has_category = cols is None or any("category" in c for c in cols)
    words = _words(f"{question} {asked or ''}")
    named_cats = set()
    for cat, aliases in _CATEGORIES.items() if has_category else ():
        named = any(f" {a} " in _words(question) for a in aliases) or (cat == "CONSUMER" and _CONSUMER_RE.search(question))
        if named and cat.lower() not in low:
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
    if _TYPE_WISE_RE.search(question) and has_category:
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
        word = lvl.replace("_", "[ _-]?")
        if (re.search(rf"\b{word}s?[- ]?wise\b|\bby {word}s?\b|\b(each|every|per) {word}\b|\b{word}s? breakdown\b",
                      question, re.I)
                and (cols is None or lvl in cols)
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
    has_type = cols is None or any("meter_type" in c for c in cols)
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
                and not _time_grouped(low, bucket)):
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
    from gmr_agents.validation.query_encoder import encode

    asked = encode(question)
    if not any(asked.get(k) for k in ("years", "months", "dates", "relative_period", "as_of")):
        dated = re.findall(r"(?:period_start|date|_at|_date|day|_month)\w*\s*(?:=|>=|<=|>|<|between)\s*(?:date\s*)?'(\d{4}-\d{2}-\d{2})'", low)
        if dated:
            problems.append(f"the SQL filters on the date {dated[0]} but the question names no period -- "
                            "no date means all time; remove that date filter")
    return problems
