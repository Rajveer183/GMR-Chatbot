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


# Zone is not here: stock is counted in its warehouse's zone (sql/hierarchy.py places it), so a
# zone filter on stock is right; circle / division still have no stock.
_WITHOUT_SAT_RE = re.compile(r"\b(without|pending|awaiting|no|not)\s+(a\s+)?sat\b|\bsat\s+(pending|not done|baaki)\b|"
                             r"\bnot\s+(yet\s+)?(cleared|passed)\s+sat\b", re.I)
_AGEING_RE = re.compile(r"\b(age|ageing|aging|aged|old|older|band|bands|days?|since|how long|stuck|why)\b", re.I)
_CONSUMER_RE = re.compile(r"\bconsumers?\b(?!\s*(number|no\b|id\b|name|account|code))", re.I)
_TYPE_WISE_RE = re.compile(r"\btype[- ]?wise\b|\bby (meter )?types?\b|\btypes? wise\b|\beach type\b", re.I)
_PLACE_LEVELS = {"circle", "division", "sub_division"}
_TREND_CUBE = "gold_mi_productivity_trend_month_agg"
# Words that name the Monthly Productivity trend card (KPI 3); the cube's own measure may also be asked by name.
# The card is monthly only: a 'daily / weekly productivity trend' is KPI 2's.
_KPI3_CARD_RE = re.compile(r"(?<!daily )(?<!weekly )\bproductivity trend\b|\bmonthly (?:mi )?productivity\b|\bkpi[- ]?3\b", re.I)
_KPI3_RE = re.compile(_KPI3_CARD_RE.pattern + r"|\b(?:average|avg\.?) active technicians\b", re.I)


def check(question: str, sql_texts: list[str], columns: set[str] | None = None) -> list[str]:
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
        named = _named(question, table)
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
        # Leak: an equality / IN filter on a value the question never named.
        leaked = _filtered(sql, list(table)) - named
        # A filter on every value of the dimension (IN ('agra','kashi','triveni')) is a breakdown, not a leak.
        if leaked and len(leaked) < len(table) and not named:
            problems.append(f"the SQL filters on {label} {sorted(leaked)} that the question does not mention -- "
                            f"remove that filter (it was probably copied from an example)")

    has_category = cols is None or any("category" in c for c in cols)
    for cat, aliases in _CATEGORIES.items() if has_category else ():
        named = any(f" {a} " in _words(question) for a in aliases) or (cat == "CONSUMER" and _CONSUMER_RE.search(question))
        if named and cat.lower() not in low:
            problems.append(f"the question asks about {cat} meters but the SQL never filters/selects category '{cat}'")

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
    if (_WITHOUT_SAT_RE.search(question) and not _AGEING_RE.search(question)
            and not re.search(r"\b[A-Za-z]{1,4}\d{6,10}\b", question)
            and not ("sat_stage is null" in low and "is_sat_eligible" in low)):
        problems.append("a count of meters without / pending SAT needs the total AND its split from "
                        "public.gold_mi_sat_invoice_snapshot: installed_not_sat = SUM(meter_count) FILTER (WHERE "
                        "sat_stage IS NULL), eligible_pending_sat (... AND is_sat_eligible), not_yet_due (... AND NOT "
                        "is_sat_eligible) -- not the ageing snapshot (eligible only)")

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

    top = re.search(r"\b(?:top|best|worst|first|bottom|lowest|highest)\s+(\d{1,3}|three|five|ten|fifteen|twenty)\b",
                    question.lower())
    if top:
        n = _NUM_WORDS.get(top.group(1)) or int(top.group(1))
        if not re.search(rf"(limit\s+{n}\b|<=\s*{n}\b|<\s*{n + 1}\b)", low):
            problems.append(f"the question asks for {n} items but the SQL has no LIMIT {n}")

    # Leak: a specific calendar date in the SQL when the question names no period at all
    # (no year / month / date / 'yesterday' / 'last N days' ...) -- 'No date = all time'.
    from gmr_agents.validation.query_encoder import encode

    asked = encode(question)
    if not any(asked.get(k) for k in ("years", "months", "dates", "relative_period", "as_of")):
        dated = re.findall(r"(?:period_start|date|_at|_date|day)\w*\s*(?:=|>=|<=|>|<|between)\s*(?:date\s*)?'(\d{4}-\d{2}-\d{2})'", low)
        if dated:
            problems.append(f"the SQL filters on the date {dated[0]} but the question names no period -- "
                            "no date means all time; remove that date filter")
    return problems
