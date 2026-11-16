"""One consumer's outage match against EVERY feeder of its substation (auto-indexing / mapping audit), built by code.

'Mapping of meter AL1157377', 'which feeder is consumer 8902396000 on', 'compare the feeders of this consumer's
substation': the consumer is looked up in the mapping audit (hes_gold.mapping_audit, the dashboard's Consumer-to-
Feeder Mapping Audit card), then every feeder of its substation is compared on the outages the audit observed:

    went dark with the feeder   supporting      (evidence FOR the feeder)
    stayed powered              contradicting   (evidence AGAINST it)
    match %                     supporting / (supporting + contradicting)
    rank                        supporting DESC, net_score DESC -- the dashboard's own candidate order

Sources: mapping_audit (verdict, declared / auto / best feeder), mapping_audit_feeder (the substation's feeders),
mapping_audit_candidates (per consumer x feeder scores; none for NO DATA consumers), mapping_audit_evidence (the
outage-by-outage log, REASSIGN consumers only). No LLM call; every figure comes from the queries below, and each
table's SQL is kept (sql / extra_sql) so table paging and downloads re-run it.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import pandas as pd

log = logging.getLogger("gmr_agents.feeder_compare")

# Mapping-audit meter serials: AL1157377, SC10025001, CA4000001 (two letters + 7-8 digits).
_SERIAL_RE = re.compile(r"\b([A-Za-z]{2}\d{7,8})\b")
_CONSUMER_RE = re.compile(r"\bconsumer(?:\s*(?:no\.?|number|id|account(?:\s*no\.?)?))?\s*[#:]?\s*(\d{7,12})\b", re.I)
_THIS_RE = re.compile(r"\b(?:this|that|same|the above|above|its|his|her|their)\s+(?:consumer|meter|customer|connection)\b"
                      r"|\b(?:its|it's|his|her|their)\b|\bit\b", re.I)

# The ask is about this consumer's feeder tagging / its substation's feeders.
_MAPPING_RE = re.compile(
    r"\bmapp|\btagg?(?:ed|ing)?\b|\bre-?tag|\bindex|\breassign|\bverdict\b|\bcandidate|\boutage match"
    r"|\bcompar|\bvs\b|\bversus\b|\bmatch(?:es|ing|ed)?\b", re.I)
_FEEDER_RE = re.compile(r"\bfeeders?\b|\bsub[- ]?stations?\b|\bss\b", re.I)
# Another domain's question about the same meter: readings, billing, voltage ... -- left to the pipeline unless
# the mapping itself is asked.
_OTHER_RE = re.compile(
    r"\bconsum(?:ption|ed)\b|\busage\b|\bkwh\b|\bkvah\b|\bbill|\bvoltage\b|\bpower factor\b|\bpf\b|\bload\b|\bread(?:ing)?s?\b"
    r"|\btamper|\btheft\b|\binstall|\bsat\b|\bticket|\bcomplain|\breplace|\brecharge|\bbalance\b|\bdemand\b|\bcurrent\b"
    r"|\bdtr?\b|\btransformer|\bevent", re.I)
_STRONG_RE = re.compile(r"\bmapp|\btagg|\bre-?tag|\bindex|\breassign|\bverdict\b|\bcandidate|\boutage match", re.I)
# The comparison itself must be asked: 'compare', 'vs', 'all / other / every feeder(s)', 'feeder-wise',
# 'feeders of / in the substation'. Everything else about one consumer is answered by the pipeline.
_COMPARE_ASK_RE = re.compile(
    r"\bcompar\w*|\bvs\.?(?=\s)|\bversus\b|\bside[- ]by[- ]side\b|\bfeeder[- ]?wise\b"
    r"|\b(?:all|other|every|each|rest of the|remaining|nearby|neighbou?ring)\b[^?.]{0,40}\bfeeders?\b"
    r"|\bfeeders\b[^?.]{0,40}\b(?:in|of|at|under|from)\b[^?.]{0,25}\bsub[- ]?station"
    r"|\bacross\b[^?.]{0,25}\bfeeders\b|\bagainst\b[^?.]{0,30}\bfeeders\b", re.I)
# A follow-up must name the feeders / substation -- 'compare it with July' is some other comparison.
_FOLLOW_RE = re.compile(r"\bfeeders?\b|\bsub[- ]?stations?\b", re.I)

_MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_MONTH_RE = re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?(?:\s*[-,']?\s*(20\d\d|\d\d))?\b", re.I)
_VERDICT_TEXT = {
    "CONFIRMED": "the declared feeder matches the outage evidence -- the tagging is correct",
    "REASSIGN": "the outage evidence points to another feeder -- the consumer should be re-tagged",
    "REVIEW": "outage evidence was seen but is not decisive -- needs a field check",
    "NO DATA": "the consumer was never in a feeder-specific outage window, so no feeder could be tested",
}


def _serial(text: str) -> str | None:
    for m in _SERIAL_RE.finditer(text or ""):
        s = m.group(1).upper()
        if not re.fullmatch(r"(?:20\d\d|\d+)", s):
            return s
    return None


def request(question: str, prior_turns: list[dict] | None = None, today: date | None = None) -> dict | None:
    """{meter, consumer_id, month} when the question asks for one consumer's feeder mapping / substation feeder
    comparison; None otherwise. No query here."""
    q = question or ""
    # Only when a COMPARISON is asked (user, 16 Nov): 'which feeder is X on' / 'mapping of X' / 'why was X
    # reassigned' get just what they ask, from the normal pipeline.
    if not _COMPARE_ASK_RE.search(q):
        return None
    meter = _serial(q)
    cons = None if meter else (_CONSUMER_RE.search(q).group(1) if _CONSUMER_RE.search(q) else None)
    mapping, feeder, other = _MAPPING_RE.search(q), _FEEDER_RE.search(q), _OTHER_RE.search(q)
    if meter or cons:
        if not (mapping or feeder):
            return None
        if other and not _STRONG_RE.search(q):
            return None
        if not (_STRONG_RE.search(q) or feeder):
            return None   # 'compare meter X with Y' without feeders / mapping is some other comparison
    else:
        # 'Compare all feeders of this consumer's substation' after a turn about one consumer.
        if not (_THIS_RE.search(q) and (_FOLLOW_RE.search(q) or _STRONG_RE.search(q))) or (other and not _STRONG_RE.search(q)):
            return None
        for t in reversed(prior_turns or []):
            ctx = t.get("context") or {}
            meter = ctx.get("meter_no") or _serial(t.get("resolved_question") or "") or _serial(t.get("question") or "")
            if meter:
                break
            m = _CONSUMER_RE.search(t.get("question") or "")
            if m:
                cons = m.group(1)
                break
        if not (meter or cons):
            return None
    month = None
    for m in _MONTH_RE.finditer(q):
        word = m.group(1).lower()[:3]
        if word == "may" and not m.group(2):
            continue   # 'may' is usually the verb
        y = m.group(2)
        year = (2000 + int(y) if len(y) == 2 else int(y)) if y else (today or date.today()).year
        month = date(year, _MONTHS[word], 1)
        break
    return {"meter": meter, "consumer_id": cons, "month": month}


def is_request(question: str, prior_turns: list[dict] | None = None) -> bool:
    return request(question, prior_turns) is not None


# --------------------------------------------------------------------------- #
# Queries
# --------------------------------------------------------------------------- #

def _lit(s: str) -> str:
    return "'" + str(s).replace("'", "''") + "'"


def _run(sql: str) -> pd.DataFrame:
    from gmr_agents.catalog.catalog import get_catalog
    from gmr_agents.execution import executor
    from gmr_agents.guards import sql_guard

    guard = sql_guard.check_sql_safety(sql, get_catalog("mdm"))
    if not guard["ok"]:
        raise RuntimeError(f"feeder comparison query failed the SQL guard: {guard.get('reason')}")
    return executor.run_sql(guard["normalized_sql"], None, "athena")


def _meter_sql(meter: str, month: date | None) -> str:
    when = (f"DATE '{month.isoformat()}'" if month else "(SELECT MAX(data_date) FROM hes_gold.mapping_audit_summary)")
    return ("SELECT data_date, meter_no, category, cluster, zone, sub_station, verdict, declared_feeder, auto_feeder, "
            "best_feeder, declared_dark, declared_powered, best_dark, best_powered, was_cross_substation "
            f"FROM hes_gold.mapping_audit WHERE data_date = {when} AND meter_no = {_lit(meter)}")


def _compare_sql(m: dict) -> str:
    d, meter = f"DATE '{m['month']}'", _lit(m["meter_no"])
    declared, auto, best = (_lit(m[k]) if m.get(k) else "NULL" for k in ("declared_feeder", "auto_feeder", "best_feeder"))
    return (
        "WITH c AS (SELECT candidate_feeder, supporting, contradicting, net_score FROM hes_gold.mapping_audit_candidates "
        f"WHERE data_date = {d} AND meter_no = {meter}), "
        "f AS (SELECT feeder_code, feeder_name, sub_station, mi FROM hes_gold.mapping_audit_feeder "
        f"WHERE data_date = {d} AND feeder_code IS NOT NULL AND (sub_station = {_lit(m['sub_station'])} "
        f"OR feeder_code IN (SELECT candidate_feeder FROM c) OR feeder_code = {declared})) "
        "SELECT CASE WHEN c.candidate_feeder IS NOT NULL THEN ROW_NUMBER() OVER (PARTITION BY c.candidate_feeder IS NULL "
        "ORDER BY c.supporting DESC, c.net_score DESC, f.feeder_code) END AS evidence_rank, "
        "f.feeder_code, f.feeder_name, "
        f"CASE WHEN f.feeder_code = {declared} AND f.feeder_code = {auto} THEN 'Declared = recommended' "
        f"WHEN f.feeder_code = {declared} THEN 'Declared (current tag)' "
        f"WHEN f.feeder_code = {auto} THEN 'Recommended (auto-indexed)' "
        f"WHEN f.feeder_code = {best} THEN 'Strongest evidence' ELSE '' END AS role, "
        "COALESCE(c.supporting, 0) + COALESCE(c.contradicting, 0) AS feeder_outages_seen, "
        "COALESCE(c.supporting, 0) AS consumer_went_dark, COALESCE(c.contradicting, 0) AS consumer_stayed_powered, "
        "ROUND(c.supporting * 100.0 / NULLIF(c.supporting + c.contradicting, 0), 1) AS outage_match_pct, "
        "c.net_score, f.mi AS consumers_tagged_to_feeder, f.sub_station "
        "FROM f LEFT JOIN c ON c.candidate_feeder = f.feeder_code "
        "ORDER BY c.candidate_feeder IS NULL, c.supporting DESC, c.net_score DESC, f.feeder_code")


def _timeline_sql(m: dict) -> str:
    d = f"DATE '{m['month']}'"
    return (
        "SELECT date_format(MIN(e.slot) AT TIME ZONE 'Asia/Kolkata', '%d %b %Y %H:%i') AS outage_start_ist, "
        "date_format((MAX(e.slot) + INTERVAL '30' MINUTE) AT TIME ZONE 'Asia/Kolkata', '%d %b %Y %H:%i') AS outage_end_ist, "
        "e.candidate_feeder AS feeder_code, MAX(f.feeder_name) AS feeder_name, e.event_id AS feeder_outage_no, "
        "COUNT(*) AS half_hours, CASE WHEN BOOL_OR(e.consumer_dark) THEN 'Went dark' ELSE 'Stayed powered' END AS consumer, "
        "CASE WHEN e.candidate_feeder = " + (_lit(m["declared_feeder"]) if m.get("declared_feeder") else "NULL")
        + " THEN 'Declared' WHEN e.candidate_feeder = " + (_lit(m["auto_feeder"]) if m.get("auto_feeder") else "NULL")
        + " THEN 'Recommended' ELSE '' END AS role "
        "FROM hes_gold.mapping_audit_evidence e LEFT JOIN hes_gold.mapping_audit_feeder f "
        f"ON f.feeder_code = e.candidate_feeder AND f.data_date = {d} "
        f"WHERE e.data_date = {d} AND e.meter_no = {_lit(m['meter_no'])} "
        "GROUP BY e.candidate_feeder, e.event_id ORDER BY MIN(e.slot), e.candidate_feeder")


def _dark_slots_sql(m: dict) -> str:
    """The half-hours the consumer itself had no supply (consumer_dark is per consumer x slot, the same on every
    candidate row) -- the timeline's 'this consumer' lane. An outage's dark flag (BOOL_OR) would paint a whole
    long outage dark for one dark half-hour (live 16 Nov: AL1157049, a 28-hour I O C outage)."""
    return ("SELECT DISTINCT date_format(slot AT TIME ZONE 'Asia/Kolkata', '%Y-%m-%d %H:%i') AS slot_ist "
            f"FROM hes_gold.mapping_audit_evidence WHERE data_date = DATE '{m['month']}' "
            f"AND meter_no = {_lit(m['meter_no'])} AND consumer_dark ORDER BY 1")


def _merge_slots(slots: pd.DataFrame | None) -> list[dict]:
    """Half-hour slot starts (IST) -> merged [start, end) windows."""
    if slots is None or not len(slots):
        return []
    times = sorted(pd.to_datetime(slots["slot_ist"], format="%Y-%m-%d %H:%M"))
    out: list[list] = []
    for t in times:
        if out and t <= out[-1][1]:
            out[-1][1] = max(out[-1][1], t + pd.Timedelta(minutes=30))
        else:
            out.append([t, t + pd.Timedelta(minutes=30)])
    return [{"start": a.isoformat(), "end": b.isoformat()} for a, b in out]


def _months_sql(meter: str) -> str:
    return (f"SELECT data_date, verdict FROM hes_gold.mapping_audit WHERE meter_no = {_lit(meter)} "
            "AND data_date >= DATE '2026-01-01' ORDER BY data_date")


def _consumer_sql(consumer_id: str) -> str:
    cid = consumer_id.lstrip("0") or "0"
    return ("SELECT meter_no FROM hes_gold.dim_meter WHERE regexp_replace(TRIM(consumer_id), '^0+', '') = "
            f"{_lit(cid)} ORDER BY meter_no")


# --------------------------------------------------------------------------- #
# Answer
# --------------------------------------------------------------------------- #

def _short(name, code) -> str:
    n = str(name or "").strip()
    n = n.rsplit("_", 1)[0] if "_" in n else n
    return f"{n} ({code})" if n else str(code)


def _frac(dark, powered) -> str:
    dark, powered = int(dark or 0), int(powered or 0)
    n = dark + powered
    return f"{dark} of {n} outage{'s' if n != 1 else ''} ({dark * 100 / n:.0f}%)" if n else "none of its outages (none observed)"


def _table(tid: str, heading: str, df: pd.DataFrame, labels: dict[str, str]) -> dict:
    cols = [labels.get(c, c.replace("_", " ").title()) for c in df.columns]
    rows = [{cols[i]: (None if (not isinstance(v, (list, dict)) and pd.isna(v)) else
                       (v.isoformat() if hasattr(v, "isoformat") else
                        (int(v) if isinstance(v, float) and v == int(v) else
                         (bool(v) if isinstance(v, bool) else v))))
             for i, v in enumerate(rr)} for rr in df.itertuples(index=False)]
    return {"id": tid, "title": heading, "heading": heading, "columns": cols, "rows": rows, "page": 1,
            "page_size": max(len(rows), 1), "total_rows": len(rows), "has_next": False, "has_previous": False}


def _not_found(req: dict, meter: str | None, months: pd.DataFrame | None) -> dict:
    who = f"Meter {meter}" if meter else f"Consumer {req['consumer_id']}"
    if months is not None and len(months):
        avail = ", ".join(f"{pd.Timestamp(r.data_date):%b %Y} ({r.verdict})" for r in months.itertuples())
        text = (f"{who} is not in the mapping audit for {req['month']:%B %Y}. It is audited for: {avail}. "
                "Ask again with one of those months.")
    elif meter:
        text = (f"{who} is not in the consumer-to-feeder mapping audit (it covers consumer meters only, Jun-Aug 2026). "
                "Check the meter number.")
    else:
        text = f"{who} was not found in the meter master, so its feeder mapping cannot be compared. Check the number."
    return {"summary": text, "insights": [], "kpis": [], "tables": [], "charts": [], "sql": None, "extra_sql": [],
            "meter": meter, "title": f"Feeder mapping - {meter or req['consumer_id']}", "found": False}


def _iso(text) -> str | None:
    """'01 Aug 2026 07:00' (IST, as the outage log shows it) -> '2026-08-01T07:00:00'."""
    try:
        return pd.to_datetime(str(text), format="%d %b %Y %H:%M").isoformat()
    except (TypeError, ValueError):
        return None


def _num(v, cast=float):
    return None if v is None or (not isinstance(v, (list, dict)) and pd.isna(v)) else cast(v)


def _panel(m: dict, meter: str, month_label: str, cmp_df: pd.DataFrame, in_ss: pd.DataFrame,
           timeline: pd.DataFrame | None, months_df: pd.DataFrame, nm, tied: bool, insights: list[str],
           notes: list[str], dark_slots: pd.DataFrame | None = None) -> dict:
    """The structured payload the frontend's FeederComparePanel draws (verdict header, current -> recommended
    feeder, the feeder leaderboard, the outage timeline, the month switcher). Same figures as the tables."""
    short = lambda code: nm(code).split(" (")[0] if code else None
    feeders = []
    for row in cmp_df.itertuples():
        feeders.append({
            "rank": _num(row.evidence_rank, int), "code": row.feeder_code, "name": short(row.feeder_code),
            "role": row.role or "", "seen": int(row.feeder_outages_seen or 0), "dark": int(row.consumer_went_dark or 0),
            "powered": int(row.consumer_stayed_powered or 0), "match_pct": _num(row.outage_match_pct),
            "net": _num(row.net_score, int), "mi": _num(row.consumers_tagged_to_feeder, int),
            "declared": row.feeder_code == m.get("declared_feeder"),
            "recommended": row.feeder_code == m.get("auto_feeder"),
            "best": row.feeder_code == m.get("best_feeder"),
            "other_substation": row.sub_station if row.sub_station != m["sub_station"] else None})
    outages = []
    if timeline is not None and len(timeline):
        for row in timeline.itertuples():
            outages.append({"feeder": row.feeder_code, "no": int(row.feeder_outage_no),
                            "start": _iso(row.outage_start_ist), "end": _iso(row.outage_end_ist),
                            "start_label": row.outage_start_ist, "end_label": row.outage_end_ist,
                            "half_hours": int(row.half_hours), "dark": row.consumer == "Went dark"})
    months = [{"month": pd.Timestamp(r.data_date).strftime("%Y-%m"), "label": pd.Timestamp(r.data_date).strftime("%b %Y"),
               "verdict": str(r.verdict)} for r in months_df.itertuples()] if len(months_df) else []

    def side(code, dark, powered):
        if not code:
            return None
        return {"code": code, "name": short(code), "dark": int(dark or 0), "seen": int(dark or 0) + int(powered or 0)}

    return {
        "meter": meter, "category": str(m.get("category") or "").replace("_", " "), "sub_station": m["sub_station"],
        "zone": (m.get("zone") or "").title() or None, "project": m.get("cluster"), "month": m["month"][:7],
        "month_label": month_label, "verdict": str(m["verdict"]), "verdict_text": _VERDICT_TEXT.get(str(m["verdict"]), ""),
        "declared": side(m.get("declared_feeder"), m.get("declared_dark"), m.get("declared_powered")),
        "recommended": side(m.get("auto_feeder"), m.get("best_dark"), m.get("best_powered"))
        if m.get("auto_feeder") and m.get("auto_feeder") == m.get("best_feeder") else
        (side(m.get("auto_feeder"), None, None) if m.get("auto_feeder") else None),
        "strongest": side(m.get("best_feeder"), m.get("best_dark"), m.get("best_powered"))
        if m.get("best_feeder") and m.get("best_feeder") not in (m.get("declared_feeder"), m.get("auto_feeder")) else None,
        "cross_substation": bool(m.get("was_cross_substation")), "tied": bool(tied),
        "feeders": feeders, "feeders_in_substation": int(len(in_ss)), "outages": outages, "months": months,
        "consumer_dark": _merge_slots(dark_slots),
        "action": insights[0] if insights else None, "notes": notes,
    }


def build(req: dict) -> dict:
    """The comparison as an answer: summary, insights, kpis, tables (feeder comparison, outage log), chart."""
    meter, lead = req.get("meter"), ""
    if not meter:
        got = _run(_consumer_sql(req["consumer_id"]))
        if got.empty:
            return _not_found(req, None, None)
        meter = str(got["meter_no"].iloc[0])
        lead = (f"Consumer {req['consumer_id']} = meter {meter}"
                + (f" (also {', '.join(got['meter_no'].iloc[1:].astype(str))})" if len(got) > 1 else "") + ". ")
    base = _run(_meter_sql(meter, req.get("month")))
    if base.empty:
        months = _run(_months_sql(meter)) if req.get("month") else None
        return _not_found(req, meter, months)
    r = base.iloc[0]
    m = {k: (None if (not isinstance(v, (list, dict)) and pd.isna(v)) else v) for k, v in r.items()}
    m["month"] = pd.Timestamp(m["data_date"]).date().isoformat()
    month_label = pd.Timestamp(m["data_date"]).strftime("%b %Y")
    verdict = str(m["verdict"])

    sqls = {"compare": _compare_sql(m)}
    if verdict == "REASSIGN":
        sqls["timeline"] = _timeline_sql(m)
        sqls["dark_slots"] = _dark_slots_sql(m)   # panel only -- not a table, never in sql / extra_sql
    with ThreadPoolExecutor(len(sqls) + 1) as pool:
        months_fut = pool.submit(_run, _months_sql(meter))   # the panel's month switcher
        f = dict(zip(sqls, pool.map(_run, sqls.values())))
        try:
            months_df = months_fut.result()
        except Exception:
            log.warning("feeder comparison: audit months for %s not read", meter, exc_info=True)
            months_df = pd.DataFrame(columns=["data_date", "verdict"])
    cmp_df = f["compare"]
    cmp_df["role"] = cmp_df["role"].fillna("")
    names = {row.feeder_code: _short(row.feeder_name, row.feeder_code) for row in cmp_df.itertuples()}
    nm = lambda code: names.get(code) or str(code)
    scored = cmp_df[cmp_df["evidence_rank"].notna()].copy()
    in_ss = cmp_df[cmp_df["sub_station"] == m["sub_station"]]
    unscored = in_ss[in_ss["evidence_rank"].isna()]

    lines = [f"{lead}**Meter {meter}** ({str(m.get('category') or 'consumer').replace('_', ' ')}) -- substation "
             f"**{m['sub_station']}**" + (f", {m['zone'].title()} zone" if m.get("zone") else "")
             + f", mapping audit {month_label}.",
             f"Verdict: **{verdict}** -- {_VERDICT_TEXT.get(verdict, '')}."]
    if m.get("declared_feeder"):
        lines.append(f"Declared (current) feeder: **{nm(m['declared_feeder'])}** -- the consumer went dark in "
                     f"{_frac(m['declared_dark'], m['declared_powered'])} of this feeder.")
    if verdict == "REASSIGN" and m.get("auto_feeder"):
        lines.append(f"Recommended (auto-indexed) feeder: **{nm(m['auto_feeder'])}** -- went dark in "
                     f"{_frac(m['best_dark'], m['best_powered'])}.")
    elif verdict == "REVIEW" and m.get("best_feeder") and m["best_feeder"] != m.get("declared_feeder"):
        lines.append(f"Strongest evidence (not decisive): {nm(m['best_feeder'])} -- went dark in "
                     f"{_frac(m['best_dark'], m['best_powered'])}.")
    if m.get("was_cross_substation"):
        lines.append("The declared feeder belongs to another substation (cross-substation tag).")

    if len(scored):
        # The full ranking is drawn by the feeder comparison panel (frontend FeederComparePanel); the text keeps
        # the headline only. The plain-text list stays in the table (and its download).
        lead_row = scored.iloc[0]
        lines.append(f"\nAll {len(in_ss)} feeders of {m['sub_station']} compared: best match **{nm(lead_row['feeder_code'])}**"
                     f", dark in {_frac(lead_row['consumer_went_dark'], lead_row['consumer_stayed_powered'])}"
                     + (f"; {len(unscored)} feeder(s) had no outage to check against." if len(unscored) else "."))
        if len(scored) > 1:
            a, b = scored.iloc[0], scored.iloc[1]
            if a["consumer_went_dark"] == b["consumer_went_dark"]:
                lines.append(f"Note: {nm(a['feeder_code'])} and {nm(b['feeder_code'])} both went dark with the consumer "
                             f"{int(a['consumer_went_dark'])} times; the net score ({int(a['net_score']):+d} vs "
                             f"{int(b['net_score']):+d}) " + ("separates them." if a["net_score"] != b["net_score"]
                                                              else "is also equal -- the match is ambiguous."))
        # The audit ranks by outage COUNT; a feeder lower down can match the consumer's outages far more often.
        solid = scored[scored["feeder_outages_seen"] >= 3]
        if len(solid):
            hi = solid.sort_values(["outage_match_pct", "feeder_outages_seen"], ascending=False).iloc[0]
            top = scored.iloc[0]
            if hi["feeder_code"] != top["feeder_code"] and float(hi["outage_match_pct"] or 0) >= float(top["outage_match_pct"] or 0) + 15:
                lines.append(f"Highest outage match %: {nm(hi['feeder_code'])} -- dark in "
                             f"{_frac(hi['consumer_went_dark'], hi['consumer_stayed_powered'])} vs "
                             f"{float(top['outage_match_pct'] or 0):.0f}% for rank 1; the audit ranks by the number of "
                             f"outages matched, so it is ranked {int(hi['evidence_rank'])}. Worth checking in the field.")
    else:
        lines.append(f"\nNo feeder of {m['sub_station']} had an outage the consumer could be checked against "
                     f"in {month_label}, so there is nothing to compare ({len(in_ss)} feeders listed below).")

    if "timeline" in f and len(f["timeline"]):
        tl = f["timeline"]
        tl["role"] = tl["role"].fillna("")
        lines.append(f"\nOutage log: {len(tl)} feeder outages at this substation in the audit window, "
                     f"{int((tl['consumer'] == 'Went dark').sum())} with the consumer dark (see the outage timeline).")

    insights = []
    # A REASSIGN whose recommended feeder only TIES the declared one (same dark count and net score): the audit's
    # pick is a tie-break, not evidence -- live 16 Nov, AL1157049: TOWN-1 2 of 2 vs AWALKHERA PTW 2 of 2.
    tied = False
    if verdict == "REASSIGN" and m.get("auto_feeder") and m.get("declared_feeder"):
        sc = scored.set_index("feeder_code")
        if m["auto_feeder"] in sc.index and m["declared_feeder"] in sc.index:
            a, b = sc.loc[m["auto_feeder"]], sc.loc[m["declared_feeder"]]
            tied = a["consumer_went_dark"] == b["consumer_went_dark"] and a["net_score"] == b["net_score"]
    if tied:
        lines.append(f"Caution: the recommended feeder only ties the declared one on the evidence -- the re-tag rests "
                     f"on a tie-break, so verify in the field before changing the index.")
        insights.append(f"Field-verify meter {meter} before re-tagging: {nm(m['declared_feeder'])} and "
                        f"{nm(m['auto_feeder'])} match its outages equally.")
    elif verdict == "REASSIGN" and m.get("auto_feeder"):
        insights.append(f"Re-tag meter {meter} from {nm(m['declared_feeder'])} to {nm(m['auto_feeder'])} in the "
                        "consumer-feeder index.")
    elif verdict == "REVIEW":
        insights.append(f"Field-verify meter {meter}'s feeder: the outage evidence does not separate the candidates.")
    elif verdict == "NO DATA":
        insights.append("No action from the audit yet: re-check after a month with feeder-specific outages.")
    elif verdict == "CONFIRMED":
        insights.append(f"No change needed: {nm(m['declared_feeder'])} is confirmed by the outage evidence.")

    kpis = [{"label": "Verdict", "value": 0.0, "formatted_value": verdict, "unit": None},
            {"label": "Declared feeder match", "value": float(m.get("declared_dark") or 0),
             "formatted_value": _frac(m.get("declared_dark"), m.get("declared_powered")).replace(" outages", "").replace(" outage", ""), "unit": None}]
    if len(scored):
        top = scored.iloc[0]
        kpis.append({"label": "Best match", "value": float(top["outage_match_pct"] or 0),
                     "formatted_value": f"{nm(top['feeder_code']).split(' (')[0]}: "
                                        f"{_frac(top['consumer_went_dark'], top['consumer_stayed_powered']).replace(' outages', '')}",
                     "unit": None})
    kpis.append({"label": "Feeders compared", "value": float(len(scored)),
                 "formatted_value": f"{len(scored)} of {len(in_ss)}", "unit": None})

    tables = [_table("feeder_compare", f"Feeders of {m['sub_station']}: outage match for meter {meter} ({month_label})",
                     cmp_df, {"evidence_rank": "Rank", "feeder_code": "Feeder code", "feeder_name": "Feeder name",
                              "role": "Role", "feeder_outages_seen": "Feeder outages seen",
                              "consumer_went_dark": "Consumer went dark", "consumer_stayed_powered": "Consumer stayed powered",
                              "outage_match_pct": "Outage match %", "net_score": "Net score",
                              "consumers_tagged_to_feeder": "Consumers tagged (MI)", "sub_station": "Substation"})]
    order = ["compare"]
    if "timeline" in f and len(f["timeline"]):
        tables.append(_table("feeder_outage_log", f"Outage log: each feeder outage and whether meter {meter} went dark (IST)",
                             f["timeline"], {"outage_start_ist": "Outage start (IST)", "outage_end_ist": "Outage end (IST)",
                                             "feeder_code": "Feeder code", "feeder_name": "Feeder name",
                                             "feeder_outage_no": "Feeder outage no.", "half_hours": "Half-hours",
                                             "consumer": "Consumer", "role": "Role"}))
        order.append("timeline")
    for i, t in enumerate(tables):
        t["table_index"] = i

    charts = []
    if len(scored):
        charts.append({"id": "feeder_compare_chart", "type": "stacked_bar",
                       "title": f"Meter {meter}: outages of each {m['sub_station']} feeder -- consumer dark vs powered",
                       "x": [nm(c).split(" (")[0] for c in scored["feeder_code"]],
                       "y": [float(v) for v in scored["consumer_went_dark"]],
                       "series": [{"name": "Consumer went dark (supports)", "data": [float(v) for v in scored["consumer_went_dark"]]},
                                  {"name": "Consumer stayed powered (contradicts)",
                                   "data": [float(v) for v in scored["consumer_stayed_powered"]]}]})

    notes = [ln.strip() for ln in lines if ln.strip().startswith(("Note:", "Highest", "Caution", "The declared feeder belongs"))]
    panel = _panel(m, meter, month_label, cmp_df, in_ss, f.get("timeline"), months_df, nm, tied, insights, notes,
                   f.get("dark_slots"))
    return {"summary": "\n".join(lines), "insights": insights, "kpis": kpis, "tables": tables, "charts": charts,
            "panel": panel,
            "sql": sqls[order[0]], "extra_sql": [sqls[k] for k in order[1:]], "meter": meter, "found": True,
            "verdict": verdict, "sub_station": m["sub_station"], "month": m["month"],
            "title": f"Feeder mapping - {meter} - {month_label}"}
