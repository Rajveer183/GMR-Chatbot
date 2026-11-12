"""Investigation: 'why did X rise / fall' tested against the data, by code.

drivers.py shows where the change happened along ONE split. An investigation asks the
planner (same single SQL call -- no extra LLM hop) for the same measure and the same two
periods split along up to two OTHER dimensions (extra_queries, run in parallel), then tests
the standard explanations on every split:

    coverage      the change comes from groups that report in only one period (a data gap,
                  not a real move) -- checked first, it overturns every other reading
    localised     one group explains most of the change
    broad-based   most groups moved the same way, none dominates
    reconciliation  the splits must agree on the total; a gap = rows without that dimension
                  or a different filter (contradiction detection)

and grades the confidence (low base, partial period, one split only, coverage gap, splits
disagree, change within a few %). Every line is a figure computed from the rows, so the
answer writer cites evidence instead of inventing a cause; it is told never to name a
field-level cause (staff, weather, supply) the data does not show.

No LLM call, no database query: it reads the rows the plan already returned.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from gmr_agents.presentation import drivers

_ASK_RE = re.compile(
    r"\b(?:investigat\w*|diagnos\w*|root[- ]?caus\w*|what could (?:explain|cause|be behind)|possible (?:reasons|causes|explanations)|"
    r"test (?:each|the|these) (?:explanation|hypothes)\w*|hypothes[ie]s)\b", re.I)

PLAN_HINT = (
    "INVESTIGATION (code tests each explanation from the rows -- no extra model call): write the main query as for "
    "'why' (one row per (period, group), the period column first, the two periods compared). ALSO add up to 2 "
    "extra_queries with the SAME measure, SAME filters and SAME two periods, each split by a DIFFERENT dimension of "
    "the same table (e.g. zone / circle / feeder / meter type / category / OEM / agency / complaint type) -- columns: "
    "period, that dimension, the measure; purpose = 'by <dimension>'. No extra when the table has no other useful "
    "dimension or keeps no history.")


_DOWN_RE = re.compile(r"\b(?:drop\w*|fall\w*|fell|declin\w*|decreas\w*|dip\w*|down|lower|reduc\w*|slow\w*|kam|ghat\w*)\b", re.I)
_UP_RE = re.compile(r"\b(?:ris(?:e|es|en|ing)|rose|increas\w*|jump\w*|grow\w*|grew|spik\w*|up|higher|badh\w*|zyada)\b", re.I)


def _table_starts(tables: list[str] | None) -> pd.Timestamp | None:
    """Earliest period any of these tables holds (data skeleton: real MIN of its date / period columns)."""
    try:
        from gmr_agents.retrieval import data_skeleton
        cards = {c.get("table"): c for c in (data_skeleton.load() or {}).get("tables") or []}
    except Exception:
        return None
    starts = []
    for t in tables or []:
        for col, span in ((cards.get(str(t).split(".")[-1]) or {}).get("dates") or {}).items():
            if col in ("data_date", "month_date", "period_start") or col.endswith("_date"):
                try:
                    starts.append(pd.Timestamp(span[0]))
                except Exception:
                    pass
    return min(starts) if starts else None


def _single_period(question: str, df, tables: list[str] | None = None) -> str:
    """A why-question whose rows hold ONE period only: the change cannot be measured from them. Whether the
    earlier period truly has no data is checked against the table's real date range (data skeleton) -- live
    11 Nov: 'feeder loss June vs July' SQL derived June from the latest month, returned July only, and the answer
    said June data does not exist (the table starts in June)."""
    from gmr_agents.presentation.stats import _split
    if not isinstance(df, pd.DataFrame) or df.empty:
        return ""
    try:
        rows, label, dates, nums = _split(df)
    except Exception:
        return ""
    if dates is None or dates.dropna().nunique() != 1:
        return ""
    only = pd.Timestamp(dates.dropna().iloc[0])
    start = _table_starts(tables)
    if start is not None and start.normalize() >= only.normalize():
        why = (f"the source table starts at {only:%b %Y}, so there is no earlier period to compare with. Say so first "
               "and give the current figures.")
    else:
        why = (f"the source holds earlier periods{f' (from {start:%b %Y})' if start is not None else ''}, but this query "
               "returned none for the place / category asked. Say the figures returned cover "
               f"{only:%b %Y} only and the comparison could not be made from them; do NOT say the earlier data does not "
               "exist or was not reported.")
    return (f"INVESTIGATION -- ONE PERIOD ONLY: the rows cover {only:%b %Y} only, so the change cannot be measured and "
            f"there is no evidence of a rise or fall: {why} Do NOT name a place or cause as driving a change, and never "
            "write the words RESULT, FACTS or rows in the answer. CONFIDENCE: low -- earlier period missing")


def asked(question: str) -> bool:
    return drivers.asked(question) or bool(_ASK_RE.search(question or ""))


def _f(v) -> str:
    return drivers._fmt(v)


def _dimension(both: pd.DataFrame, measure: str, group: str, p0: str, p1: str, partial: str | None) -> dict:
    """Hypotheses tested on one split."""
    rate = bool(drivers._RATE_RE.search(str(measure)))
    n = int(len(both))
    d: dict = {"by": group, "measure": measure, "periods": [p0, p1], "rate": rate, "groups": n,
               "partial": partial, "hypotheses": [], "top": []}
    new = both[both["prev"].isna() & both["curr"].notna()]
    gone = both[both["prev"].notna() & both["curr"].isna()]
    if rate:
        b = both.dropna(subset=["prev", "curr"]).copy()
        b["chg"] = b["curr"] - b["prev"]
        if len(b) < 2:
            return d
        b = b.sort_values("chg", key=lambda s: -s.abs())
        d["top"] = [{"name": str(i), "prev": float(r.prev), "curr": float(r.curr), "chg": float(r.chg)}
                    for i, r in b.head(3).iterrows()]
        absum = float(b["chg"].abs().sum())
        top = d["top"][0]
        up, down = int((b["chg"] > 0).sum()), int((b["chg"] < 0).sum())
        d.update(change=float(b["chg"].mean()), up=up, down=down)
        loc = absum > 0 and abs(top["chg"]) >= 0.5 * absum and len(b) >= 3
        d["lead_score"] = (abs(top["chg"]) / absum * 100) if loc else None
        d["lead_item"] = top if loc else None
        d["hypotheses"].append({"h": f"localised in one {group}", "status": "SUPPORTED" if loc else "NOT SUPPORTED",
                                "evidence": f"{top['name']} {_f(top['prev'])} -> {_f(top['curr'])} ({top['chg']:+.2f} points)"
                                            f"{', the largest move' if not loc else ', most of all movement'}"})
        same = max(up, down)
        broad = len(b) >= 3 and same / len(b) >= 0.7 and not loc
        d["hypotheses"].append({"h": f"broad-based across {group} values", "status": "SUPPORTED" if broad else "NOT SUPPORTED",
                                "evidence": f"{up} rose, {down} fell"})
        if len(new) or len(gone):
            d["hypotheses"].append({"h": "data coverage", "status": "CHECK",
                                    "evidence": f"{len(new)} only in {p1}, {len(gone)} only in {p0} -- left out of the comparison"})
        return d
    t0, t1 = float(both["prev"].fillna(0).sum()), float(both["curr"].fillna(0).sum())
    total = t1 - t0
    d.update(total_prev=t0, total_curr=t1, change=total, pct=(total / t0 * 100) if t0 else None)
    # 1. coverage: groups present in one period only, or a whole period without figures
    cov = float(new["curr"].sum()) - float(gone["prev"].sum())
    if t0 == 0 and t1 > 0:
        d["hypotheses"].append({"h": "data coverage", "status": "SUPPORTED",
                                "evidence": f"{p0} has no figures at all ({_f(t1)} in {p1}): the 'change' is the start of "
                                            "reporting, not a real rise"})
        d["coverage"] = True
    elif len(new) or len(gone):
        share = cov / total if total else 0.0
        both_in = both.dropna(subset=["prev", "curr"])
        l0, l1 = float(both_in["prev"].sum()), float(both_in["curr"].sum())
        sup = total != 0 and share >= 0.5
        d["hypotheses"].append({
            "h": "data coverage", "status": "SUPPORTED" if sup else "PARTLY" if abs(share) >= 0.2 else "NOT SUPPORTED",
            "evidence": (f"{len(new)} {group} value(s) only in {p1}" + (f" ({', '.join(map(str, new.index[:3]))})" if len(new) else "")
                         + f", {len(gone)} only in {p0}; they account for {cov:+,.0f} of the {total:+,.0f} change; "
                         f"like-for-like ({len(both_in)} in both periods): {_f(l0)} -> {_f(l1)} ({l1 - l0:+,.0f})")})
        d["coverage"] = sup
        d["like_for_like"] = {"prev": l0, "curr": l1, "groups": int(len(both_in))}
    else:
        d["hypotheses"].append({"h": "data coverage", "status": "NOT SUPPORTED",
                                "evidence": f"all {n} {group} values report in both periods"})
    if total == 0:
        return d
    b = both.fillna(0).copy()
    if d.get("coverage"):
        # The new / dropped groups ARE the coverage explanation: where it moved is tested on the groups that
        # report in both periods only (live 11 Nov: 6 Kashi / Triveni zones '0 -> 24,966' read as 'broad-based').
        b = both.dropna(subset=["prev", "curr"]).copy()
        t0, total = float(b["prev"].sum()), float(b["curr"].sum() - b["prev"].sum())
        n = int(len(b))
        group = f"{group} (like-for-like)"
        if n < 3 or total == 0:
            d["hypotheses"].append({"h": f"localised / broad-based across {group} values", "status": "NOT TESTABLE",
                                    "evidence": f"only {n} value(s) report in both periods"})
            return d
    b["chg"] = b["curr"] - b["prev"]
    sign = np.sign(total)
    same = b[np.sign(b["chg"]) == sign].sort_values("chg", key=lambda s: -s.abs())
    against = b[np.sign(b["chg"]) == -sign].sort_values("chg", key=lambda s: -s.abs())
    # base share: the group's share of the previous period. A group that is 96% of the volume explaining 98%
    # of the change only follows volume (live 11 Nov: CONSUMER meters, 98% of an installation change).
    base = t0 if t0 else float(b["curr"].sum()) or 1.0
    d["top"] = [{"name": str(i), "prev": float(r.prev), "curr": float(r.curr), "chg": float(r.chg),
                 "share": float(r.chg / total * 100), "base_share": float((r.prev if t0 else r.curr) / base * 100)}
                for i, r in same.head(3).iterrows()]
    for t in d["top"]:
        t["excess"] = t["share"] - t["base_share"]
    d["against"] = [{"name": str(i), "chg": float(r.chg)} for i, r in against.head(2).iterrows()]
    d["moved_with_trend"] = int(len(same))
    if not d["top"]:
        return d
    top = max(d["top"], key=lambda t: t["excess"])     # the most disproportionate mover
    big = d["top"][0]
    loc = n >= 3 and top["share"] >= 40 and top["excess"] >= 15
    if loc:
        ev = (f"{top['name']} {_f(top['prev'])} -> {_f(top['curr'])} ({top['chg']:+,.0f}): {top['share']:.0f}% of the "
              f"change from {top['base_share']:.0f}% of the base")
    elif big["share"] >= 50:
        ev = (f"{big['name']} explains {big['share']:.0f}% of the change but is {big['base_share']:.0f}% of the base -- it "
              "follows volume, not a disproportionate move")
    else:
        ev = f"largest: {big['name']} {big['chg']:+,.0f} ({big['share']:.0f}% of the change, {big['base_share']:.0f}% of the base)"
    d["lead_score"] = top["excess"] if loc else None
    d["lead_item"] = top if loc else None
    d["hypotheses"].append({"h": f"localised in one {group}", "status": "SUPPORTED" if loc else "NOT SUPPORTED",
                            "evidence": ev})
    broad = n >= 3 and len(same) / n >= 0.7 and not loc and big["share"] < 50
    d["hypotheses"].append({"h": f"broad-based across {group} values", "status": "SUPPORTED" if broad else "NOT SUPPORTED",
                            "evidence": f"{len(same)} of {n} moved the same way; top 3 explain "
                                        f"{sum(t['share'] for t in d['top']):.0f}%"})
    return d


def assess(question: str, main, extras: list[tuple[str, object]] | None = None,
           today: pd.Timestamp | None = None, tables: list[str] | None = None) -> tuple[str, dict | None]:
    """(INVESTIGATION block for the answer prompt, structured result) -- ('', None) when the
    question is not a why / investigate ask or no result splits a measure over two periods."""
    if not asked(question):
        return "", None
    frames = [("main", main)] + list(extras or [])
    dims: list[dict] = []
    for purpose, df in frames:
        try:
            got = drivers.pair(question, df, today)
        except Exception:
            got = None
        if not got:
            continue
        both, measure, group, p0, p1, partial = got
        if len(both) < 2:
            continue
        d = _dimension(both, measure, group, p0, p1, partial)
        if d.get("hypotheses"):
            dims.append(d)
    if not dims:
        gap = _single_period(question, main, tables)
        if gap:
            return gap, {"measure": None, "periods": None, "dimensions": [], "contradictions": [], "lead": None,
                         "premise": None, "coverage_gap": True,
                         "confidence": {"level": "low", "reasons": ["the rows cover one period only"]}}
        return "", None
    main_d = dims[0]
    # Reconciliation (contradiction detection): additive splits of the same measure over the same periods
    # must agree on the totals.
    contradictions = []
    for d in dims[1:]:
        if d["rate"] or main_d["rate"] or d["periods"] != main_d["periods"]:
            continue
        for k in ("total_prev", "total_curr"):
            a, b = main_d.get(k), d.get(k)
            if a is None or b is None:
                continue
            if abs(a - b) > max(1.0, 0.005 * max(abs(a), abs(b))):
                contradictions.append(f"split by {main_d['by']} and split by {d['by']} disagree on the "
                                      f"{'previous' if k == 'total_prev' else 'current'} total ({_f(a)} vs {_f(b)}): rows "
                                      f"without a {d['by']} / {main_d['by']} value or a different filter -- quote the "
                                      f"{main_d['by']} total and say the other split does not cover every row")
                break
    # Premise: 'why did X drop' when X rose (live 11 Nov: installations 'dropped' while they rose 58.6%).
    premise = None
    if not main_d["rate"] and main_d.get("change"):
        says_down, says_up = bool(_DOWN_RE.search(question)), bool(_UP_RE.search(question))
        if says_down != says_up and (main_d["change"] > 0) == says_down:
            premise = (f"PREMISE CHECK: the question says it {'fell' if says_down else 'rose'}, but it "
                       f"{'ROSE' if main_d['change'] > 0 else 'FELL'} ({_f(main_d['total_prev'])} -> "
                       f"{_f(main_d['total_curr'])}, {main_d['change']:+,.0f}). Correct the premise in the first "
                       "sentence, then explain the actual move.")
    # Confidence: reasons the conclusion may not hold.
    reasons = []
    if any(d.get("coverage") for d in dims):
        reasons.append("part of the change is a data-coverage gap, not a real move")
    if not main_d["rate"] and (main_d.get("total_prev") or 0) < 30:
        reasons.append(f"small base ({_f(main_d.get('total_prev') or 0)} in {main_d['periods'][0]}): % changes are unreliable")
    if main_d.get("partial"):
        reasons.append(f"{main_d['partial']} still in progress (left out)")
    if len(dims) == 1:
        reasons.append("only one split was available to test")
    if contradictions:
        reasons.append("the splits disagree on the total")
    pct = main_d.get("pct")
    if pct is not None and abs(pct) < 2:
        reasons.append(f"the change is small ({pct:+.1f}%) -- may be normal month-to-month variation")
    level = "low" if any(r.startswith(("part of the change", "small base")) for r in reasons) else \
        "medium" if reasons else "high"
    # Strongest lead: the split where one group explains the most.
    leads = [(d["lead_score"], d) for d in dims if d.get("lead_score") is not None and not d.get("coverage")]
    lead = max(leads, key=lambda x: x[0])[1] if leads else None
    out = {"measure": main_d["measure"], "periods": main_d["periods"], "dimensions": dims,
           "contradictions": contradictions, "confidence": {"level": level, "reasons": reasons},
           "lead": ({"by": lead["by"], **lead["lead_item"]} if lead else None), "premise": premise}
    lines = [f"INVESTIGATION (each explanation tested by code on {len(dims)} split(s) of the rows -- figures exact). "
             "Structure the answer: (1) verdict -- the explanation the data SUPPORTS, coverage first (if the change is a "
             "data-coverage gap, say so and do NOT name a place as the cause); (2) evidence -- the figures below; (3) what "
             "was ruled out (NOT SUPPORTED); (4) confidence and why; (5) one concrete action aimed at the lead. This shows "
             "WHERE the change is, not the field-level cause: never invent one (staff, weather, supply) the data does not show."]
    for d in dims:
        p0, p1 = d["periods"]
        if d["rate"]:
            head = f"[by {d['by']}] {d['measure']} (a rate), {p0} vs {p1}, {d['groups']} values: average move {d['change']:+.2f} points"
        else:
            pc = f", {d['pct']:+.1f}%" if d.get("pct") is not None else ""
            head = f"[by {d['by']}] {d['measure']}, {p0} vs {p1}: total {_f(d['total_prev'])} -> {_f(d['total_curr'])} ({d['change']:+,.0f}{pc}), {d['groups']} values"
        lines.append(head)
        for t in d["top"]:
            sh = f", {t['share']:.0f}% of the change" if "share" in t else ""
            unit = " points" if d["rate"] else ""
            lines.append(f"  driver: {t['name']} {_f(t['prev'])} -> {_f(t['curr'])} ({t['chg']:+,.2f}{unit}{sh})"
                         if d["rate"] else f"  driver: {t['name']} {_f(t['prev'])} -> {_f(t['curr'])} ({t['chg']:+,.0f}{sh})")
        for a in d.get("against") or []:
            lines.append(f"  against the trend: {a['name']} ({a['chg']:+,.0f})")
        for h in d["hypotheses"]:
            lines.append(f"  H: {h['h']} -- {h['status']}: {h['evidence']}")
    for c in contradictions:
        lines.append(f"CONTRADICTION: {c}")
    if premise:
        lines.insert(1, premise)
    if lead:
        lines.append(f"LEAD: {lead['by']} = {lead['lead_item']['name']}")
    elif dims and not any(d.get("coverage") for d in dims):
        lines.append("LEAD: none -- no group moved out of proportion to its size; the change is spread with volume")
    lines.append(f"CONFIDENCE: {level}" + (f" -- {'; '.join(reasons)}" if reasons else " -- splits agree, base and periods complete"))
    return "\n".join(lines[:40]), out


# --------------------------------------------------------------------------- #
# Code-generated splits: the planner often skips the extra breakdowns the hint asks for (live 11 Nov: the
# installation 'drop' came back split by zone only). When the main query is a plain single-table GROUP BY on
# one dimension, the same query is re-run with that column swapped for up to 2 other dimension columns of the
# table -- no LLM call; each variant passes the same safety guard and runs read-only, in parallel.
# --------------------------------------------------------------------------- #

# One place split + one category split (discom is a group of projects: never a second place view).
_PLACE_DIMS = ["project", "cluster", "zone", "circle", "division", "sub_division"]
_CATEGORY_DIMS = ["progress_category", "category", "meter_type", "oem", "agency", "complaint_type", "feeder_category",
                  "warehouse", "sku"]
_TEXT_TYPE = re.compile(r"char|text|string|varchar", re.I)
SPLIT_WAIT_S = 15.0


def split_sqls(sql: str, columns: dict[str, str], engine: str, limit: int = 2) -> list[tuple[str, str]]:
    """[(dimension, sql)] variants of a single-table query grouped by one dimension column ([] otherwise)."""
    import sqlglot
    from sqlglot import exp
    from gmr_agents.execution.dialect import DIALECT

    dialect = "postgres" if engine == "postgres" else DIALECT
    try:
        root = sqlglot.parse_one(sql, read=dialect)
    except Exception:
        return []
    if not isinstance(root, exp.Select) or any(root.find(t) for t in (exp.With, exp.Join, exp.Subquery, exp.Union)):
        return []
    if len(list(root.find_all(exp.Table))) != 1:
        return []
    group = root.args.get("group")
    if not group or not group.expressions or not all(isinstance(e, exp.Column) for e in group.expressions):
        return []                                    # ROLLUP / expressions / positions: not touched
    cols = {k.lower(): v for k, v in columns.items()}
    dims = [e.name.lower() for e in group.expressions
            if e.name.lower() in cols and _TEXT_TYPE.search(str(cols[e.name.lower()]))]
    if len(dims) != 1:
        return []
    x = dims[0]
    where = root.args.get("where")
    pinned = {c.name.lower() for node in (where.find_all(exp.EQ, exp.In) if where else []) for c in node.find_all(exp.Column)}
    usable = lambda d: d in cols and d != x and d not in pinned and _TEXT_TYPE.search(str(cols[d]))
    cands = [next((d for d in group if usable(d)), None) for group in (_PLACE_DIMS, _CATEGORY_DIMS)]
    cands = [d for d in cands if d]
    out = []
    for d in cands[:limit]:
        new = root.copy()
        for c in new.find_all(exp.Column):
            if c.name.lower() == x:
                c.set("this", exp.to_identifier(d))
        out.append((d, new.sql(dialect=dialect)))
    return out


def auto_splits(plan: dict) -> list[tuple[str, object]]:
    """Run the code-generated splits for the plan's main query: [(purpose, DataFrame)] -- never raises."""
    try:
        from concurrent.futures import ThreadPoolExecutor, wait
        from gmr_agents.catalog.catalog import get_catalog
        from gmr_agents.execution.executor import run_sql
        from gmr_agents.guards import sql_guard

        tables, engine, sql = plan.get("tables") or [], plan.get("data_engine"), plan.get("sql") or ""
        if len(tables) != 1 or not engine or not sql:
            return []
        catalog = get_catalog("all")
        meta = catalog.get_table(tables[0])
        if meta is None:
            return []
        variants = split_sqls(sql, {c.column_name: c.sql_type for c in meta.columns}, engine)
        ok = [(d, s) for d, s in variants if sql_guard.check_sql_safety(s, catalog).get("ok")]
        if not ok:
            return []
        pool = ThreadPoolExecutor(max_workers=len(ok))
        futs = {pool.submit(run_sql, s, plan.get("database"), engine): d for d, s in ok}
        done, _ = wait(futs, timeout=SPLIT_WAIT_S)
        pool.shutdown(wait=False)
        out = []
        for f in done:
            try:
                out.append((f"by {futs[f]} (code split)", f.result()))
            except Exception:
                continue
        return out
    except Exception:
        return []
