"""Grade a KPI-card run (results/<name>.jsonl from run_bank.py --ids k_all).

    python eval/question_bank/grade_kpi.py kpi1

Per question, automatic checks (no LLM):
  answered   not refused / busy / HTTP error
  kind       explain-type questions answered as explain (no SQL), data questions with SQL
  table      the SQL reads the card's own gold table (the dashboard KPI source), or a
             gmr_wfm_silver table for a single-meter lookup
  rows       the result has rows
  repairs    SQL repair rounds
  seconds    latency
A question passes when answered + kind + table + rows hold. Prints a per-card table,
the failures, and writes results/<name>_grade.json.
"""

from __future__ import annotations

import json
import os
import re
import statistics as st
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from gmr_agents.schema.table_metadata import SCHEMA_RE, WFM_TABLES  # noqa: E402

# Card prefix -> the gold tables its KPI is built from (backend kpi_catalog / kpi_definitions.yaml).
CARD_TABLES = {
    "mp": {"gold_mi_installation_day_agg", "gold_mi_installation_month_agg", "gold_mi_sat_invoice_snapshot",
           "gold_meter_defect_event_day_agg"},
    "pr": {"gold_mi_technician_productivity_day_agg", "gold_mi_technician_productivity_month_agg",
           "gold_mi_productivity_trend_month_agg"},
    "rp": {"gold_meter_defect_event_day_agg", "gold_mi_installation_day_agg", "gold_mi_installation_month_agg"},
    "ut": {"gold_meter_inventory_event_day_agg", "gold_meter_inventory_event_month_agg", "gold_inventory_monthly_balance",
           "gold_mi_installation_day_agg", "gold_mi_installation_month_agg"},
    "sk": {"gold_inventory_unutilized_stock_snapshot"},
    "ms": {"gold_mi_sat_invoice_snapshot", "gold_mi_sat_invoice_day_agg", "sat_eligibility_config"},
    "na": {"gold_non_sat_ageing_snapshot"},
    "jt": {"gold_meter_journey_stage_agg"},
    "cs": {"gold_meter_revenue_pending_stage_snapshot", "gold_meter_installation_pending_stage_snapshot"},
    "fu": {"gold_mi_sat_invoice_snapshot", "gold_mi_sat_invoice_day_agg"},
    "rv": {"revenue_receivables"},
    "cb": {"revenue_receivables"},
    "ra": {"gold_revenue_ageing_snapshot"},
    "op": {"gold_om_technician_productivity_day_agg", "gold_om_technician_productivity_month_agg",
           "gold_om_closed_tickets_day_agg", "gold_om_meter_complaints_agg"},
    "oa": {"gold_om_open_ageing_snapshot", "gold_om_meter_complaints_agg"},
    "oc": {"gold_om_closed_tickets_day_agg"},
}
_TBL = re.compile(rf"\b{SCHEMA_RE}\.\"?([A-Za-z0-9_]+)", re.I)
_SILVER = re.compile(r"\b(?:" + "|".join(map(re.escape, WFM_TABLES)) + r")\.", re.I)
_METER = re.compile(r"\b[A-Z]{2}\d{6,9}\b")
_TICKET = re.compile(r"\bticket \d{5,}\b|\btickets? (closed|on meter)\b|\bcomplaints? on meter\b|technician \"|technician [A-Z]{3,}", re.I)


def grade(rec: dict) -> dict:
    prefix = re.match(r"[a-z]+", rec["id"]).group(0)
    sql = str(rec.get("sql") or "")
    used = {t.lower() for t in _TBL.findall(sql)}
    busy = bool(rec.get("busy")) or "HTTP error" in str(rec.get("summary"))
    answered = not busy and not rec.get("refused")
    expect = rec.get("expect_kind")
    explain_like = expect == "explain"
    kind_ok = (not sql) if explain_like else (bool(sql) if expect == "data" else True)
    lookup = bool(_METER.search(rec["question"])) or bool(_TICKET.search(rec["question"]))
    if not sql:
        table_ok = explain_like or expect == "any"
    elif lookup:
        table_ok = bool(_SILVER.search(sql)) or bool(used & CARD_TABLES.get(prefix, set()))
    else:
        table_ok = bool(used & CARD_TABLES.get(prefix, set()))
    rows = len(rec.get("table_rows") or []) or len(rec.get("kpis") or [])
    rows_ok = (not sql) or rows > 0
    ok = answered and kind_ok and table_ok and rows_ok
    why = [w for w, bad in (("busy", busy), ("refused", rec.get("refused") and not busy), ("kind", not kind_ok),
                            ("table", not table_ok), ("no rows", not rows_ok)) if bad]
    return {"id": rec["id"], "prefix": prefix, "ok": ok, "why": why, "repairs": max(0, rec.get("repairs") or 0),
            "seconds": rec.get("seconds") or 0, "question": rec["question"], "tables": sorted(used),
            "summary": str(rec.get("summary") or "")[:220], "plan_errors": rec.get("plan_errors") or []}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    name = sys.argv[1] if len(sys.argv) > 1 else "kpi1"
    with open(os.path.join(HERE, "results", f"{name}.jsonl"), encoding="utf-8") as fh:
        recs = [json.loads(line) for line in fh if line.strip()]
    graded = [grade(r) for r in recs]
    by = defaultdict(list)
    for g in graded:
        by[g["prefix"]].append(g)
    print(f"{'card':5s} {'n':>3s} {'pass':>5s} {'refused':>7s} {'table':>5s} {'kind':>4s} {'norows':>6s} {'busy':>4s} {'repairs':>7s} {'p50 s':>6s} {'p95 s':>6s}")
    for p, gs in by.items():
        secs = sorted(g["seconds"] for g in gs)
        cnt = lambda w: sum(w in g["why"] for g in gs)
        print(f"{p:5s} {len(gs):3d} {sum(g['ok'] for g in gs):5d} {cnt('refused'):7d} {cnt('table'):5d} {cnt('kind'):4d} "
              f"{cnt('no rows'):6d} {cnt('busy'):4d} {sum(g['repairs'] for g in gs):7d} {st.median(secs):6.1f} {secs[int(.95 * (len(secs) - 1))]:6.1f}")
    allsecs = sorted(g["seconds"] for g in graded)
    print(f"TOTAL {len(graded)} pass {sum(g['ok'] for g in graded)} ({sum(g['ok'] for g in graded) / len(graded):.0%}) "
          f"repairs {sum(g['repairs'] for g in graded)} p50 {st.median(allsecs):.1f}s p95 {allsecs[int(.95 * (len(allsecs) - 1))]:.1f}s")
    print("\nFAILURES:")
    for g in graded:
        if not g["ok"]:
            print(f"  {g['id']:6s} {','.join(g['why']):14s} {g['question'][:70]:70s} tables={g['tables']} :: {g['summary'][:120]}")
    with open(os.path.join(HERE, "results", f"{name}_grade.json"), "w", encoding="utf-8") as fh:
        json.dump(graded, fh, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
