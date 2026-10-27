"""Ask every dashboard-parity chart question and check the chart against the dashboard card.

    python eval/question_bank/run_dashboard_charts.py [--workers 2]

Checks per question: tagged with the expected card (charts[0].dashboard) and drawn as one of
the card's chart types. Colours are checked by frontend/scripts/verify-dashboard-colours.mjs
over the saved results (results/dashboard_charts.json).
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
API = os.environ.get("GMR_API", "http://127.0.0.1:8000")


def ask(case: dict) -> dict:
    t0 = time.perf_counter()
    try:
        r = requests.post(f"{API}/api/ai-query", json={"question": case["q"]}, timeout=400).json()
    except Exception as exc:   # one failed request is a FAIL row, not a crashed run
        r = {"error": str(exc)}
    chart = (r.get("charts") or [{}])[0]
    got, tag = chart.get("type", "none"), chart.get("dashboard")
    return {**case, "got": got, "tag": tag, "ok_type": got in case["want"], "ok_kpi": tag == case["kpi"],
            "secs": round(time.perf_counter() - t0, 1), "chart": chart, "summary": r.get("summary"),
            "error": r.get("error") or r.get("detail")}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--misses", action="store_true",
                    help="re-ask only the questions the saved results missed (e.g. 'AI busy' answers); keep the rest")
    args = ap.parse_args()
    bank = yaml.safe_load(open(os.path.join(HERE, "dashboard_chart_questions.yaml"), encoding="utf-8"))["modules"]
    cases = [{**c, "module": m} for m, cs in bank.items() for c in cs]
    saved_path = os.path.join(HERE, "results", "dashboard_charts.json")
    kept: dict[str, dict] = {}
    if args.misses and os.path.exists(saved_path):
        for r in json.load(open(saved_path, encoding="utf-8")):
            case = next((c for c in cases if c["q"] == r["q"]), None)
            # re-checked against the bank as it is now (expectations may have been corrected)
            if case and r["got"] in case["want"] and r["tag"] == case["kpi"]:
                kept[r["q"]] = {**r, **case, "ok_type": True, "ok_kpi": True}
    with ThreadPoolExecutor(args.workers) as pool:
        asked = dict(zip([c["q"] for c in cases if c["q"] not in kept],
                         pool.map(ask, [c for c in cases if c["q"] not in kept])))
    out = [kept.get(c["q"]) or asked[c["q"]] for c in cases]
    module = None
    for r in out:
        if r["module"] != module:
            module = r["module"]
            print(f"\n{module}")
        ok = r["ok_type"] and r["ok_kpi"]
        series = [s["name"] for s in r["chart"].get("series") or []] or ["(single)"]
        rates = [s["name"] for s in r["chart"].get("rates") or []]
        want_kpi = "" if r["ok_kpi"] else f" (kpi {r['kpi']})"
        extra = (f" rates={rates}" if rates else "") + (f"  ERROR {r['error']}" if r["error"] else "")
        print(f"  {'ok ' if ok else 'XX '} {r['got']:22s} kpi={str(r['tag']):18s} want {'/'.join(r['want'])}{want_kpi}"
              f" {r['secs']:5.1f}s n={len(r['chart'].get('x') or [])} series={series}{extra} | {r['q']}")
    hits = sum(r["ok_type"] and r["ok_kpi"] for r in out)
    print(f"\n{hits}/{len(out)} charts match the dashboard card "
          f"(card tag {sum(r['ok_kpi'] for r in out)}/{len(out)}, type {sum(r['ok_type'] for r in out)}/{len(out)})")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "dashboard_charts.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=str)


if __name__ == "__main__":
    main()
