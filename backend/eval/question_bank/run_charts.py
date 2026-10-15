"""Ask one question per chart type and report the chart the chatbot drew.

    python eval/question_bank/run_charts.py
"""
import json
import os
import sys
import time

import requests
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
API = os.environ.get("GMR_API", "http://127.0.0.1:8000")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    cases = yaml.safe_load(open(os.path.join(HERE, "chart_questions.yaml"), encoding="utf-8"))["charts"]
    hits, out = 0, []
    for c in cases:
        t0 = time.perf_counter()
        r = requests.post(f"{API}/api/ai-query", json={"question": c["q"]}, timeout=240).json()
        chart = (r.get("charts") or [{}])[0]
        got = chart.get("type", "none")
        ok = got in c["want"]
        hits += ok
        note = f" (chose {chart['chosen_type']}: {chart['fallback_reason']})" if chart.get("chosen_type") else ""
        print(f"{'ok ' if ok else 'XX '} {got:22s} want {'/'.join(c['want']):34s} {time.perf_counter() - t0:5.1f}s  "
              f"series={len(chart.get('series') or [1]) if chart else 0} pts={len(chart.get('x') or [])}{note}  | {c['q']}")
        out.append({**c, "got": got, "chart": chart, "summary": r.get("summary")})
        time.sleep(1)
    print(f"\n{hits}/{len(cases)} charts matched")
    with open(os.path.join(HERE, "results", "charts.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=str)


if __name__ == "__main__":
    main()
