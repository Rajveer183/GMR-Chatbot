"""Compact review of a results file: python eval/question_bank/show.py run1 [from] [to]"""

import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
name = sys.argv[1] if len(sys.argv) > 1 else "run1"
lo = int(sys.argv[2]) if len(sys.argv) > 2 else 0
hi = int(sys.argv[3]) if len(sys.argv) > 3 else 10**6
with open(os.path.join(HERE, "results", f"{name}.jsonl"), encoding="utf-8") as fh:
    rows = [json.loads(line) for line in fh]
for n, r in enumerate(rows[lo:hi], lo):
    print(f"=== #{n} {r['id']} [{r['kind']}/{r['expect_kind']}] {r['seconds']}s repairs={r['repairs']} {r['question']}")
    print(f"  expect: {r['expect']}")
    if r.get("resolved") and r["resolved"] != r["question"]:
        print(f"  resolved: {r['resolved']}")
    print(f"  ANSWER: {r['summary']}")
    if r.get("kpis"):
        print(f"  kpis: {r['kpis']}")
    if r.get("table_rows"):
        print(f"  rows: {json.dumps(r['table_rows'][:5], ensure_ascii=False)[:500]}")
    if r.get("plan_errors"):
        print(f"  plan_errors: {r['plan_errors']}")
