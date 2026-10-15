"""Run every SQL in gmr_agents/schema/data/examples.yaml through the same
safety check and executor the chatbot uses, and print the first rows.

    python eval/question_bank/verify_examples.py          # all
    python eval/question_bank/verify_examples.py 12 13    # by index
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

import pandas as pd  # noqa: E402

from gmr_agents.schema import business_rules  # noqa: E402
from gmr_agents.schema.catalog import get_catalog  # noqa: E402
from gmr_agents.sql import executor  # noqa: E402
from gmr_agents.validation import sql_guard  # noqa: E402

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 20)


def run(sql: str) -> str:
    domain = "mdm" if "hes_gold." in sql else "wfm"
    catalog = get_catalog(domain)
    guard = sql_guard.check_sql_safety(sql.strip().rstrip(";"), catalog)
    if not guard["ok"]:
        return f"  GUARD REJECTED: {guard['reason']}"
    plan = {"engine": "sql", "data_engine": guard["data_engine"], "sql": guard["normalized_sql"], "tables": [],
            "metrics": [], "group_by": [], "filters": [], "warnings": [], "labels": {}, "skip_scope_count": True,
            "database": None}
    res = executor.execute_sql(plan, catalog)
    if res.get("execution_error"):
        return f"  DB ERROR: {res.get('error_text')}"
    df = res.get("result")
    return f"  {len(df)} row(s)\n" + df.head(8).to_string(max_colwidth=40)


def main() -> None:
    examples = business_rules.examples()
    pick = {int(a) for a in sys.argv[1:]}
    bad = 0
    for i, ex in enumerate(examples):
        if pick and i not in pick:
            continue
        print(f"\n#{i} {ex['q'][0]}")
        for sql in [ex["sql"]] + [e["sql"] for e in ex.get("extra") or []]:
            out = run(sql)
            bad += "REJECTED" in out or "DB ERROR" in out
            print(out)
    print(f"\n{bad} failing quer(ies)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
