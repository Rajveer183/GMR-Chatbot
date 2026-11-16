"""Live answers vs verified SQL -- no LLM calls.

For every logged question (outputs/logs/questions.jsonl) whose wording is a verified example's
q[0] or `suggest` (the wordings that are meant to get exactly that example's answer), run the
logged live SQL and the example's SQL now, read-only, and compare the rows:

  SAME    same values (rows sorted, numbers rounded to 2 dp; column names ignored)
  SUBSET  every column of one answer appears, value for value, in the other
  DIFF    anything else -- read these by hand
  ERROR   one of the two queries failed now

    python eval/question_bank/parity_audit.py                 # questions since 2026-11-06
    python eval/question_bank/parity_audit.py --since 2026-11-10 --all-wordings --limit 80
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
warnings.filterwarnings("ignore")

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

import pandas as pd  # noqa: E402

from gmr_agents.catalog import business_rules  # noqa: E402
from gmr_agents.catalog.catalog import get_catalog  # noqa: E402
from gmr_agents.execution import executor  # noqa: E402
from gmr_agents.guards import sql_guard  # noqa: E402


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w%&\s]", " ", str(text or "").lower())).strip()


def run(sql: str) -> pd.DataFrame:
    domain = "mdm" if ("hes_gold." in sql or "hes_silver." in sql) else "wfm"
    guard = sql_guard.check_sql_safety(sql.strip().rstrip(";"), get_catalog(domain))
    if not guard["ok"]:
        raise ValueError(f"guard: {guard['reason']}")
    return executor.run_sql(guard["normalized_sql"], None, guard["data_engine"])


def columns(df: pd.DataFrame) -> list[tuple]:
    out = []
    for c in df.columns:
        vals = []
        for v in df[c].tolist():
            try:
                f = float(v)
                vals.append("nan" if f != f else round(f, 2))  # NaN != NaN: compare blanks as equal
            except (TypeError, ValueError):
                vals.append(str(v).strip().lower())
        out.append(tuple(sorted(vals, key=str)))
    return out


def compare(live: pd.DataFrame, ref: pd.DataFrame) -> str:
    a, b = columns(live), columns(ref)
    if len(live) == len(ref) and sorted(a, key=str) == sorted(b, key=str):
        return "SAME"
    if len(live) == len(ref) and (set(a) <= set(b) or set(b) <= set(a)):
        return "SUBSET"
    return "DIFF"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-11-06")
    ap.add_argument("--all-wordings", action="store_true", help="also match q[1:] (patterns, may differ by place)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    wording: dict[str, dict] = {}
    for ex in business_rules.examples():
        qs = list(ex.get("q") or [])
        keys = qs if args.all_wordings else qs[:1] + ([ex["suggest"]] if ex.get("suggest") else [])
        for q in keys:
            wording.setdefault(norm(q), ex)

    latest: dict[str, tuple[str, dict]] = {}
    for line in open(os.path.join("outputs", "logs", "questions.jsonl"), encoding="utf-8"):
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if str(rec.get("at") or "") < args.since or rec.get("refused") or rec.get("error"):
            continue
        sql = rec.get("sql")
        ex = wording.get(norm(rec.get("question")))
        if ex and isinstance(sql, str) and sql.strip():
            latest[norm(rec.get("question"))] = (sql, ex)  # the most recent answer per wording

    items = list(latest.items())[: args.limit or None]
    counts: dict[str, int] = {}
    for q, (sql, ex) in items:
        try:
            verdict = compare(run(sql), run(ex["sql"]))
            detail = ""
        except Exception as exc:
            verdict, detail = "ERROR", f"{type(exc).__name__}: {str(exc)[:160]}"
        counts[verdict] = counts.get(verdict, 0) + 1
        if verdict != "SAME":
            print(f"{verdict:6} | {q[:90]}" + (f" | {detail}" if detail else ""))
    print(json.dumps({"pairs": len(items), **counts}))


if __name__ == "__main__":
    main()
