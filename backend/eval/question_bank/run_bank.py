"""Run the GMR question bank against the live backend and record every answer.

    python eval/question_bank/run_bank.py                 # all questions
    python eval/question_bank/run_bank.py --ids mi01,su05 # only these (chains run whole)
    python eval/question_bank/run_bank.py --ids a         # a named set from id_sets.yaml
    python eval/question_bank/run_bank.py --section 4.6   # one section
    python eval/question_bank/run_bank.py --failed latest # re-run what failed last time

Questions in the same `chain` share one chat session, in order. Results go to
eval/question_bank/results/<run>.jsonl (one record per question) and a short
markdown report next to it. Pacing keeps the Groq free-tier token budget.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

import requests
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
API = os.environ.get("GMR_API", "http://127.0.0.1:8000")
RESULTS = os.path.join(HERE, "results")


def load_questions() -> list[dict]:
    with open(os.path.join(HERE, "questions.yaml"), encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    out = []
    for section, items in data["sections"].items():
        for item in items:
            out.append({**item, "section": section})
    return out


def ask(question: str, session_id: str | None) -> dict:
    t0 = time.perf_counter()
    try:
        r = requests.post(f"{API}/api/ai-query", json={"question": question, "session_id": session_id}, timeout=240)
        r.raise_for_status()
        body = r.json()
    except Exception as exc:
        body = {"summary": f"HTTP error: {exc}", "refused": True, "debug": {}}
    body["_seconds"] = round(time.perf_counter() - t0, 1)
    return body


def record(q: dict, resp: dict) -> dict:
    dbg = resp.get("debug") or {}
    trace = dbg.get("trace") or []
    triage = next((s for s in trace if s.get("step") == "triage"), {})
    first_table = (resp.get("tables") or [{}])[0]
    return {
        "id": q["id"], "section": q["section"], "chain": q.get("chain"), "question": q["q"],
        "expect_kind": q.get("kind"), "expect": q.get("expect"),
        "kind": triage.get("kind"), "resolved": dbg.get("resolved_question"), "topics": dbg.get("topics"),
        "sql": dbg.get("sql"), "reasoning": dbg.get("reasoning"), "refused": resp.get("refused"),
        "summary": resp.get("summary"), "insights": resp.get("insights"),
        "kpis": [f"{k.get('label')}: {k.get('formatted_value')}" for k in resp.get("kpis") or []],
        "table_cols": first_table.get("columns"), "table_rows": (first_table.get("rows") or [])[:12],
        "charts": [c.get("type") for c in resp.get("charts") or []],
        "repairs": sum(1 for s in trace if s.get("step") == "plan") - 1,
        "plan_errors": [s.get("reason") or s.get("error") for s in trace
                        if s.get("outcome") == "rejected" or s.get("error")],
        "models": sorted({c.get("model") for c in dbg.get("llm_calls") or [] if str(c.get("status", "")).startswith("ok")}),
        "busy": "busy" in str(resp.get("summary", "")).lower() and "rate limit" in str(resp.get("summary", "")).lower(),
        "seconds": resp.get("_seconds"), "session_id": resp.get("session_id"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids")
    ap.add_argument("--section")
    ap.add_argument("--failed", help="results file name (or 'latest') whose busy/error rows to re-run")
    ap.add_argument("--pause", type=float, default=12.0, help="seconds between questions (free-tier TPM)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--resume", action="store_true", help="skip ids already in results/<name>.jsonl and append")
    args = ap.parse_args()

    qs = load_questions()
    if args.ids:
        import yaml

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "id_sets.yaml"), encoding="utf-8") as fh:
            sets = (yaml.safe_load(fh) or {}).get("sets") or {}
        want = {i for part in args.ids.split(",") for i in (sets.get(part) or [part])}
        chains = {q.get("chain") for q in qs if q["id"] in want and q.get("chain")}
        qs = [q for q in qs if q["id"] in want or (q.get("chain") and q.get("chain") in chains)]
    if args.section:
        qs = [q for q in qs if q["section"].startswith(args.section)]
    if args.failed:
        files = sorted(f for f in os.listdir(RESULTS) if f.endswith(".jsonl"))
        name = files[-1] if args.failed == "latest" else args.failed
        with open(os.path.join(RESULTS, name), encoding="utf-8") as fh:
            bad = {r["id"] for r in map(json.loads, fh) if r.get("busy") or "HTTP error" in str(r.get("summary"))}
        qs = [q for q in qs if q["id"] in bad]

    os.makedirs(RESULTS, exist_ok=True)
    run = args.name or datetime.now().strftime("run-%Y%m%d-%H%M%S")
    path = os.path.join(RESULTS, f"{run}.jsonl")
    if args.resume and os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            done = {json.loads(line)["id"] for line in fh if line.strip()}
        chains_done = {q.get("chain") for q in qs if q["id"] in done and q.get("chain")}
        qs = [q for q in qs if q["id"] not in done and q.get("chain") not in chains_done]
    sessions: dict[str, str] = {}
    print(f"{len(qs)} question(s) -> {path}", flush=True)
    for i, q in enumerate(qs, 1):
        sid = sessions.get(q["chain"]) if q.get("chain") else None
        resp = ask(q["q"], sid)
        for _ in range(2):  # free-tier busy: wait and retry once or twice
            if not ("rate limit" in str(resp.get("summary", "")).lower()):
                break
            time.sleep(60)
            resp = ask(q["q"], sid)
        if q.get("chain"):
            sessions[q["chain"]] = resp.get("session_id")
        rec = record(q, resp)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        flag = "BUSY" if rec["busy"] else ("REFUSED" if rec["refused"] else "ok")
        print(f"[{i}/{len(qs)}] {q['id']} {flag} {rec['seconds']}s :: {str(rec['summary'])[:150]}", flush=True)
        time.sleep(args.pause)
    print("done", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
