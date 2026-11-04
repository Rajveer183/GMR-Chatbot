"""Offline replay: the exact prompts and outputs of the pipeline for logged questions.

    python eval/question_bank/replay_prompts.py --out before.jsonl            # record
    python eval/question_bank/replay_prompts.py --out after.jsonl --compare before.jsonl

A refactor gate that needs no LLM quota and no database: every logged question
(outputs/logs/questions.jsonl) runs through run_agent with
    * the LLM replaced by canned replies built from that question's own log
      (triage fields, SQL, extra SQL, summary),
    * the database replaced by a deterministic DataFrame per SQL,
    * the answer cache, result cache and log writes switched off.
Each LLM call is recorded as (task, sha of system, sha of prompt, max_tokens) and the
final answer (summary, insights, follow-ups, kpis, charts, tables, plan SQL, trace
step names) as a hash. A pure restructure must reproduce every record exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, BACKEND)

import pandas as pd  # noqa: E402
import sqlglot  # noqa: E402
from sqlglot import exp  # noqa: E402

from shared import llm_gateway  # noqa: E402

QUESTIONS_LOG = os.path.join(BACKEND, "outputs", "logs", "questions.jsonl")
_TEXT_HINTS = ("name", "project", "zone", "circle", "division", "discom", "month", "date", "day", "week", "type",
               "category", "status", "stage", "bucket", "label", "period", "feeder", "warehouse", "sku", "oem",
               "meter", "verdict", "reason", "id", "serial", "year", "level", "group", "region", "substation")


def sha(text: Any) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:16]


def load_cases(limit: int) -> list[dict]:
    rows, seen = [], set()
    with open(QUESTIONS_LOG, encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            q = r.get("question")
            if not q or q in seen or r.get("error") or not r.get("kind"):
                continue
            seen.add(q)
            rows.append(r)
    return rows[-limit:]


def canned_triage(case: dict) -> dict:
    step = next((s for s in case.get("steps") or [] if isinstance(s, dict) and s.get("step") == "triage"), {})
    return {"kind": case.get("kind"), "question": case.get("resolved_question") or case["question"],
            "followup": False, "domain": case.get("domain") or step.get("domain") or "wfm",
            "language": case.get("language") or "en", "parts": [], "examples": [],
            "assumption": "", "topics": case.get("topics") or step.get("topics") or [],
            "tables": step.get("tables") or case.get("tables") or [],
            "reply": case.get("summary") or "", "named": ""}


def canned_plan(case: dict) -> dict:
    if not case.get("sql"):
        return {"answer_kind": "unanswerable", "reason": case.get("summary") or "no data", "sql": ""}
    return {"reasoning": case.get("reasoning") or "", "answer_kind": "breakdown", "sql": case["sql"],
            "extra_queries": [{"purpose": f"extra {i}", "sql": s} for i, s in enumerate(case.get("extra_sql") or [], 1)],
            "labels": {}, "display": {"cards": [], "chart": {"type": "bar"}, "table": True}, "title": "Result"}


def fake_frame(sql: str, engine: str) -> pd.DataFrame:
    dialect = "postgres" if engine == "postgres" else "presto"
    try:
        tree = sqlglot.parse_one(sql, dialect=dialect)
        select = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
        cols = [e.alias_or_name for e in select.expressions] if select is not None else []
    except Exception:
        cols = []
    cols = [c for c in cols if c and c != "*"] or ["value"]
    if any(c == "n" for c in cols) and len(cols) == 1:
        return pd.DataFrame({"n": [3]})
    data = {}
    for c in cols:
        text = any(h in c.lower() for h in _TEXT_HINTS) and not c.lower().endswith(("_count", "_pct", "_total"))
        data[c] = [f"{c}_{i}" if text else int(sha(f"{c}{i}"), 16) % 5000 + i for i in range(3)]
    return pd.DataFrame(data)


class Recorder:
    def __init__(self) -> None:
        self.case: dict = {}
        self.calls: list[dict] = []

    def reply(self, task: str, prompt: str, system: str, max_tokens: Any) -> llm_gateway.LLMResult:
        self.calls.append({"task": task, "system": sha(system), "prompt": sha(prompt), "max_tokens": max_tokens,
                           "len": len(prompt)})
        c = self.case
        if task == "triage":
            data = canned_triage(c)
        elif task in ("sql", "repair"):
            data = canned_plan(c)
        elif task == "explain":
            data = {"summary": c.get("summary") or "", "insights": [], "citations": [], "supported": True,
                    "abstain": False, "conflicts": [], "followups": [1, 2, 3]}
        else:
            data = {"summary": c.get("summary") or "", "insights": [], "followups": [1, 2, 3], "text": ""}
        return llm_gateway.LLMResult(data=data, model="replay", provider="replay", prompt_tokens=len(prompt) // 4,
                                     completion_tokens=50, ms=1.0)


def install_stubs(rec: Recorder) -> None:
    def chat_json(task, prompt, *, system, models=None, max_tokens=800, temperature=None, effort="low"):
        return rec.reply(task, prompt, system, max_tokens)

    def chat_json_stream(task, prompt, *, system, on_text, models=None, max_tokens=800, temperature=None,
                         effort="low", **_kw):
        res = rec.reply(task, prompt, system, max_tokens)
        on_text(json.dumps(res.data))
        return res

    llm_gateway.chat_json = chat_json
    llm_gateway.chat_json_stream = chat_json_stream
    llm_gateway.limit_hit = lambda: None

    import importlib
    executor = importlib.import_module("gmr_agents.execution.executor")
    knowledge_store = importlib.import_module("gmr_agents.retrieval.knowledge_store")
    answer_cache = importlib.import_module("gmr_agents.ops.answer_cache")
    executor.run_sql = lambda sql, database=None, engine="athena": fake_frame(sql, engine)
    knowledge_store.get_result = lambda *a, **k: None
    knowledge_store.put_result = lambda *a, **k: None
    answer_cache.current_version = lambda: "2026-11-04 06:00:00|replay"
    # No log records from a replay: every module holding a reference to append_jsonl gets a no-op.
    from shared import logging_config
    for mod in list(sys.modules.values()):
        if getattr(mod, "append_jsonl", None) is logging_config.append_jsonl:
            mod.append_jsonl = lambda *a, **k: None
    logging_config.append_jsonl = lambda *a, **k: None


def _clean(v: Any) -> Any:
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items() if k not in ("ms", "t0", "ttft_ms", "at")}
    if isinstance(v, list):
        return [_clean(x) for x in v]
    if isinstance(v, pd.DataFrame):
        return v.astype(str).values.tolist()
    return v


def outcome(state: dict) -> dict:
    keys = ("summary", "insights", "followups", "kpis", "charts", "tables", "refused", "agent", "resolved_question",
            "answer_language", "citations", "row_count")
    out = {k: _clean(state.get(k)) for k in keys}
    plan = state.get("plan") or {}
    out["plan_sql"] = plan.get("sql")
    out["extra_sql"] = plan.get("extra_sql")
    out["steps"] = [s.get("step") for s in state.get("trace") or []]
    out["graph"] = state.get("graph")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--compare")
    args = ap.parse_args()

    rec = Recorder()
    install_stubs(rec)
    from gmr_agents.llm_pipeline import run_agent

    cases = load_cases(args.limit)
    with open(args.out, "w", encoding="utf-8") as fh:
        for i, case in enumerate(cases):
            rec.case, rec.calls = case, []
            state = run_agent(case["question"], trace_id=f"replay{i:04d}", session_id=f"replay-{i:04d}")
            out = outcome(state)
            fh.write(json.dumps({"i": i, "question": case["question"], "calls": rec.calls,
                                 "outcome": sha(json.dumps(out, sort_keys=True, default=str)),
                                 "steps": out["steps"]}, ensure_ascii=False) + "\n")
    print(f"replayed {len(cases)} questions -> {args.out}")
    if args.compare:
        compare(args.compare, args.out)


def compare(before_path: str, after_path: str) -> None:
    load = lambda p: [json.loads(line) for line in open(p, encoding="utf-8")]
    before, after = load(before_path), load(after_path)
    diffs = 0
    for b, a in zip(before, after):
        if b != a:
            diffs += 1
            if diffs <= 10:
                what = [k for k in ("calls", "outcome", "steps") if b.get(k) != a.get(k)]
                print(f"DIFF #{b['i']} {b['question'][:70]!r}: {what}")
    if len(before) != len(after):
        diffs += 1
        print(f"DIFF count {len(before)} vs {len(after)}")
    calls = sum(len(r["calls"]) for r in before)
    print(f"{len(before)} questions, {calls} LLM calls compared: {'IDENTICAL' if not diffs else f'{diffs} differ'}")
    sys.exit(1 if diffs else 0)


if __name__ == "__main__":
    main()
