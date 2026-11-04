"""Service metrics from the per-question and per-LLM-call logs -- what to optimise against.

    GET /api/metrics?hours=24                      (JSON)
    python -m gmr_agents.metrics --hours 24        (same, printed)

Read from outputs/logs/questions.jsonl (one record per question, cache hits included)
and outputs/logs/llm_calls.jsonl (one record per LLM attempt). Only real traffic counts:
records without a session (in-process evaluation scripts) are left out; test runs log
elsewhere (shared/config.py TESTING).

    latency      P50 / P95 total time and time to first token (first numbers or words shown)
    tokens       input / output tokens per question (LLM-answered questions)
    cost         estimated USD per question (free tier = 0 unless LLM_PRICING is set)
    work         LLM calls, failed LLM attempts, tool calls, DB queries, agent steps per question
    quality      refusal, AI-busy, error, SQL-repair and number-correction rates; terminations
    cache        answer-cache hit rate
    cost_per_success  tokens / LLM calls / USD of ALL questions divided by the answered ones,
                 and the share of tokens spent on questions that were refused or failed
    attribution  heaviest sessions (the only user identity the app has) and query patterns
                 (kind + topic) by tokens, with tokens per success, success rate and P50 time
    targets      SLO_* settings from .env and whether each is met
"""

from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime, timedelta, timezone

from shared import config

_IST = timezone(timedelta(hours=5, minutes=30))


def _read(name: str, since: str) -> list[dict]:
    path = os.path.join(config.LOG_DIR, f"{name}.jsonl")
    out = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.startswith('{"at": "') and line[8:27] < since:
                    continue  # cheap pre-filter: every record starts with its timestamp
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if str(r.get("at", "")) >= since:
                    out.append(r)
    except OSError:
        pass
    return out


def _pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    v = sorted(values)
    return round(v[min(len(v) - 1, int(round(p / 100 * (len(v) - 1))))], 1)


def _rate(n: int, d: int) -> float | None:
    return round(n / d, 4) if d else None


def _succeeded(q: dict) -> bool:
    """A finished task: answered, not refused / busy / errored (a cache hit counts)."""
    return not q.get("refused") and not q.get("error")


def _pattern(q: dict) -> str:
    """Query pattern for cost attribution: kind + first topic (else first table)."""
    if q.get("cached"):
        return "cache_hit"
    what = (q.get("topics") or q.get("tables") or ["-"])[0]
    return f"{q.get('kind') or 'none'}:{what}"


def _cost_per_success(qs: list[dict], ex) -> dict:
    """Everything spent -- including on refused / failed questions -- divided by the
    questions that were actually answered. Cost per LLM call hides retries and failures."""
    ok = [q for q in qs if _succeeded(q)]
    tokens = sum(q.get("tokens") or 0 for q in qs)
    wasted = sum(q.get("tokens") or 0 for q in qs if not _succeeded(q))
    d = len(ok)
    return {"successes": d, "success_rate": _rate(d, len(qs)),
            "tokens": round(tokens / d) if d else None,
            "llm_calls": round(sum(q.get("llm_calls") or ex(q, "llm_calls") for q in qs) / d, 2) if d else None,
            "usd": round(sum(ex(q, "estimated_cost_usd", 0.0) for q in qs) / d, 6) if d else None,
            "wasted_token_share": _rate(wasted, tokens)}


def _attribute(qs: list[dict], ex, key, top: int = 10) -> list[dict]:
    """Heaviest groups by total tokens: who / which query shapes the spend goes to."""
    groups: dict[str, list[dict]] = {}
    for q in qs:
        groups.setdefault(key(q), []).append(q)
    total = sum(q.get("tokens") or 0 for q in qs) or 1
    rows = []
    for k, g in groups.items():
        t = sum(q.get("tokens") or 0 for q in g)
        ok = sum(1 for q in g if _succeeded(q))
        rows.append({"key": k, "questions": len(g), "tokens": t, "token_share": round(t / total, 3),
                     "tokens_per_success": round(t / ok) if ok else None, "success_rate": _rate(ok, len(g)),
                     "p50_ms": _pct([q.get("total_ms") or 0 for q in g], 50),
                     "usd": round(sum(ex(q, "estimated_cost_usd", 0.0) for q in g), 6)})
    return sorted(rows, key=lambda r: -r["tokens"])[:top]


def compute(hours: float = 24.0) -> dict:
    now = datetime.now(_IST)
    since = (now - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%S")
    qs = [q for q in _read("questions", since) if q.get("session_id")]
    traces = {q.get("trace_id") for q in qs}
    calls = [c for c in _read("llm_calls", since) if c.get("trace_id") in traces]
    n = len(qs)
    cached = [q for q in qs if q.get("cached")]
    live = [q for q in qs if not q.get("cached")]
    llm_answered = [q for q in live if (q.get("llm_calls") or 0) > 0]

    def ex(q: dict, key: str, default=0):
        return (q.get("execution") or {}).get(key, default) or default

    lat = [q.get("total_ms") or 0 for q in qs]
    ttft = [ex(q, "ttft_ms", q.get("total_ms") or 0) for q in qs]
    busy = [q for q in live if "busy" in str(q.get("summary") or "").lower() and q.get("refused")]
    repaired = [q for q in live if sum(1 for s in q.get("steps") or [] if s.get("step") == "plan") > 1]
    corrected = [q for q in live if any(s.get("step") == "number_check" for s in q.get("steps") or [])]
    failed_attempts = [c for c in calls if not str(c.get("status", "")).startswith("ok")]

    m = {
        "window": {"hours": hours, "since": since, "questions": n, "llm_calls_logged": len(calls)},
        "latency_ms": {"p50": _pct(lat, 50), "p95": _pct(lat, 95),
                       "p50_uncached": _pct([q.get("total_ms") or 0 for q in live], 50),
                       "p95_uncached": _pct([q.get("total_ms") or 0 for q in live], 95),
                       "ttft_p50": _pct(ttft, 50), "ttft_p95": _pct(ttft, 95)},
        "tokens_per_question": {
            # older records have no per-question input/output split -- leave them out, not count them as 0
            "input_p50": _pct([ex(q, "input_tokens") for q in llm_answered if ex(q, "input_tokens")], 50),
            "output_p50": _pct([ex(q, "output_tokens") for q in llm_answered if ex(q, "input_tokens")], 50),
            "total_mean": round(sum(q.get("tokens") or 0 for q in llm_answered) / len(llm_answered)) if llm_answered else None,
            "total_p95": _pct([q.get("tokens") or 0 for q in llm_answered], 95)},
        "cost_usd_per_question": round(sum(ex(q, "estimated_cost_usd", 0.0) for q in qs) / n, 6) if n else None,
        "work_per_question": {
            "llm_calls_mean": round(sum(ex(q, "llm_calls") for q in live) / len(live), 2) if live else None,
            "failed_llm_attempts": len(failed_attempts),
            "failed_llm_attempt_ms": round(sum(c.get("ms") or 0 for c in failed_attempts)),
            "tool_calls_mean": round(sum(ex(q, "tool_calls") for q in live) / len(live), 2) if live else None,
            "db_queries_mean": round(sum(ex(q, "db_queries") for q in live) / len(live), 2) if live else None,
            "agent_steps_mean": round(sum(ex(q, "steps") for q in live) / len(live), 2) if live else None},
        "quality": {
            "refusal_rate": _rate(sum(1 for q in qs if q.get("refused")), n),
            "busy_rate": _rate(len(busy), n),
            "error_rate": _rate(sum(1 for q in qs if q.get("error")), n),
            "sql_repair_rate": _rate(len(repaired), len(live)),
            "number_correction_rate": _rate(len(corrected), len(llm_answered)),
            "terminations": dict(Counter(ex(q, "termination", "") or q.get("termination") or "" for q in qs).most_common())},
        "cache": {"answer_cache_hit_rate": _rate(len(cached), n)},
        "llm_failures_by_model": dict(Counter(f"{c.get('model')}:{c.get('status')}" for c in failed_attempts).most_common(8)),
        "cost_per_success": _cost_per_success(qs, ex),
        "attribution": {"by_session": _attribute(qs, ex, lambda q: q.get("session_id") or "-"),
                        "by_pattern": _attribute(qs, ex, _pattern)},
    }
    targets = {"latency_ms.p50": config.SLO_P50_MS, "latency_ms.p95": config.SLO_P95_MS,
               "latency_ms.ttft_p50": config.SLO_TTFT_P50_MS, "quality.refusal_rate": config.SLO_MAX_REFUSAL_RATE,
               "quality.busy_rate": config.SLO_MAX_BUSY_RATE, "quality.sql_repair_rate": config.SLO_MAX_REPAIR_RATE,
               "tokens_per_question.total_mean": config.SLO_MAX_TOKENS_PER_QUESTION}
    m["targets"] = {}
    for path, target in targets.items():
        section, key = path.split(".")
        value = m[section].get(key)
        m["targets"][path] = {"target": target, "value": value,
                              "met": None if value is None or not target else value <= target}
    return m


if __name__ == "__main__":
    import argparse
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=24.0)
    print(json.dumps(compute(ap.parse_args().hours), indent=1))
