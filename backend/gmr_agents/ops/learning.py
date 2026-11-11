"""Continuous learning from real traffic -- user feedback, failure root causes, recurring
failures, model comparison and auto-generated regression cases.

    POST /api/feedback                 {"trace_id", "session_id", "rating": "up"|"down"|"clear", "reason", "comment"}
    GET  /api/feedback?session_id=     {trace_id: rating} -- the UI's thumbs state after a reload
    GET  /api/insights/quality?hours=  everything below, as JSON
    python -m gmr_agents.ops.learning --hours 168 [--write-cases]

Reads only the logs the app already writes (outputs/logs/questions.jsonl, llm_calls.jsonl)
plus outputs/logs/feedback.jsonl. Nothing here runs on the question path: no LLM call, no
database query, no change to how an answer is made. Every number is computed by code.

    feedback            up / down counts, satisfaction, reasons, the latest downvoted answers
    root_causes         why questions did not get a good answer (one cause per question)
    unanswered          the same question failing again and again (normalised text)
    recurring_failures  validation / SQL error signatures (numbers, names and quotes masked)
    models              per model: attempts, success rate, P50 / P95 latency, tokens per call
    regression_cases    new eval cases from failures and downvotes (+ golden cases from upvotes)
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

from shared import config
from shared.logging_config import append_jsonl

_IST = timezone(timedelta(hours=5, minutes=30))

RATINGS = ("up", "down", "clear")
REASONS = ("wrong_number", "wrong_period", "wrong_filter", "wrong_chart", "incomplete", "not_what_i_asked",
           "slow", "unclear", "other")
_TRACE_RE = re.compile(r"^[0-9a-f]{6,32}$")

# Terminations that are a correct outcome, not a failure.
_EXPECTED_REFUSALS = {"out_of_scope", "chitchat"}


def _now() -> str:
    return datetime.now(_IST).strftime("%Y-%m-%dT%H:%M:%S")


def _since(hours: float) -> str:
    return (datetime.now(_IST) - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%S")


def _read(name: str, since: str = "") -> list[dict]:
    from gmr_agents.ops.metrics import _read as read_window

    return read_window(name, since)


def _pct(values: list[float], p: float) -> float | None:
    from gmr_agents.ops.metrics import _pct as pct

    return pct(values, p)


# --------------------------------------------------------------------------- #
# Feedback
# --------------------------------------------------------------------------- #

def record_feedback(trace_id: str, rating: str, *, session_id: str | None = None, reason: str | None = None,
                    comment: str | None = None) -> dict:
    """Append one vote. The latest vote per trace wins ('clear' withdraws it). Raises
    ValueError on a malformed trace id / rating so the API can answer 422."""
    trace_id = str(trace_id or "").strip().lower()
    if not _TRACE_RE.match(trace_id):
        raise ValueError("trace_id must be the answer's trace id.")
    if rating not in RATINGS:
        raise ValueError(f"rating must be one of {RATINGS}.")
    reason = reason if reason in REASONS else (None if not reason else "other")
    record = {"at": _now(), "trace_id": trace_id, "session_id": session_id, "rating": rating,
              "reason": reason if rating == "down" else None,
              "comment": (comment or "").strip()[:500] or None}
    append_jsonl("feedback", record)
    return record


def latest_votes(since: str = "", session_id: str | None = None) -> dict[str, dict]:
    """trace_id -> the latest feedback record for it ('clear' removes the vote)."""
    votes: dict[str, dict] = {}
    for r in _read("feedback", since):
        if session_id and r.get("session_id") != session_id:
            continue
        if r.get("rating") == "clear":
            votes.pop(r.get("trace_id"), None)
        elif r.get("trace_id"):
            votes[r["trace_id"]] = r
    return votes


def session_votes(session_id: str) -> dict[str, str]:
    return {t: v["rating"] for t, v in latest_votes(session_id=session_id).items()}


# --------------------------------------------------------------------------- #
# Root causes and failure signatures
# --------------------------------------------------------------------------- #

def _step(q: dict, name: str) -> list[dict]:
    return [s for s in q.get("steps") or [] if s.get("step") == name]


def _termination(q: dict) -> str:
    return str((q.get("execution") or {}).get("termination") or q.get("termination") or "")


def mask(text: str, limit: int = 140) -> str:
    """A failure message with the parts that change between occurrences masked, so the same
    failure groups together: quoted names, numbers, hosts / IPs, SQL caret lines."""
    t = str(text or "").split("\n")[0]
    t = re.sub(r"\"[^\"]*\"|'[^']*'", "'?'", t)
    t = re.sub(r"\b\d+(?:\.\d+)*\b", "N", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:limit]


def signatures(q: dict) -> list[str]:
    """Failure signatures this question hit (a question can hit several before it recovers)."""
    out = []
    for s in _step(q, "validate"):
        outcome = s.get("outcome")
        if outcome in (None, "ok", "entity_mismatch_kept"):
            continue
        detail = s.get("reason") or "; ".join(map(str, s.get("problems") or [])) or ""
        out.append(f"validate:{outcome}: {mask(detail)}" if detail else f"validate:{outcome}")
    for s in _step(q, "execute"):
        if s.get("error"):
            out.append(f"sql_error: {mask(s['error'])}")
    return out


def root_cause(q: dict, vote: dict | None = None) -> str | None:
    """One cause per question that did not get a good answer; None for a good one.
    Ordered from the most to the least specific signal."""
    term = _termination(q)
    summary = str(q.get("summary") or "").lower()
    if vote and vote.get("rating") == "down":
        return f"downvoted:{vote.get('reason') or 'unspecified'}"
    if term == "LLM_UNAVAILABLE" or (q.get("refused") and "busy" in summary):
        return "llm_unavailable"
    if term == "TIMEOUT_OR_DB_UNAVAILABLE":
        return "db_unavailable"
    if term == "TOOL_EXECUTION_FAILED":
        err = " ".join(str(s.get("error") or "") for s in _step(q, "execute")).lower()
        if "timeout" in err or "timed out" in err or "canceling statement" in err:
            return "sql_timeout"
        if "connection" in err:
            return "db_unavailable"
        return "sql_error"
    if term == "VALIDATION_FAILED":
        reasons = " ".join(str(s.get("reason") or s.get("outcome") or "") for s in _step(q, "validate")).lower()
        if "unknown table" in reasons or "unknown column" in reasons:
            return "hallucinated_schema"
        if "entity" in reasons:
            return "entity_mismatch"
        return "validation_failed"
    if term in ("LOOP_DETECTED", "DUPLICATE_ACTION"):
        return "agent_loop"
    if term == "ABSTAINED_NO_EVIDENCE":
        return "no_evidence"
    if q.get("error"):
        return "error"
    if q.get("refused"):
        return None if q.get("kind") in _EXPECTED_REFUSALS else "refused"
    if q.get("kind") == "data" and q.get("rows") == 0 and q.get("sql"):
        return "empty_result"
    return None


def _norm_question(text: str) -> str:
    t = re.sub(r"[^\w\s%]", " ", str(text or "").lower())
    return re.sub(r"\s+", " ", t).strip()


# --------------------------------------------------------------------------- #
# Model comparison
# --------------------------------------------------------------------------- #

def models(calls: list[dict]) -> list[dict]:
    by: dict[str, list[dict]] = defaultdict(list)
    for c in calls:
        by[str(c.get("model") or "-")].append(c)
    rows = []
    for model, cs in by.items():
        ok = [c for c in cs if str(c.get("status", "")).startswith("ok")]
        statuses = Counter(str(c.get("status")) for c in cs if c not in ok)
        rows.append({
            "model": model, "attempts": len(cs), "success_rate": round(len(ok) / len(cs), 3),
            "p50_ms": _pct([c.get("ms") or 0 for c in ok], 50), "p95_ms": _pct([c.get("ms") or 0 for c in ok], 95),
            "prompt_tokens_mean": round(sum(c.get("prompt_tokens") or 0 for c in ok) / len(ok)) if ok else None,
            "completion_tokens_mean": round(sum(c.get("completion_tokens") or 0 for c in ok) / len(ok)) if ok else None,
            "tasks": dict(Counter(str(c.get("task")) for c in cs).most_common(5)),
            "failures": dict(statuses.most_common(4)),
        })
    return sorted(rows, key=lambda r: -r["attempts"])


# --------------------------------------------------------------------------- #
# Regression cases (self-evolving evaluation)
# --------------------------------------------------------------------------- #

def regression_cases(qs: list[dict], votes: dict[str, dict], limit: int = 60) -> list[dict]:
    """Eval cases from real traffic, deduplicated by question:
       failure  -- a question that failed (expect: it gets answered); newest first
       downvote -- a user said the answer was wrong (expect: the user's reason / comment)
       golden   -- an upvoted answer: its SQL is a reference to compare a future answer against
    Same shape as eval/question_bank/questions.yaml entries plus `source`, `trace_id`, `ref_sql`."""
    seen: set[str] = set()
    out: list[dict] = []
    for q in sorted(qs, key=lambda r: str(r.get("at")), reverse=True):
        text = q.get("question") or ""
        key = _norm_question(q.get("resolved_question") or text)
        if not key or key in seen:
            continue
        vote = votes.get(q.get("trace_id"))
        cause = root_cause(q, vote)
        if vote and vote.get("rating") == "up" and q.get("sql") and not q.get("refused"):
            case = {"source": "golden", "kind": q.get("kind") or "data",
                    "expect": f"same figures as the upvoted answer: {str(q.get('summary') or '')[:200]}",
                    "ref_sql": q.get("sql")}
        elif cause and cause.startswith("downvoted"):
            case = {"source": "downvote", "kind": q.get("kind") or "any",
                    "expect": f"user said {vote.get('reason') or 'wrong'}" + (f": {vote['comment']}" if vote.get("comment") else ""),
                    "bad_sql": q.get("sql")}
        elif cause and cause not in ("llm_unavailable", "db_unavailable"):   # infra outages aren't test cases
            case = {"source": "failure", "kind": q.get("kind") or "any", "expect": f"answered (failed as {cause})"}
        else:
            continue
        seen.add(key)
        out.append({"id": f"auto_{q.get('trace_id')}", "q": text, **{k: v for k, v in case.items() if v},
                    "trace_id": q.get("trace_id"), "seen_at": q.get("at")})
        if len(out) >= limit:
            break
    return out


def write_cases(cases: list[dict], path: str | None = None) -> str:
    """Write the cases as a question-bank YAML (sections by source) for run_bank-style replay."""
    import yaml

    path = path or os.path.join(os.path.dirname(config.OUTPUT_DIR), "eval", "question_bank", "auto_regression.yaml")
    sections: dict[str, list] = defaultdict(list)
    for c in cases:
        sections[f"auto {c['source']}"].append(c)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"# Generated {_now()} by gmr_agents/ops/learning.py from real traffic -- review before trusting.\n")
        yaml.safe_dump({"sections": dict(sections)}, fh, allow_unicode=True, sort_keys=False, width=140)
    return path


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #

def analyse(qs: list[dict], votes: dict[str, dict]) -> dict:
    """Root causes, unanswered questions, recurring failures and feedback for these questions."""
    causes: Counter = Counter()
    unanswered: dict[str, dict] = {}
    sigs: dict[str, dict] = {}
    for q in qs:
        cause = root_cause(q, votes.get(q.get("trace_id")))
        for sig in signatures(q):
            g = sigs.setdefault(sig, {"signature": sig, "count": 0, "recovered": 0, "last_seen": "", "examples": []})
            g["count"] += 1
            g["recovered"] += cause is None
            g["last_seen"] = max(g["last_seen"], str(q.get("at")))
            if len(g["examples"]) < 3 and q.get("question") not in g["examples"]:
                g["examples"].append(q.get("question"))
        if not cause:
            continue
        causes[cause] += 1
        if cause in ("llm_unavailable", "db_unavailable"):
            continue  # outages, not questions the bot can't handle
        key = _norm_question(q.get("resolved_question") or q.get("question"))
        u = unanswered.setdefault(key, {"question": q.get("question"), "count": 0, "causes": Counter(),
                                        "last_seen": "", "last_trace_id": None, "topics": q.get("topics")})
        u["count"] += 1
        u["causes"][cause] += 1
        if str(q.get("at")) >= u["last_seen"]:
            u["last_seen"], u["last_trace_id"] = str(q.get("at")), q.get("trace_id")
    n = len(qs)
    by_trace = {q.get("trace_id"): q for q in qs}
    in_window = [v for t, v in votes.items() if t in by_trace]
    up = sum(1 for v in in_window if v["rating"] == "up")
    down = [v for v in in_window if v["rating"] == "down"]
    return {
        "questions": n,
        "good_answer_rate": round(1 - sum(causes.values()) / n, 4) if n else None,
        "root_causes": [{"cause": c, "count": k, "share": round(k / n, 4)} for c, k in causes.most_common()],
        "unanswered": [{**u, "causes": dict(u["causes"])} for u in
                       sorted(unanswered.values(), key=lambda u: (-u["count"], u["last_seen"]))[:25]],
        "recurring_failures": sorted((g for g in sigs.values() if g["count"] >= 2),
                                     key=lambda g: -g["count"])[:25],
        "feedback": {
            "votes": len(in_window), "up": up, "down": len(down),
            "satisfaction": round(up / len(in_window), 3) if in_window else None,
            "reasons": dict(Counter(v.get("reason") or "unspecified" for v in down).most_common()),
            "latest_downvotes": [{"at": v["at"], "trace_id": v["trace_id"], "reason": v.get("reason"),
                                  "comment": v.get("comment"), "question": by_trace[v["trace_id"]].get("question"),
                                  "summary": str(by_trace[v["trace_id"]].get("summary") or "")[:300],
                                  "sql": by_trace[v["trace_id"]].get("sql")}
                                 for v in sorted(down, key=lambda v: v["at"], reverse=True)[:15]],
        },
    }


def compute(hours: float = 168.0) -> dict:
    since = _since(hours)
    qs = [q for q in _read("questions", since) if q.get("session_id")]
    traces = {q.get("trace_id") for q in qs}
    calls = [c for c in _read("llm_calls", since) if c.get("trace_id") in traces]
    votes = latest_votes()
    report = analyse(qs, votes)
    return {"window": {"hours": hours, "since": since}, **report, "models": models(calls),
            "regression_cases": regression_cases(qs, votes)}


if __name__ == "__main__":
    import argparse
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hours", type=float, default=168.0)
    ap.add_argument("--write-cases", action="store_true", help="write eval/question_bank/auto_regression.yaml")
    a = ap.parse_args()
    out = compute(a.hours)
    if a.write_cases:
        print("wrote", write_cases(out["regression_cases"]), file=sys.stderr)
    print(json.dumps(out, indent=1, default=str))
