"""Proactive self-monitoring: alerts and a daily digest, computed by code from the logs.

    GET /api/alerts?hours=24                    current alerts, most severe first
    GET /api/insights/digest?hours=24&format=md the daily digest (JSON, or Markdown to paste / mail)
    python -m gmr_agents.ops.alerts --hours 24 [--md]

Each check compares the last `hours` with the BASELINE_DAYS before it, so an alert means
"worse than usual", not "worse than perfect":

    quality_regression   share of questions without a good answer jumped
    cause_spike          one root cause (ops/learning.py) at least doubled its share
    new_failure          a failure signature seen >= 3 times now and never in the baseline
    latency_regression   P95 latency 30% above the baseline
    slo_breach           an SLO_* target from .env is missed (ops/metrics.py)
    model_unhealthy      a model succeeds on fewer than 70% of its attempts
    low_satisfaction     users downvote more than 30% of the answers they rate
    schema_change        a source table / column changed, emptied or shrank (ops/schema_watch.py)
    stale_data           a source table is behind (checks/freshness.py, from the data skeleton)

No LLM call and no database query: the data-freshness check reads the data skeleton the
background job already built. Nothing here is on the question path.
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from datetime import datetime, timedelta, timezone

from shared import config

log = logging.getLogger(__name__)
_IST = timezone(timedelta(hours=5, minutes=30))
BASELINE_DAYS = 7
MIN_QUESTIONS = 20          # fewer questions than this in a window: rates are noise, no rate alert
_ORDER = {"high": 0, "medium": 1, "low": 2}


def _ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def _windows(hours: float) -> tuple[list[dict], list[dict], str]:
    from gmr_agents.ops.metrics import _read

    now = datetime.now(_IST)
    cur_since = _ts(now - timedelta(hours=hours))
    base_since = _ts(now - timedelta(hours=hours) - timedelta(days=BASELINE_DAYS))
    rows = [q for q in _read("questions", base_since) if q.get("session_id")]
    return ([q for q in rows if str(q.get("at")) >= cur_since],
            [q for q in rows if str(q.get("at")) < cur_since], cur_since)


def _alert(kind: str, severity: str, title: str, detail: str, **evidence) -> dict:
    return {"kind": kind, "severity": severity, "title": title, "detail": detail, "evidence": evidence}


def _quality_alerts(cur: dict, base: dict, base_sigs: set[str]) -> list[dict]:
    out = []
    n_cur, n_base = cur["questions"], base["questions"]
    if n_cur < MIN_QUESTIONS or n_base < MIN_QUESTIONS:
        return out
    fail_cur = 1 - (cur["good_answer_rate"] or 0)
    fail_base = 1 - (base["good_answer_rate"] or 0)
    if fail_cur > fail_base + 0.05 and fail_cur > fail_base * 1.5:
        out.append(_alert("quality_regression", "high" if fail_cur > fail_base + 0.15 else "medium",
                          f"{fail_cur:.0%} of questions did not get a good answer (usual {fail_base:.0%})",
                          "See root_causes for what changed.", now=round(fail_cur, 4), baseline=round(fail_base, 4),
                          questions=n_cur))
    base_share = {r["cause"]: r["share"] for r in base["root_causes"]}
    for r in cur["root_causes"]:
        b = base_share.get(r["cause"], 0.0)
        if r["count"] >= 5 and r["share"] >= b * 2 and r["share"] - b >= 0.03:
            out.append(_alert("cause_spike", "medium", f"'{r['cause']}' up to {r['share']:.1%} of questions (usual {b:.1%})",
                              f"{r['count']} questions in this window.", cause=r["cause"], now=r["share"], baseline=b))
    known = base_sigs
    new = [g for g in cur["recurring_failures"]
           if g["count"] >= 3 and g["signature"] not in known and g["recovered"] < g["count"]]
    for g in new[:5]:   # the rest are in /api/insights/quality
        out.append(_alert("new_failure", "high" if g["recovered"] == 0 else "medium",
                          f"New recurring failure ({g['count']}x, {g['recovered']} recovered)", g["signature"],
                          examples=g["examples"]))
    return out


def _latency_alert(cur_qs: list[dict], base_qs: list[dict]) -> list[dict]:
    from gmr_agents.ops.metrics import _pct

    live_c = [q.get("total_ms") or 0 for q in cur_qs if not q.get("cached")]
    live_b = [q.get("total_ms") or 0 for q in base_qs if not q.get("cached")]
    if len(live_c) < MIN_QUESTIONS or len(live_b) < MIN_QUESTIONS:
        return []
    c, b = _pct(live_c, 95), _pct(live_b, 95)
    if c and b and c > b * 1.3:
        return [_alert("latency_regression", "medium", f"P95 latency {c / 1000:.1f}s (usual {b / 1000:.1f}s)",
                       "Uncached questions only.", now_ms=c, baseline_ms=b)]
    return []


def _slo_alerts(hours: float) -> list[dict]:
    from gmr_agents.ops import metrics

    m = metrics.compute(hours)
    if (m["window"]["questions"] or 0) < MIN_QUESTIONS:
        return []
    return [_alert("slo_breach", "medium", f"SLO missed: {path} = {t['value']} (target {t['target']})", "From .env SLO_* targets.",
                   metric=path, value=t["value"], target=t["target"])
            for path, t in m["targets"].items() if t["met"] is False]


def _model_alerts(models: list[dict]) -> list[dict]:
    return [_alert("model_unhealthy", "low" if m["success_rate"] >= 0.5 else "medium",
                   f"{m['model']} succeeded on {m['success_rate']:.0%} of {m['attempts']} attempts",
                   "Fallback models answered instead; each failed attempt still costs time.", failures=m["failures"])
            for m in models if m["attempts"] >= 10 and m["success_rate"] < 0.7]


def _feedback_alerts(fb: dict) -> list[dict]:
    if fb["votes"] >= 5 and fb["satisfaction"] is not None and fb["satisfaction"] < 0.7:
        return [_alert("low_satisfaction", "high" if fb["satisfaction"] < 0.5 else "medium",
                       f"Users downvoted {fb['down']} of {fb['votes']} rated answers",
                       "See feedback.latest_downvotes.", reasons=fb["reasons"])]
    return []


def _schema_alerts(hours: float) -> list[dict]:
    from gmr_agents.ops import schema_watch

    out = []
    for ch in schema_watch.recent(hours):
        severe = any(k in ch for k in ("tables_removed", "columns_removed", "emptied"))
        keys = [k for k in ch if k not in ("at", "domain")]
        out.append(_alert("schema_change", "high" if severe else "medium" if any(k in ch for k in ("shrunk", "type_changes")) else "low",
                          f"{ch.get('domain', '').upper()} source changed: {', '.join(keys)}", f"Detected {ch.get('at')}.",
                          **{k: ch[k] for k in keys}))
    return out


def stale_sources() -> list[dict]:
    """Every source table's data-up-to date, from the data skeleton (no query)."""
    from gmr_agents.checks import freshness
    from gmr_agents.retrieval import data_skeleton

    skeleton = data_skeleton.load() or {}
    names = [c.get("table") for c in skeleton.get("tables") or [] if c.get("table")]
    return [{**i, "label": freshness.label(i)} for i in freshness.check(names, None, skeleton)]


def _stale_alerts() -> list[dict]:
    try:
        stale = [s for s in stale_sources() if s["stale"]]
    except Exception:
        log.debug("freshness scan failed", exc_info=True)
        return []
    if not stale:
        return []
    by: dict[str, list[str]] = {}
    for s in stale:
        by.setdefault(f"{s['label']} ({s['behind']} behind)", []).append(s["table"])
    return [_alert("stale_data", "medium", f"{len(stale)} source table(s) behind",
                   "Answers from these tables already carry a 'data up to' warning.", groups=by)]


def compute(hours: float = 24.0) -> dict:
    from gmr_agents.ops import learning
    from gmr_agents.ops.metrics import _read

    cur_qs, base_qs, cur_since = _windows(hours)
    votes = learning.latest_votes()
    cur, base = learning.analyse(cur_qs, votes), learning.analyse(base_qs, votes)
    base_sigs = {s for q in base_qs for s in learning.signatures(q)}
    traces = {q.get("trace_id") for q in cur_qs}
    models = learning.models([c for c in _read("llm_calls", cur_since) if c.get("trace_id") in traces])
    alerts: list[dict] = []
    for check in (lambda: _quality_alerts(cur, base, base_sigs), lambda: _latency_alert(cur_qs, base_qs),
                  lambda: _slo_alerts(hours), lambda: _model_alerts(models),
                  lambda: _feedback_alerts(cur["feedback"]), lambda: _schema_alerts(hours), _stale_alerts):
        try:
            alerts.extend(check())
        except Exception:       # one broken check must not hide the others
            log.exception("alert check failed")
    alerts.sort(key=lambda a: _ORDER.get(a["severity"], 9))
    return {"at": _ts(datetime.now(_IST)), "window_hours": hours, "baseline_days": BASELINE_DAYS,
            "alerts": alerts, "_cur": cur, "_models": models, "_cur_qs": cur_qs}


# --------------------------------------------------------------------------- #
# Daily digest
# --------------------------------------------------------------------------- #

def digest(hours: float = 24.0) -> dict:
    from gmr_agents.ops import metrics

    a = compute(hours)
    cur, qs = a.pop("_cur"), a.pop("_cur_qs")
    models = a.pop("_models")
    m = metrics.compute(hours)
    topics = Counter(t for q in qs for t in (q.get("topics") or [])[:1])
    asked = Counter(learning_norm(q.get("resolved_question") or q.get("question")) for q in qs)
    examples = {learning_norm(q.get("resolved_question") or q.get("question")): q.get("question") for q in qs}
    return {
        **a,
        "headline": {"questions": len(qs), "good_answer_rate": cur["good_answer_rate"],
                     "p50_ms": m["latency_ms"]["p50"], "p95_ms": m["latency_ms"]["p95"],
                     "cache_hit_rate": m["cache"]["answer_cache_hit_rate"],
                     "tokens_mean": m["tokens_per_question"]["total_mean"],
                     "satisfaction": cur["feedback"]["satisfaction"], "votes": cur["feedback"]["votes"]},
        "most_asked_topics": [{"topic": t, "questions": n} for t, n in topics.most_common(10)],
        "most_asked_questions": [{"question": examples[k], "count": n} for k, n in asked.most_common(10) if k],
        "root_causes": cur["root_causes"][:8],
        "unanswered": cur["unanswered"][:10],
        "models": models[:6],
    }


def learning_norm(text) -> str:
    from gmr_agents.ops.learning import _norm_question

    return _norm_question(text)


def _fmt_rate(v) -> str:
    return "-" if v is None else f"{v:.0%}"


def to_markdown(d: dict) -> str:
    h = d["headline"]
    lines = [f"# GMR assistant digest -- last {d['window_hours']:g}h (to {d['at']})", "",
             f"- Questions: **{h['questions']}**, good answers **{_fmt_rate(h['good_answer_rate'])}**, "
             f"cache hits {_fmt_rate(h['cache_hit_rate'])}",
             f"- Latency P50 {(h['p50_ms'] or 0) / 1000:.1f}s, P95 {(h['p95_ms'] or 0) / 1000:.1f}s; "
             f"tokens/question {h['tokens_mean'] or '-'}",
             f"- User satisfaction {_fmt_rate(h['satisfaction'])} ({h['votes']} rated)", "", "## Alerts", ""]
    lines += [f"- **[{x['severity']}]** {x['title']} -- {x['detail']}" for x in d["alerts"]] or ["- None."]
    lines += ["", "## Most asked topics", ""]
    lines += [f"- {t['topic']}: {t['questions']}" for t in d["most_asked_topics"]] or ["- None."]
    lines += ["", "## Why answers failed", ""]
    lines += [f"- {r['cause']}: {r['count']} ({r['share']:.1%})" for r in d["root_causes"]] or ["- None."]
    lines += ["", "## Questions that keep failing", ""]
    lines += [f"- ({u['count']}x) {u['question']} -- {', '.join(u['causes'])}" for u in d["unanswered"]] or ["- None."]
    lines += ["", "## Models", "", "| model | attempts | success | P50 s |", "|---|---:|---:|---:|"]
    lines += [f"| {x['model']} | {x['attempts']} | {x['success_rate']:.0%} | {(x['p50_ms'] or 0) / 1000:.1f} |" for x in d["models"]]
    return "\n".join(lines) + "\n"


def write_daily_digest() -> str | None:
    """Write outputs/digests/digest_<date>.md once a day (background task in main.py)."""
    folder = os.path.join(config.OUTPUT_DIR, "digests")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"digest_{datetime.now(_IST):%Y-%m-%d}.md")
    if os.path.exists(path):
        return None
    d = digest(24.0)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(to_markdown(d))
    for x in d["alerts"]:
        if x["severity"] == "high":
            log.warning("ALERT: %s -- %s", x["title"], x["detail"])
    return path


if __name__ == "__main__":
    import argparse
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--md", action="store_true")
    args = ap.parse_args()
    d = digest(args.hours)
    print(to_markdown(d) if args.md else json.dumps(d, indent=1, default=str))
