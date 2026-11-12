"""Investigation memory and decision-impact tracking. No LLM call.

Every answered investigation (presentation/investigate.py) is kept in
outputs/logs/investigations.jsonl: the question, the lead it found, its hypotheses and
confidence, the recommended action, and the SQL that produced it. That gives:

    related(question)   past investigations of a similar question (cross-session episodic
                        memory) -- attached to a new investigation's response, no prompt tokens
    recall(question)    'have we investigated X before?' answered from the store (no LLM, no DB)
    recheck(id)         'has it improved since?': the stored SQL re-run read-only and the lead
                        compared with what it was when the investigation ran (one DB query)

Relative-date SQL ('last month vs the month before') re-runs on today's periods, so a recheck
compares the latest complete period now with the latest one at investigation time.
"""

from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timedelta, timezone

from shared import config

_IST = timezone(timedelta(hours=5, minutes=30))
_LOCK = threading.Lock()
_STOP = set("""a an the of in on at to for by and or is are was were be been did do does has have had why what how which
who when where it its this that these those we our you your me my i show tell give find out vs versus than then from with
over between last month week year months weeks previous before compared since about any all per there their them""".split())

RECALL_RE = re.compile(
    r"\b(?:(?:have|did|had) we (?:ever |already )?(?:investigat|look(?:ed)? into|analy[sz]|root[- ]?caus|diagnos)\w*"
    r"|(?:previous|past|earlier|last|prior|old) investigations?|what did we find|investigation history"
    r"|(?:improv\w*|better|worse|changed|recovered) since (?:we|the|our|that|last) (?:investigat\w*|investigation|recommend\w*|looked)"
    r"|(?:did|has|have) (?:the|our|that|those) (?:action|recommendation|fix)\w* (?:work|help|improv\w*))", re.I)
_RECHECK_RE = re.compile(r"\b(?:improv\w*|better|worse|changed|recovered|work(?:ed)?|help(?:ed)?|since|now|still)\b", re.I)


def _path() -> str:
    from shared.logging_config import _log_dir

    return os.path.join(_log_dir(), "investigations.jsonl")


def _words(text: str) -> set[str]:
    words = re.findall(r"[a-z]{3,}", (text or "").lower())
    out = set()
    for w in words:
        if w in _STOP:
            continue
        out.add(_stem(w))
    return out


def _stem(w: str) -> str:
    s = re.sub(r"(?:ations?|ing|ed|es|s)$", "", w) or w
    if len(s) > 3 and s[-1] == s[-2] and s[-1] not in "lsz":   # dropp -> drop
        s = s[:-1]
    return s


def _sim(a: str, b: str) -> float:
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def all_records() -> list[dict]:
    try:
        with open(_path(), encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]
    except (OSError, ValueError):
        return []


def record(*, trace_id: str, session_id: str | None, question: str, investigation: dict, plan: dict,
           summary: str, insights: list[str]) -> dict | None:
    """Store one investigation. Never raises."""
    try:
        lead = investigation.get("lead")
        main = (investigation.get("dimensions") or [{}])[0]
        rec = {
            "id": trace_id, "at": datetime.now(_IST).strftime("%Y-%m-%dT%H:%M:%S"), "session_id": session_id,
            "question": question, "measure": investigation.get("measure"), "periods": investigation.get("periods"),
            "by": main.get("by"), "total_prev": main.get("total_prev"), "total_curr": main.get("total_curr"),
            "lead": lead, "confidence": investigation.get("confidence"),
            "hypotheses": [{"by": d.get("by"), **h} for d in investigation.get("dimensions") or [] for h in d.get("hypotheses") or []],
            "contradictions": investigation.get("contradictions") or [],
            "summary": summary[:600], "action": _action(insights)[:300],
            "sql": plan.get("sql"), "engine": plan.get("data_engine"), "database": plan.get("database"),
        }
        with _LOCK, open(_path(), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str, ensure_ascii=False) + "\n")
        return rec
    except Exception:
        return None


def related(question: str, *, exclude: str | None = None, limit: int = 3, min_sim: float = 0.34) -> list[dict]:
    """Past investigations of a similar question, most similar (then newest) first."""
    scored = []
    for r in all_records():
        if r.get("id") == exclude:
            continue
        s = _sim(question, r.get("question") or "")
        if s >= min_sim:
            scored.append((s, r.get("at") or "", r))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    seen, out = set(), []
    for s, _, r in scored:
        key = re.sub(r"\W+", " ", (r.get("question") or "").lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        out.append({"id": r.get("id"), "at": r.get("at"), "question": r.get("question"), "similarity": round(s, 2),
                    "lead": r.get("lead"), "confidence": (r.get("confidence") or {}).get("level"),
                    "action": r.get("action"), "periods": r.get("periods")})
        if len(out) >= limit:
            break
    return out


_ACTION_RE = re.compile(r"^\s*(?:\W*\s*)?(?:review|check|audit|investigate|deploy|redeploy|prioriti[sz]e|focus|replicate|"
                        r"escalate|assign|add|increase|reduce|schedule|target|clear|fix|monitor|verify|reallocate|"
                        r"move|shift|plan|consider|ensure|follow up|track|conduct|launch|expedite|accelerate|address)\b", re.I)


def _action(insights: list[str] | None) -> str:
    """The recommended action among the answer's insights (imperative), '' when none is one -- 11 Nov: the last
    insight 'Feeder meter installations increased by 13 units' was stored as the recommendation."""
    return next((i for i in reversed(insights or []) if _ACTION_RE.search(str(i))), "")


def is_recall(question: str) -> bool:
    return bool(RECALL_RE.search(question or ""))


def _fmt(v) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    return f"{v:,.0f}" if v.is_integer() else f"{v:,.2f}"


def _lead_text(r: dict) -> str:
    lead = r.get("lead") or {}
    if not lead:
        return "no single place or category stood out"
    share = f", {lead['share']:.0f}% of the change" if lead.get("share") is not None else ""
    return f"{lead.get('by')} = {lead.get('name')} ({_fmt(lead.get('prev'))} -> {_fmt(lead.get('curr'))}{share})"


def recheck(inv_id: str) -> dict:
    """Re-run the stored SQL (read-only) and compare the lead and the total now with then."""
    rec = next((r for r in reversed(all_records()) if r.get("id") == inv_id), None)
    if rec is None:
        return {"ok": False, "error": "investigation not found"}
    if not rec.get("sql") or not rec.get("engine"):
        return {"ok": False, "error": "no stored query to re-run", "record": rec}
    from gmr_agents.execution.executor import run_sql
    from gmr_agents.presentation import drivers

    try:
        df = run_sql(rec["sql"], rec.get("database"), rec["engine"])
    except Exception as exc:
        return {"ok": False, "error": f"re-run failed: {str(exc)[:200]}", "record": rec}
    got = drivers.pair(rec.get("question") or "why", df)
    if not got:
        return {"ok": False, "error": "the re-run result has no two periods to compare", "record": rec}
    both, measure, group, p0, p1, _ = got
    now = {"periods": [p0, p1], "total_prev": float(both["prev"].fillna(0).sum()),
           "total_curr": float(both["curr"].fillna(0).sum())}
    lead = rec.get("lead") or {}
    if lead.get("name") is not None and str(lead["name"]) in both.index.astype(str):
        row = both.loc[str(lead["name"])]
        now["lead_curr"] = None if row["curr"] != row["curr"] else float(row["curr"])
    then_periods = rec.get("periods") or []
    same_window = then_periods == [p0, p1]
    out = {"ok": True, "record": rec, "now": now, "same_window": same_window}
    lines = []
    if same_window:
        lines.append(f"The latest complete period is still {p1}: nothing new to compare yet "
                     f"({measure} {_fmt(now['total_curr'])}, was {_fmt(rec.get('total_curr'))} when investigated).")
    else:
        then = rec.get("total_curr")
        delta = (now["total_curr"] - then) if then is not None else None
        lines.append(f"{measure}: {_fmt(then)} in {then_periods[-1] if then_periods else 'the investigated period'} -> "
                     f"{_fmt(now['total_curr'])} in {p1}" + (f" ({delta:+,.0f})" if delta is not None else "") + ".")
        if lead.get("name") is not None and "lead_curr" in now:
            lines.append(f"Lead {lead.get('by')} {lead['name']}: {_fmt(lead.get('curr'))} -> {_fmt(now['lead_curr'])}.")
    out["summary"] = " ".join(lines)
    return out


def recall(question: str) -> dict:
    """Answer 'have we investigated X before / has it improved since' from the store."""
    recs = related(question, min_sim=0.2, limit=3)
    if not recs:
        n = len(all_records())
        return {"summary": ("No earlier investigation matches this question" + (f" ({n} investigations stored)." if n else
                            " -- none has been recorded yet.") + " Ask a 'why did ... change' question to start one."),
                "insights": [], "tables": [], "found": []}
    top = recs[0]
    parts = [f"Yes -- on {top['at'][:10]} we investigated \"{top['question']}\"."]
    full = next((r for r in reversed(all_records()) if r.get("id") == top["id"]), {})
    parts.append(f"Lead: {_lead_text(full)}; confidence {top.get('confidence') or 'n/a'}.")
    if top.get("action"):
        parts.append(f"Recommended: {top['action']}")
    insights = []
    if _RECHECK_RE.search(question or "") and not re.search(r"\bwhat did we find\b", question or "", re.I):
        chk = recheck(top["id"])
        if chk.get("ok"):
            parts.append("Now: " + chk["summary"])
        else:
            insights.append(f"Could not re-check it: {chk.get('error')}")
    for r in recs[1:]:
        insights.append(f"Also on {r['at'][:10]}: \"{r['question']}\" (confidence {r.get('confidence') or 'n/a'})")
    store = {x.get("id"): x for x in all_records()}
    cols = ["When", "Question", "Lead found", "Confidence"]
    rows = [dict(zip(cols, [r["at"][:16].replace("T", " "), r["question"], _lead_text(store.get(r["id"], {})),
                            r.get("confidence") or ""])) for r in recs]
    table = {"id": "investigations", "title": "Earlier investigations", "heading": None, "columns": cols,
             "rows": rows, "page": 1, "page_size": len(rows), "total_rows": len(rows), "has_next": False,
             "has_previous": False}
    return {"summary": " ".join(parts), "insights": insights, "tables": [table], "found": recs}


def recent(limit: int = 50) -> list[dict]:
    keep = ("id", "at", "session_id", "question", "measure", "periods", "by", "total_prev", "total_curr", "lead",
            "confidence", "contradictions", "action")
    return [{k: r.get(k) for k in keep} for r in reversed(all_records()[-limit:])]
