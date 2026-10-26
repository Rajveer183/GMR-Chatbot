"""Cache pre-warmer: keeps common questions instant.

Whenever the answer cache's version changes -- a gold data refresh, an edit to
the training files (business_rules.yaml / examples.yaml), or a new day (IST) --
the bank questions (PREWARM_QUESTIONS_FILE) and the most-asked questions from
the history log (outputs/logs/questions.jsonl) are answered once through the
normal pipeline, so the next person to ask any of them gets the stored LLM
answer in well under a second.

It runs one question at a time with a pause (PREWARM_PAUSE) so real users keep
most of NVIDIA's free-tier request rate. Bank chains (multi-turn) run in order
in their own session, so their follow-ups are cached with the right context.
Warm-up sessions go to a separate store, not the user's chat list.

Free-tier budget. One warmed answer costs 3-5 LLM calls, and warming the whole
bank (~470 questions) after every YAML edit used ~2,000 calls a run -- every
Groq / Gemini daily quota, leaving real users "all AI models are busy"
(measured 23 Oct: 971 of 1,093 questions that day were warm-ups). So: the bank
is warmed only with PREWARM_BANK=1, a run answers at most PREWARM_MAX_QUESTIONS,
a new version must stay unchanged PREWARM_SETTLE_SECONDS (a burst of edits warms
once), and a run stops at the first "AI busy" answer -- it resumes on the next
check, skipping what is already cached.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import Counter

import yaml

from shared import config
from shared.session_store import SessionStore
from gmr_agents import answer_cache

log = logging.getLogger(__name__)

_STORE_PATH = os.path.join(config.OUTPUT_DIR, "prewarm_sessions.json")
_STATE = {"warmed_version": None, "running": False, "last": {}, "seen_version": None, "seen_at": 0.0}


def _bank_groups() -> list[list[str]]:
    """Bank questions as groups: a chain is one group (asked in order), every
    other question its own group."""
    path = config.PREWARM_QUESTIONS_FILE
    if not os.path.isabs(path):
        path = os.path.join(config.PROJECT_ROOT, path)
    try:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except OSError:
        log.warning("prewarm: question file %s not found", path)
        return []
    groups: list[list[str]] = []
    chains: dict[str, list[str]] = {}
    for items in (data.get("sections") or {}).values():
        for item in items or []:
            q = str(item.get("q") or "").strip()
            if not q:
                continue
            if item.get("chain"):
                if item["chain"] not in chains:
                    chains[item["chain"]] = []
                    groups.append(chains[item["chain"]])
                chains[item["chain"]].append(q)
            else:
                groups.append([q])
    return groups


def _top_history(n: int) -> list[str]:
    """The n most-asked standalone questions that were answered (not refused)."""
    path = os.path.join(config.LOG_DIR, "questions.jsonl")
    counts: Counter = Counter()
    first: dict[str, str] = {}
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                q = str(r.get("question") or "").strip()
                resolved = r.get("resolved_question")
                if not q or r.get("refused") or r.get("error") or (resolved and resolved != q):
                    continue  # skip refusals and rewritten follow-ups (they need their conversation)
                key = re.sub(r"\s+", " ", q.lower())
                counts[key] += 1
                first.setdefault(key, q)
    except OSError:
        return []
    return [first[k] for k, _ in counts.most_common(n)]


def _busy(resp: dict) -> bool:
    return "AI busy" in str(resp.get("source") or (resp.get("debug") or {}).get("source") or "") or \
        "busy right now" in str(resp.get("summary") or "")


def warm(answer_fn) -> dict:
    """Answer the warm-up questions once (skipping ones already cached for this
    version). answer_fn(question, session_id, store=...) -> response. Returns the
    run's stats; stats["complete"] is False when it stopped because the models were
    busy, so the caller warms the same version again on its next check."""
    if _STATE["running"]:
        return _STATE["last"]
    _STATE["running"] = True
    t0 = time.perf_counter()
    stats = {"asked": 0, "already_cached": 0, "failed": 0, "complete": True}
    try:
        if os.path.exists(_STORE_PATH):
            os.remove(_STORE_PATH)  # warm-up conversations are throwaway
        store = SessionStore(_STORE_PATH)
        groups = _bank_groups() if config.PREWARM_BANK else []
        bank = {q.lower() for g in groups for q in g}
        # Most-asked first: they are what users actually repeat.
        groups = [[q] for q in _top_history(config.PREWARM_TOP_N) if q.lower() not in bank] + groups
        stop = None
        for group in groups:
            if stop:
                break
            session_id = None
            for q in group:
                if session_id is None and answer_cache.get(q, None) is not None:
                    stats["already_cached"] += 1
                    if len(group) == 1:
                        continue
                if config.PREWARM_MAX_QUESTIONS and stats["asked"] >= config.PREWARM_MAX_QUESTIONS:
                    stop = f"run cap of {config.PREWARM_MAX_QUESTIONS} questions reached"
                    break
                try:
                    resp = answer_fn(q, session_id, store=store)
                    session_id = resp.get("session_id")
                    stats["asked"] += 0 if (resp.get("debug") or {}).get("cached") else 1
                    stats["failed"] += bool(resp.get("refused"))
                    if _busy(resp):
                        stop = "AI models busy -- leaving the free-tier quota to users"
                        break
                except Exception:
                    stats["failed"] += 1
                    log.warning("prewarm: %r failed", q, exc_info=True)
                time.sleep(config.PREWARM_PAUSE)
        if stop:
            # Busy: resume later. The cap is this version's whole allowance: done.
            stats["complete"] = not stop.startswith("AI models busy")
            log.info("prewarm: stopped early -- %s", stop)
    finally:
        _STATE["running"] = False
    stats["seconds"] = round(time.perf_counter() - t0)
    _STATE["last"] = stats
    log.info("prewarm: done -- %s", stats)
    return stats


def due() -> str | None:
    """The cache version to warm for, when it differs from the last warmed one and
    has stayed unchanged PREWARM_SETTLE_SECONDS (editing the YAML several times in a
    row warms once, for the final version)."""
    version = answer_cache.current_version()
    if not version or version == _STATE["warmed_version"] or _STATE["running"]:
        return None
    now = time.monotonic()
    if version != _STATE["seen_version"]:
        _STATE.update(seen_version=version, seen_at=now)
    if now - _STATE["seen_at"] < config.PREWARM_SETTLE_SECONDS:
        return None
    return version


def mark_warmed(version: str) -> None:
    _STATE["warmed_version"] = version


def status() -> dict:
    return {"enabled": config.PREWARM_ENABLED, "running": _STATE["running"],
            "warmed_version": _STATE["warmed_version"], "last_run": _STATE["last"]}
