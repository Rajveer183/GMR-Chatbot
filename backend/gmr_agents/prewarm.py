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
_STATE = {"warmed_version": None, "running": False, "last": {}}


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


def warm(answer_fn) -> dict:
    """Answer every warm-up question once (skipping ones already cached for
    this version). answer_fn(question, session_id, store=...) -> response."""
    if _STATE["running"]:
        return _STATE["last"]
    _STATE["running"] = True
    t0 = time.perf_counter()
    stats = {"asked": 0, "already_cached": 0, "failed": 0}
    try:
        if os.path.exists(_STORE_PATH):
            os.remove(_STORE_PATH)  # warm-up conversations are throwaway
        store = SessionStore(_STORE_PATH)
        groups = _bank_groups()
        bank = {q.lower() for g in groups for q in g}
        groups += [[q] for q in _top_history(config.PREWARM_TOP_N) if q.lower() not in bank]
        for group in groups:
            session_id = None
            for q in group:
                if session_id is None and answer_cache.get(q, None) is not None:
                    stats["already_cached"] += 1
                    if len(group) == 1:
                        continue
                try:
                    resp = answer_fn(q, session_id, store=store)
                    session_id = resp.get("session_id")
                    stats["asked"] += 0 if (resp.get("debug") or {}).get("cached") else 1
                    stats["failed"] += bool(resp.get("refused"))
                except Exception:
                    stats["failed"] += 1
                    log.warning("prewarm: %r failed", q, exc_info=True)
                time.sleep(config.PREWARM_PAUSE)
    finally:
        _STATE["running"] = False
    stats["seconds"] = round(time.perf_counter() - t0)
    _STATE["last"] = stats
    log.info("prewarm: done -- %s", stats)
    return stats


def due() -> str | None:
    """The cache version to warm for, when it differs from the last warmed one."""
    version = answer_cache.current_version()
    if version and version != _STATE["warmed_version"] and not _STATE["running"]:
        return version
    return None


def mark_warmed(version: str) -> None:
    _STATE["warmed_version"] = version


def status() -> dict:
    return {"enabled": config.PREWARM_ENABLED, "running": _STATE["running"],
            "warmed_version": _STATE["warmed_version"], "last_run": _STATE["last"]}
