"""Answer cache: a question asked again returns the stored LLM answer instantly,
consuming 0 further LLM tokens -- checked in main.answer_question() BEFORE the
LangGraph pipeline runs at all.

Every cached answer was written by the LLM pipeline; the cache only replays it
while nothing it depends on has changed. An entry is valid only for the same:
  * data version   -- gold_dashboard_stats_snapshot.refreshed_at (gold refresh)
  * training       -- content hash of business_rules / examples / charts / table_metadata /
                      kpi_definitions YAML, .env and the answer-writing code (llm_pipeline.py,
                      answer_cache.py, every validation/*.py check, graph/nodes.py)
  * date (IST)     -- 'yesterday', 'last 30 days' ... move every day
Keys: a standalone question (triage said it is not a follow-up) is stored under
its own normalised text; a follow-up is stored under (previous question, text),
so "Which OEM?" never replays an answer from another conversation. This is
exact-match caching on purpose, not fuzzy/semantic similarity: a near-identical
question can have a different answer (a different filter, a different period),
and the existing mid-conversation guard below (_answered_as_asked) exists
specifically to stop a false-positive replay -- loosening the match would
reopen exactly that risk. Refusals, busy and error answers are never stored.

Storage: Redis (shared/redis_client.py) when available -- key = sha1 of the
cache key above, value = {version, response}, TTL = REDIS_RESPONSE_CACHE_TTL on
top of the version check (defense in depth: even if something slipped past the
version check, Redis would still expire the entry within the hour) -- AND the
original local JSON file, written on every put() regardless of Redis, so a
Redis outage never loses what was already cached: reads try Redis first, then
fall back to the file transparently. Settings live in .env.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import re
import threading
import time
from datetime import datetime, timedelta, timezone

from shared import config, redis_client
from gmr_agents.schema import business_rules

log = logging.getLogger(__name__)

_PATH = os.path.join(config.OUTPUT_DIR, "answer_cache.json")
_IST = timezone(timedelta(hours=5, minutes=30))
_LOCK = threading.Lock()
_STATE: dict = {"entries": None, "version": None, "version_at": 0.0}


def _norm(text: str | None) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w%&\s]", " ", str(text or "").lower())).strip()


def _data_refreshed_at() -> str:
    from gmr_agents.sql.connections import pooled_postgres_connection

    with pooled_postgres_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT refreshed_at FROM public.gold_dashboard_stats_snapshot WHERE singleton")
        row = cur.fetchone()
    return str(row[0]) if row else ""


def _hash(paths: list[str], digest=None):
    digest = digest or hashlib.sha1()
    for p in paths:
        if os.path.exists(p):
            with open(p, "rb") as fh:
                digest.update(fh.read())
    return digest


def _code_digest():
    """The answer-writing / checking code, hashed ONCE at import -- the code this process
    actually runs. Hashing it from disk on every check let a server still running the old
    code (no --reload) write answers under the NEW code's version after an edit: verified
    live, the pre-fix 92.03% Kashi utilization was re-cached and survived the restart."""
    here = os.path.dirname(os.path.abspath(__file__))
    sources = [os.path.join(here, "llm_pipeline.py"), os.path.join(here, "answer_cache.py"),
               os.path.join(here, "graph", "nodes.py"),
               # which past answers are offered as hints (a filter fix must not replay old answers)
               os.path.join(here, "schema", "knowledge.py")]
    # Every query / answer check (entity_check, sql_guard, number_check, extreme_check ...):
    # a fixed check must not keep replaying an answer it would now reject.
    # The same for query execution (sql/: hierarchy repair, stock labels, executor).
    for sub in ("validation", "sql"):
        vdir = os.path.join(here, sub)
        sources += sorted(os.path.join(vdir, f) for f in os.listdir(vdir) if f.endswith(".py"))
    return _hash(sources)


_CODE_DIGEST = _code_digest()


def _version() -> str:
    """Data refresh + training files + IST date; re-checked every ANSWER_CACHE_VERSION_TTL seconds."""
    now = time.monotonic()
    if _STATE["version"] and now - _STATE["version_at"] < config.ANSWER_CACHE_VERSION_TTL:
        return _STATE["version"]
    try:
        refreshed = _data_refreshed_at()
    except Exception:
        log.warning("answer cache: data version unavailable -- cache bypassed for now", exc_info=True)
        return ""
    # Training files AND the code that writes / checks answers: a fix to either must not
    # keep replaying answers written before it.
    # Content hash, not mtimes: an edit that changes nothing (save, touch) keeps the cache,
    # any real change -- rules, examples, column meanings, KPI definitions, prompt settings
    # in .env, the answer-writing code -- invalidates every entry written before it.
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.dirname(business_rules.PATH)
    sources = [business_rules.PATH, business_rules.EXAMPLES_PATH, business_rules.CHARTS_PATH,
               os.path.join(data_dir, "table_metadata.yaml"), os.path.join(data_dir, "kpi_definitions.yaml"),
               os.path.join(os.path.dirname(here), ".env")]
    digest = _hash(sources, _CODE_DIGEST.copy())
    version = f"{refreshed}|{digest.hexdigest()[:16]}|{datetime.now(_IST):%Y-%m-%d}"
    _STATE.update(version=version, version_at=now)
    return version


def current_version() -> str:
    """The version answers are cached under right now ('' when unavailable)."""
    return _version()


def _entries() -> dict:
    if _STATE["entries"] is None:
        try:
            with open(_PATH, encoding="utf-8") as fh:
                _STATE["entries"] = json.load(fh)
        except (OSError, ValueError):
            _STATE["entries"] = {}
    return _STATE["entries"]


def _save() -> None:
    entries = _entries()
    if len(entries) > config.ANSWER_CACHE_MAX_ENTRIES:  # drop the oldest
        for key in sorted(entries, key=lambda k: entries[k]["at"])[: len(entries) - config.ANSWER_CACHE_MAX_ENTRIES]:
            entries.pop(key, None)
    tmp = _PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(entries, fh, ensure_ascii=False, default=str)
    os.replace(tmp, _PATH)


def _keys(question: str, prior_turns: list[dict] | None) -> tuple[str, str | None]:
    last = (prior_turns or [None])[-1]
    prev = _norm((last or {}).get("resolved_question") or (last or {}).get("question")) if last else ""
    return f"q:{_norm(question)}", (f"f:{prev}|{_norm(question)}" if prev else None)


def _redis_key(key: str) -> str:
    # A deterministic hash of the cache key (itself derived from the question / prior
    # turn, not the model -- see module docstring) rather than the raw text: shorter,
    # fixed-length Redis keys, and never leaks question text into `redis-cli keys`.
    return f"cache:answer:{hashlib.sha1(key.encode('utf-8')).hexdigest()}"


def _redis_hit(key: str, version: str) -> dict | None:
    raw = redis_client.get(_redis_key(key))
    if raw is None:
        return None
    try:
        hit = json.loads(raw)
    except ValueError:
        return None
    return hit if hit.get("version") == version else None


def get(question: str, prior_turns: list[dict] | None) -> dict | None:
    if not config.ANSWER_CACHE_ENABLED:
        return None
    version = _version()
    if not version:
        return None
    standalone, followup = _keys(question, prior_turns)
    for key in (followup, standalone):
        if not key:
            continue
        hit = _redis_hit(key, version)
        with _LOCK:
            if hit is None:
                hit = _entries().get(key)
            if hit and hit.get("version") == version:
                if key == standalone and prior_turns and not _answered_as_asked(hit, question):
                    # Mid-conversation, a bare-text hit is reused only when it answered the
                    # message as typed: 'only dt' was cached as 'How many DT meters are
                    # installed?' and would answer every chat's 'only dt' that way.
                    continue
                return copy.deepcopy(hit["response"])
    return None


def _answered_as_asked(hit: dict, question: str) -> bool:
    debug = ((hit.get("response") or {}).get("response") or {}).get("debug") or {}
    resolved = debug.get("resolved_question")
    return not resolved or _norm(resolved) == _norm(question)


def put(question: str, prior_turns: list[dict] | None, state: dict, entry: dict) -> None:
    """entry = {"response": the API response, "context": the session turn context}."""
    response = entry.get("response") or {}
    if not config.ANSWER_CACHE_ENABLED or response.get("refused") or not str(response.get("summary") or "").strip():
        return
    version = _version()
    if not version:
        return
    standalone, followup = _keys(question, prior_turns)
    key = followup if (state.get("followup") and followup) else standalone
    payload = {"version": version, "at": time.time(), "response": entry}
    with _LOCK:
        _entries()[key] = payload
        try:
            _save()
        except OSError:
            log.warning("answer cache: could not save", exc_info=True)
    # Best-effort mirror -- the local file above is already the durable write;
    # this just makes a repeat hit fast without touching disk, and self-expires.
    redis_client.set(_redis_key(key), json.dumps(payload, ensure_ascii=False, default=str),
                     ttl=config.REDIS_RESPONSE_CACHE_TTL)
