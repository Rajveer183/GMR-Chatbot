"""The GMR chatbot's FastAPI app.

    POST /api/ai-query   {"question": "...", "session_id": "...", "response_language": "auto"}  ->
      {"agent": "mdm"|"wfm"|null, "domain_routing": {...}, "summary": "...",
       "kpis": [...], "charts": [...], "tables": [...], "insights": [...],
       "warnings": [...], "suggested_followups": [...], "refused": bool,
       "response_language": "auto", "detected_language": "hinglish", "answer_language": "hinglish",
       "session_id": "...", "debug": {"sql": ..., "trace_id": "...", "trace": [...]}}

    POST /api/translate  {"session_id", "trace_id", "target_language"} -> an earlier answer's
                         summary / insights / warnings in another language (language dropdown;
                         no SQL re-run, numbers unchanged -- gmr_agents/language.py)

plus chat sessions, the dataset list, text-to-speech, and the question log:

    GET /api/history            every question asked (newest first; ?q= search, ?session_id=, ?limit/offset)
    GET /api/history/{trace_id} one question's full record + every LLM call it made
    GET /api/llm/status         each model's live free-tier budget
    GET /api/metrics            P50/P95 latency + time to first token, tokens, cost, refusal / repair /
                                cache-hit rates vs the SLO_* targets (?hours=24)

Every answer is written by an LLM (gmr_agents/llm_pipeline.py); debug.trace
has one entry per stage with its timing and the model that ran it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import copy
import os
import queue
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from shared import config, llm_gateway, redis_client
from shared.logging_config import append_jsonl, configure_logging, new_trace_id
from shared.session_store import SessionStore
from gmr_agents import answer_cache, language, prewarm
from gmr_agents.llm_pipeline import run_agent
from gmr_agents.schema import business_rules
from gmr_agents.schema.catalog import MAX_CACHE_AGE_SECONDS, get_catalog

configure_logging()
log = logging.getLogger(__name__)

# Chat sessions (the pre-warmer keeps its throwaway sessions in a store of its own).
sessions = SessionStore(os.path.join(config.OUTPUT_DIR, "gmr_chat_sessions.json"))


class AIQueryRequest(BaseModel):
    question: str
    prior_questions: list[str] | None = None
    session_id: str | None = None
    via_followup: bool | None = None
    # Language dropdown: 'auto' (default -- answer in the user's own language) or a code from
    # gmr_agents/language.py LANGUAGES ('en', 'hi', 'hinglish', 'ta' ...). Unknown values = 'auto'.
    response_language: str | None = None
    # A clicked follow-up shown in another language: `question` is the translated text the user
    # saw (stored, and sets the answer language); this is the verified English question it
    # stands for, which is what runs. Ignored unless it is one of the verified suggestions.
    canonical_question: str | None = None


class SessionCreateRequest(BaseModel):
    session_name: str | None = None


class SessionPatchRequest(BaseModel):
    session_name: str | None = None


def _build_response(question: str, state: dict) -> dict:
    plan = state.get("plan") or {}
    return {
        "question": question,
        "agent": state.get("agent"),
        "domain_routing": state.get("routing") or {},
        "summary": state.get("summary", ""),
        "kpis": state.get("kpis") or [],
        "charts": state.get("charts") or [],
        "tables": state.get("tables") or [],
        "insights": state.get("insights") or [],
        "warnings": state.get("warnings") or [],
        "suggested_followups": state.get("followups") or [],
        "citations": state.get("citations") or [],
        "execution": state.get("execution") or {},
        "refused": bool(state.get("refused")),
        "response_language": state.get("language_pref") or language.AUTO,
        "detected_language": state.get("detected_language"),
        "answer_language": state.get("answer_language") or "en",  # what the summary is written in
        "debug": {
            "planner_used": plan.get("planner", "none"),
            "data_engine": plan.get("data_engine"),  # "athena" | "postgres" | None
            "sql": plan.get("sql"),
            "extra_sql": plan.get("extra_sql") or [],
            "verification": state.get("verification") or {},
            "grounding": state.get("grounding") or {},
            "trace_id": state.get("trace_id"),
            "resolved_question": state.get("resolved_question"),
            "followup": state.get("followup"),  # "regex" | "llm" | None
            "understood": state.get("understood") or {},
            "trace": state.get("trace") or [],
            "reasoning": plan.get("reasoning"),
            "topics": (state.get("triage") or {}).get("topics") or [],
            "schema_tables": state.get("schema_tables") or [],
            "answer_model": state.get("narration_source"),
            "total_ms": state.get("total_ms"),
            "llm_calls": [
                {k: c.get(k) for k in ("task", "model", "status", "ms", "prompt_tokens", "completion_tokens", "cached_tokens")}
                for c in state.get("llm_calls") or []
            ],
        },
    }


def _turn_context(state: dict, response: dict) -> dict:
    """What a later follow-up needs to know about this answer -- stored on
    the assistant message and read back by SessionStore.recent_turns()."""
    plan = state.get("plan") or {}
    ctx: dict = {
        "domain": state.get("agent"),
        "answer_kind": plan.get("answer_kind"),
        # the next turn keeps this turn's place only on the same topic (validation/scope.py)
        "topics": (state.get("routing") or {}).get("topics") or [],
        "filters": (plan.get("filters") or [])[:8],
        "group_by": [g.get("alias") or g.get("source_column") for g in plan.get("group_by") or [] if isinstance(g, dict)],
        "kpis": [f"{k.get('label')}: {k.get('formatted_value') or k.get('value')}" for k in (response.get("kpis") or [])[:5]],
        "sql": (plan.get("sql") or "")[:2000] or None,
    }
    tables = response.get("tables") or []
    if tables and tables[0].get("columns") and tables[0].get("rows"):
        key_col = tables[0]["columns"][0]
        keys = [str(r.get(key_col)) for r in tables[0]["rows"] if str(r.get(key_col, "")).strip()]
        ctx.update(key_column=key_col, keys=[k for k in keys if k.lower() != "total"][:20])
    return {k: v for k, v in ctx.items() if v}


async def _refresh_one_catalog(domain: str) -> None:
    try:
        tables = await asyncio.to_thread(get_catalog(domain).refresh)
        log.info("Catalog refresh: %s catalog -- %d table(s).", domain, len(tables))
    except Exception:
        # A transient Athena / Postgres outage must not kill future refreshes;
        # the catalog already in memory keeps serving meanwhile.
        log.exception("Catalog refresh failed for %s -- will retry next cycle.", domain)


async def _refresh_catalogs(domains: list[str]) -> None:
    """Both domains concurrently, each in a background thread: a refresh is a
    blocking Athena / Postgres round-trip that must never stall the event
    loop (no request would be served at all meanwhile)."""
    log.info("Catalog refresh: starting (%s, in background threads).", " + ".join(domains))
    await asyncio.gather(*(_refresh_one_catalog(d) for d in domains))
    log.info("Catalog refresh: cycle complete.")


async def _keep_catalogs_fresh(stale: list[str]) -> None:
    """Refresh any catalog that started from a stale cache right away, then
    both every MAX_CACHE_AGE_SECONDS -- so a server that runs for days still
    answers from metadata at most an hour old."""
    if stale:
        await _refresh_catalogs(stale)
    while True:
        await asyncio.sleep(MAX_CACHE_AGE_SECONDS)
        await _refresh_catalogs(["mdm", "wfm"])


async def _keep_skeleton_fresh() -> None:
    """Rebuild the read-only data skeleton (and the indexes built from it) in the
    background when it is older than DATA_SKELETON_MAX_AGE -- never on a question."""
    from gmr_agents.schema import data_skeleton

    while True:
        try:
            if await asyncio.to_thread(data_skeleton.build_if_stale):
                await asyncio.to_thread(business_rules.warm_retrieval)
                log.info("Data skeleton refreshed and re-indexed.")
        except Exception:
            log.exception("data skeleton refresh failed -- will retry")
        await asyncio.sleep(3600)


async def _keep_cache_warm() -> None:
    """Re-warm the answer cache whenever its version changes (gold refresh,
    training edit, new day), in a background thread."""
    await asyncio.sleep(config.PREWARM_START_DELAY)
    while True:
        try:
            version = await asyncio.to_thread(prewarm.due)
            if version:
                log.info("prewarm: cache version changed -- warming common questions")
                stats = await asyncio.to_thread(prewarm.warm, answer_question)
                if stats.get("complete", True):  # stopped on "AI busy": try this version again later
                    prewarm.mark_warmed(version)
        except Exception:
            log.exception("prewarm cycle failed -- will retry")
        await asyncio.sleep(config.PREWARM_CHECK_SECONDS)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Start from the cached catalogs whatever their age (a live rebuild takes
    # minutes); a stale one is refreshed in the background straight after.
    stale: list[str] = []
    for domain in ("mdm", "wfm"):
        try:
            if await asyncio.to_thread(get_catalog(domain).warm_start):
                stale.append(domain)
            log.info("Ready: %s catalog -- %d table(s).", domain, len(get_catalog(domain).list_tables()))
        except Exception:
            log.exception("Failed to build %s catalog at startup -- will retry on first request.", domain)
    # Load the local embedding model and build the example/topic search indexes now,
    # so the first question doesn't pay for it.
    try:
        await asyncio.to_thread(business_rules.warm_retrieval)
        log.info("Ready: hybrid retrieval (BM25 + embeddings + rerank) over all knowledge indexes.")
    except Exception:
        log.exception("Retrieval warm-up failed -- it will build on first use.")
    # Orphan place values (blank division / circle with a known child) -> their parent, learnt
    # from the gold data in the background (~15 s); queries run unchanged until it is ready.
    from gmr_agents.sql import hierarchy
    hierarchy.refresh_in_background()
    refresher = asyncio.create_task(_keep_catalogs_fresh(stale))
    skeleton = asyncio.create_task(_keep_skeleton_fresh())
    warmer = asyncio.create_task(_keep_cache_warm()) if config.PREWARM_ENABLED else None
    log.info("Catalog refresh scheduled every %d seconds%s.", MAX_CACHE_AGE_SECONDS,
             f" (stale now: {', '.join(stale)})" if stale else "")
    log.info("Redis: %s (%s:%s) -- %s", "reachable" if redis_client.available() else "unreachable",
             config.REDIS_HOST, config.REDIS_PORT,
             "session mirror / response cache / rate limit active" if redis_client.available() else
             "falling back to local JSON / in-memory logic everywhere Redis would have been used")
    yield
    refresher.cancel()
    skeleton.cancel()
    if warmer:
        warmer.cancel()


app = FastAPI(title="GMR Agentic Router (MDM/WFM)", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)


@app.get("/api/cache/status")
def cache_status() -> dict:
    return {"version": answer_cache.current_version(), "prewarm": prewarm.status()}


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "nl2sql_enabled": config.NVIDIA_ENABLED,
        "mdm_tables": [t.qualified_name for t in get_catalog("mdm").list_tables()],
        "wfm_tables": [t.qualified_name for t in get_catalog("wfm").list_tables()],
    }


class TextToSpeechRequest(BaseModel):
    text: str


@app.post("/api/text-to-speech")
def text_to_speech(req: TextToSpeechRequest) -> Response:
    """Audio for the summary "Play" button (Groq TTS via shared/tts.py)."""
    from shared import tts

    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text must not be empty.")
    if not tts.available():
        raise HTTPException(
            status_code=503,
            detail="Text-to-speech is not configured: set GROQ_API_KEY in backend/.env and restart the backend.",
        )
    audio = tts.text_to_speech(text)
    if not audio:
        raise HTTPException(status_code=502, detail="Text-to-speech service failed -- see backend log for details.")
    return Response(content=audio, media_type="audio/wav")


def _answer_by_trace(store: SessionStore, session_id: str | None, trace_id: str | None = None) -> dict | None:
    """A stored assistant message: the one whose answer has `trace_id`, or the session's latest
    answer. Always the FIRST message carrying that answer -- the original as written, never a
    later re-presentation of it in another language."""
    session = store.get_session(session_id) if session_id else None
    messages = [m for m in (session or {}).get("messages") or []
                if m.get("role") == "assistant" and isinstance(m.get("response"), dict)]
    if not messages:
        return None
    trace_of = lambda m: (m["response"].get("debug") or {}).get("trace_id")
    wanted = trace_id or trace_of(messages[-1])
    return next((m for m in messages if trace_of(m) == wanted), None if trace_id else messages[-1])


def _in_language(response: dict, target: str) -> dict:
    """The answer in its target language. The text is usually written there already by the
    pipeline; the translation layer then only adds the labels and follow-up questions in that
    language (one call, cached by content; none for English). A canned / refusal text or a cached
    answer in another language is translated in that same call. An 'AI busy' answer is left as
    is: no model to translate with."""
    if target == "en" and (response.get("answer_language") or "en") == "en":
        return response  # any other target also gets its labels / follow-up questions translated
    if (response.get("debug") or {}).get("answer_model") == "none (AI busy)":
        return response
    return language.translate_response(response, target)


def _switch_language(question: str, previous: dict, target: str, trace_id: str, session_id: str, t0: float) -> tuple[dict, dict]:
    """'Explain this in English' / 'Hindi mein batao': the previous answer re-presented in another
    language by the translation layer -- no SQL, no new numbers. Its debug (trace_id included)
    stays the original's, so table paging still reads the original query."""
    response = language.translate_response(previous["response"], target)
    response["question"] = question
    ms = round((time.perf_counter() - t0) * 1000, 1)
    response["debug"] = {**(response.get("debug") or {}), "language_switch": True, "total_ms": ms}
    log.info("[%s] LANGUAGE SWITCH -> %s (%s) %.0fms", trace_id, target, response["translation"]["status"], ms)
    append_jsonl("questions", {
        "at": datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime("%Y-%m-%dT%H:%M:%S"),
        "trace_id": trace_id, "session_id": session_id, "question": question, "kind": "language_switch",
        "source_trace_id": (previous["response"].get("debug") or {}).get("trace_id"),
        "refused": bool(response.get("refused")), "summary": str(response.get("summary") or "")[:500],
        "total_ms": ms, "execution": {"latency_ms": ms, "ttft_ms": ms, "db_queries": 0, "termination": "LANGUAGE_SWITCH"}})
    return response, dict(previous.get("context") or {})


def answer_question(question: str, session_id: str | None, *, store: SessionStore | None = None,
                    prior_questions: list[str] | None = None, via_followup: bool = False, emit=None,
                    response_language: str | None = None, canonical_question: str | None = None) -> dict:
    """One question end to end: session, answer cache, LLM pipeline, response language, storage.
    Shared by /api/ai-query, /api/ai-query/stream and the cache pre-warmer.
    emit(event) -- optional progress events for a streaming client.
    response_language -- the dropdown preference ('auto' / None = the user's own language).
    canonical_question -- a clicked follow-up shown translated: the verified English question
    behind it, which is what runs (answered in the language of `question`)."""
    store = store or sessions
    trace_id = new_trace_id()
    if not session_id or store.get_session(session_id) is None:
        session_id = store.create_session("New Session")["session_id"]
        log.info("[%s] new session created: %s", trace_id, session_id)

    prior = prior_questions or store.recent_questions(session_id)
    prior_turns = store.recent_turns(session_id, config.MEMORY_TURNS)
    pref = language.normalize_preference(response_language)
    log.info("[%s] REQUEST session=%s question=%r language=%s", trace_id, session_id, question, pref)

    # `asked` is what the user sent and sees; `question` is what runs. They differ only for a
    # translated follow-up chip: the verified question runs (router fast path, checked SQL),
    # answered in the language the chip was shown in.
    asked, run_pref = question, pref
    if canonical_question and canonical_question != question and business_rules.is_suggestion(canonical_question):
        question = canonical_question
        if pref == language.AUTO:
            run_pref = language.resolve(asked, pref, None, prior_turns)[0]
        log.info("[%s] follow-up %r runs as the verified %r (%s)", trace_id, asked, question, run_pref)

    t0 = time.perf_counter()
    switch_to = language.pure_switch(asked)
    previous = _answer_by_trace(store, session_id) if switch_to else None
    cached = None if previous else answer_cache.get(question, prior_turns)
    if previous is not None:
        response, context = _switch_language(asked, previous, switch_to, trace_id, session_id, t0)
        state = {}
        target, detected = switch_to, language.resolve(asked, pref, None, prior_turns)[1]
    elif cached is not None:
        # Same question, same data refresh, same training, same day: replay the stored LLM answer.
        response, context = cached["response"], cached["context"]
        response["debug"] = {**(response.get("debug") or {}), "trace_id": trace_id, "cached": True,
                             "total_ms": round((time.perf_counter() - t0) * 1000, 1)}
        state = {"resolved_question": (response.get("debug") or {}).get("resolved_question")}
        log.info("[%s] CACHE HIT %.0fms", trace_id, response["debug"]["total_ms"])
        # One record per question, cache hits included, so latency / cache-hit metrics see all
        # traffic (no "sql" field: it never enters the past-question index).
        ms = response["debug"]["total_ms"]
        append_jsonl("questions", {
            "at": datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime("%Y-%m-%dT%H:%M:%S"),
            "trace_id": trace_id, "session_id": session_id, "question": question, "cached": True,
            "refused": bool(response.get("refused")), "summary": str(response.get("summary") or "")[:500],
            "total_ms": ms, "llm_calls": 0, "tokens": 0,
            "execution": {"latency_ms": ms, "ttft_ms": ms, "llm_calls": 0, "input_tokens": 0, "output_tokens": 0,
                          "tool_calls": 0, "db_queries": 0, "estimated_cost_usd": 0.0, "termination": "CACHE_HIT"}})
        # The cache is keyed on the question text, so the replay may be in another language than
        # this request wants (dropdown / explicit request): translated below, never re-queried.
        target, detected = language.resolve(asked, run_pref, response.get("detected_language"), prior_turns)
    else:
        # run_agent never raises: a failure comes back as an error answer, so the
        # session always gets both the question and its (error) reply.
        state = run_agent(question, prior_questions=prior, prior_turns=prior_turns, trace_id=trace_id,
                          session_id=session_id, emit=emit, response_language=run_pref)
        response = _build_response(question, state)
        context = _turn_context(state, response)
        answer_cache.put(question, prior_turns, state, {"response": copy.deepcopy(response), "context": context})
        # target_language is missing only when the pipeline stopped before triage (AI busy).
        target, detected = (state["target_language"], state.get("detected_language")) if state.get("target_language") \
            else language.resolve(asked, run_pref, None, prior_turns)
        if asked != question:
            detected = language.resolve(asked, pref, None, prior_turns)[1]
    response["question"] = asked
    # Answers are written in the target language; anything that is not (a canned refusal, a cached
    # answer in another language) goes through the translation layer -- numbers never change.
    if previous is None:  # a language switch already went through the translation layer
        response = _in_language(response, target)
    response.update(response_language=pref, detected_language=detected)
    response.setdefault("answer_language", "en")
    context = {**context, "language": response["answer_language"]}  # an ambiguous next message keeps it
    response["session_id"] = session_id
    # The triage step's standalone English question (translation / follow-up resolved).
    store.append_message(session_id, role="user", content=asked,
                         normalized=state.get("resolved_question") or question)
    store.append_message(
        session_id, role="assistant", content=response["summary"],
        response=response, via_followup=via_followup, context=context,
    )
    log.info("[%s] RESPONSE stored -> session=%s", trace_id, session_id)
    return response


def _rate_limit_key(req: AIQueryRequest | TranslateRequest, request: Request) -> str:
    """Per-session when a session already exists; a brand-new chat (no
    session_id yet) is keyed by client IP instead, since there is no other
    identity to rate-limit against for its first message."""
    return f"ratelimit:{req.session_id}" if req.session_id else f"ratelimit:ip:{request.client.host if request.client else 'unknown'}"


def _check_rate_limit(req: AIQueryRequest | TranslateRequest, request: Request) -> None:
    """Redis-backed fixed window (REDIS_RATE_LIMIT per REDIS_RATE_WINDOW seconds).
    Bypassed (never blocks) when Redis is unreachable -- requirement: a cache/
    rate-limit layer going down must never take real traffic down with it."""
    allowed = redis_client.rate_limit_allow(_rate_limit_key(req, request), config.REDIS_RATE_LIMIT, config.REDIS_RATE_WINDOW)
    if allowed is False:
        raise HTTPException(status_code=429, detail=f"Too many requests -- limit is {config.REDIS_RATE_LIMIT} per "
                                                     f"{config.REDIS_RATE_WINDOW}s. Please slow down and try again shortly.")


@app.post("/api/ai-query")
def ai_query(req: AIQueryRequest, request: Request) -> dict:
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    _check_rate_limit(req, request)
    with prewarm.live_question():
        return answer_question(question, req.session_id, prior_questions=req.prior_questions,
                               via_followup=bool(req.via_followup), response_language=req.response_language,
                               canonical_question=req.canonical_question)


@app.post("/api/ai-query/stream")
def ai_query_stream(req: AIQueryRequest, request: Request) -> StreamingResponse:
    """Same answer as /api/ai-query, streamed as newline-delimited JSON events:
      {"type": "status", "message": "..."}            pipeline progress
      {"type": "partial", "kpis", "charts", "tables"}  the numbers, once the SQL ran
      {"type": "summary_delta", "text": "..."}         the written answer, piece by piece
      {"type": "done", "response": {...}}              the final response (as /api/ai-query)
      {"type": "error", "message": "..."}"""
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    _check_rate_limit(req, request)
    events: queue.Queue = queue.Queue()

    def work() -> None:
        try:
            with prewarm.live_question():
                response = answer_question(question, req.session_id, prior_questions=req.prior_questions,
                                           via_followup=bool(req.via_followup), emit=events.put,
                                           response_language=req.response_language,
                                           canonical_question=req.canonical_question)
            events.put({"type": "done", "response": response})
        except Exception as exc:  # answer_question already turns pipeline failures into answers
            log.exception("stream request failed")
            events.put({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        finally:
            events.put(None)

    threading.Thread(target=work, daemon=True).start()

    def lines():
        while True:
            event = events.get()
            if event is None:
                return
            yield json.dumps(event, ensure_ascii=False, default=str) + "\n"

    return StreamingResponse(lines(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


class TranslateRequest(BaseModel):
    session_id: str
    trace_id: str
    target_language: str


@app.post("/api/translate")
def translate_answer(req: TranslateRequest, request: Request) -> dict:
    """An earlier answer's text in another language, for the language dropdown. The stored
    answer -- its original wording and validated numbers -- is translated: no SQL, no new chat
    message, KPI cards / charts / tables untouched. target_language 'auto' returns the original.
    A failed translation is not an error: the original text comes back with
    translation.status 'failed'."""
    _check_rate_limit(req, request)
    message = _answer_by_trace(sessions, req.session_id, req.trace_id)
    if message is None:
        raise HTTPException(status_code=404, detail="That answer is not in this chat any more.")
    original = message["response"]
    target = language.normalize_preference(req.target_language)
    with prewarm.live_question():
        out = original if target == language.AUTO else language.translate_response(original, target)
    return {"trace_id": req.trace_id, "target_language": target,
            "answer_language": out.get("answer_language") or "en",
            "summary": out.get("summary") or "", "insights": out.get("insights") or [],
            "warnings": out.get("warnings") or [],
            # values unchanged; display_* fields / column_labels carry the translated labels
            "kpis": out.get("kpis") or [], "charts": out.get("charts") or [], "tables": out.get("tables") or [],
            "suggested_followups": out.get("suggested_followups") or [],
            "suggested_followup_labels": out.get("suggested_followup_labels"),
            "translation": out.get("translation") or {"target": target, "status": "skipped"}}


# --------------------------------------------------------------------------- #
# Question log (outputs/logs/questions.jsonl, llm_calls.jsonl)
# --------------------------------------------------------------------------- #

def _read_jsonl(name: str, max_lines: int = 20000) -> list[dict]:
    """The newest `max_lines` records of outputs/logs/<name>.jsonl, newest first."""
    path = os.path.join(config.LOG_DIR, f"{name}.jsonl")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        lines = deque(fh, maxlen=max_lines)
    out = []
    for line in reversed(lines):
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


@app.get("/api/history")
def history(limit: int = 50, offset: int = 0, q: str | None = None, session_id: str | None = None,
            errors_only: bool = False) -> dict:
    records = _read_jsonl("questions")
    if session_id:
        records = [r for r in records if r.get("session_id") == session_id]
    if q:
        needle = q.lower()
        records = [r for r in records if needle in json.dumps(
            [r.get("question"), r.get("resolved_question"), r.get("summary"), r.get("sql")], default=str).lower()]
    if errors_only:
        records = [r for r in records if r.get("error") or r.get("refused")]
    limit = max(1, min(limit, 500))
    items = [{k: r.get(k) for k in ("at", "trace_id", "session_id", "question", "resolved_question", "kind", "domain",
                                    "topics", "rows", "refused", "error", "total_ms", "models", "llm_calls",
                                    "tokens", "summary")} for r in records[offset: offset + limit]]
    return {"total": len(records), "offset": offset, "items": items}


@app.get("/api/history/{trace_id}")
def history_item(trace_id: str) -> dict:
    record = next((r for r in _read_jsonl("questions") if r.get("trace_id") == trace_id), None)
    if record is None:
        raise HTTPException(status_code=404, detail="No question with that trace id in the log.")
    calls = [c for c in _read_jsonl("llm_calls", max_lines=50000) if c.get("trace_id") == trace_id]
    return {**record, "llm_call_log": list(reversed(calls))}


class TablePageRequest(BaseModel):
    trace_id: str
    page: int = 1
    page_size: int = 50
    table_index: int = 0          # 0 = the answer's main table, i = its extra query i
    offset: int | None = None     # explicit first row (overrides page) -- the UI continues after the
                                  # rows the answer itself showed (e.g. a top-5 answer, then rows 6..)


_PAGE_SIZES = (25, 50, 100)
_TOTALS: dict[tuple[str, int], int] = {}     # (trace_id, table) -> total rows, counted once


def _without_limit(sql: str, engine: str) -> str:
    import sqlglot

    dialect = "postgres" if engine == "postgres" else "presto"
    tree = sqlglot.parse_one(sql, dialect=dialect)
    tree.set("limit", None)
    tree.set("offset", None)
    return tree.sql(dialect=dialect)


def _paged_sql(sql: str, engine: str, offset: int, size: int) -> str:
    """The query's own ORDER BY kept, its LIMIT replaced by this page's OFFSET / LIMIT
    (OFFSET before LIMIT is valid in both Postgres and Trino/Athena)."""
    return f"{_without_limit(sql, engine)} OFFSET {int(offset)} LIMIT {int(size)}"


@app.post("/api/table-page")
def table_page(req: TablePageRequest) -> dict:
    """One page of an earlier answer's table, server-side: the answer's already
    validated SQL (looked up by trace_id) is re-checked by the SQL guard and the
    read-only layer and run with OFFSET / LIMIT -- no LLM call, no new SQL, and
    the rows never enter an LLM prompt."""
    from gmr_agents.llm_pipeline import _cell, raw_cols
    from gmr_agents.schema.catalog import get_catalog
    from gmr_agents.sql import executor
    from gmr_agents.validation import sql_guard

    if req.page < 1 or req.page_size not in _PAGE_SIZES:
        raise HTTPException(status_code=422, detail=f"page must be >= 1 and page_size one of {_PAGE_SIZES}.")
    record = next((r for r in _read_jsonl("questions") if r.get("trace_id") == req.trace_id), None)
    sqls = ([record.get("sql")] + list(record.get("extra_sql") or [])) if record else []
    if not record or req.table_index >= len(sqls) or not sqls[req.table_index]:
        raise HTTPException(status_code=404, detail="No table for that answer -- ask the question again.")
    sql = str(sqls[req.table_index])
    domain = "mdm" if "hes_gold." in sql else "wfm"
    catalog = get_catalog(domain)
    guard = sql_guard.check_sql_safety(sql, catalog)
    if not guard["ok"]:
        raise HTTPException(status_code=422, detail="That answer's query can no longer be validated.")
    base = guard["normalized_sql"]
    engine = guard["data_engine"]
    first = next((t for t in catalog.list_tables() if f".{t.table_name}" in base), None)
    database = first.database if first is not None and engine == "athena" else None
    key = (req.trace_id, req.table_index)
    try:
        if key not in _TOTALS:
            # Same total the answer reported: the query without its LIMIT (a 'top 5' answer is page 1
            # of the full ranking; 'Next' continues it).
            _TOTALS[key] = int(executor.run_sql(f"SELECT COUNT(*) AS n FROM ({_without_limit(base, engine)}) AS counted",
                                                database, engine).iloc[0, 0])
        total = _TOTALS[key]
        offset = req.offset if req.offset is not None and req.offset >= 0 else (req.page - 1) * req.page_size
        if offset >= max(total, 1):
            raise HTTPException(status_code=416, detail=f"Page {req.page} is past the last page.")
        df = executor.run_sql(_paged_sql(base, engine, offset, req.page_size), database, engine)
    except HTTPException:
        raise
    except Exception as exc:
        log.warning("table page %s/%s failed: %s", req.trace_id, req.page, exc)
        raise HTTPException(status_code=503, detail="The page could not be loaded -- please try again.") from None
    pages = max(1, -(-total // req.page_size))
    raw = raw_cols(df.columns)
    has_next = offset + len(df) < total
    return {"columns": [str(c) for c in df.columns], "rows": [[_cell(v, i in raw) for i, v in enumerate(r)] for r in df.itertuples(index=False)],
            "pagination": {"page": req.page, "page_size": req.page_size, "total_count": total, "total_pages": pages,
                           "has_next": has_next, "has_previous": offset > 0, "offset": offset}}


@app.get("/api/llm/status")
def llm_status() -> dict:
    return {"models": llm_gateway.model_status(), "max_repairs": config.LLM_MAX_REPAIRS,
            "question_budget_seconds": config.QUESTION_LLM_BUDGET_SECONDS}


@app.get("/api/metrics")
def service_metrics(hours: float = 24.0) -> dict:
    """P50/P95 latency and time to first token, tokens, cost, LLM / tool / DB work, refusal /
    busy / repair / number-correction rates, cache hit rate and SLO targets (gmr_agents/metrics.py)."""
    from gmr_agents import metrics

    return metrics.compute(max(0.1, min(hours, 24 * 30)))


# --------------------------------------------------------------------------- #
# Sessions
# --------------------------------------------------------------------------- #

@app.get("/api/sessions")
def list_sessions() -> list[dict]:
    return sessions.list_sessions()


@app.post("/api/sessions")
def create_session(req: SessionCreateRequest) -> dict:
    return sessions.create_session(req.session_name or "New Session")


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str) -> dict:
    session = sessions.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return session


@app.patch("/api/sessions/{session_id}")
def patch_session(session_id: str, req: SessionPatchRequest) -> dict:
    updated = sessions.patch_session(session_id, req.session_name)
    if updated is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return updated


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str) -> dict:
    if not sessions.delete_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"deleted": True, "session_id": session_id}


# --------------------------------------------------------------------------- #
# Datasets -- synthesized from the live MDM/WFM catalog, not an ingested file.
# There is no upload/reset here: "dataset" just means "a table this app can
# currently query," refreshed from Athena/Postgres via schema/catalog.py.
# --------------------------------------------------------------------------- #

@app.get("/api/datasets")
def list_datasets() -> dict:
    datasets = []
    for domain in ("mdm", "wfm"):
        for t in get_catalog(domain).list_tables():
            datasets.append({
                "dataset_id": t.qualified_name,
                "display_name": t.table_name,
                "domain": domain,
                "row_count": t.row_count,
                "loaded_at": None,
                "is_default": False,
            })
    return {"datasets": datasets}
