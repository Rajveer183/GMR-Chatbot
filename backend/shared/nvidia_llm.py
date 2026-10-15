"""NVIDIA NIM REST transport. Pure I/O -- no prompts and no business logic
live here, mirroring backend/llm.py's own split (that module stays the Groq
client for question understanding and TTS; this one is NVIDIA NIM, used for
NL2SQL, routing tie-breaks and answer text).

NVIDIA NIM (https://build.nvidia.com) exposes a free, OpenAI-compatible
chat/completions API -- same wire format as the paid OpenAI transport this
module replaced, just pointed at a different base URL/key/model.

Public interface: `available()` and `chat_json()`. Callers: the SQL planner
and router (gmr_agents/planner, gmr_agents/supervisor), the KPI knowledge
answers (gmr_agents/schema/kpi_knowledge.py) and the narrator's polish
(shared/agent/narrator.py).

Every function returns None (or {}) on failure rather than raising, and logs
the real reason. Callers are required to have a deterministic fallback --
an LLM outage degrades what gets shown, never crashes the request. This is
the same contract backend/llm.py already promises; this module keeps it for
a second provider rather than inventing a different one.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import requests

from shared import config
from shared.llm import _extract_json  # same tolerant JSON-from-text parser; no reason to duplicate it

log = logging.getLogger(__name__)

# Same retry posture as the transport this replaces: a rate limit or
# transient server overload is worth waiting out and asking again, unlike a
# 4xx (bad request, bad auth) which fails the same way every time no matter
# how many times it's retried. NVIDIA's free tier is rate-limited (40 req/
# min as of writing), so 429s here are an expected, retryable steady state,
# not just an outage signal.
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
# 3, not 4 -- one fewer attempt also means one fewer request burned against
# the free tier's 40 req/min ceiling every time a call is struggling, so
# this helps the rate limit recover instead of compounding it.
_MAX_ATTEMPTS = config.NVIDIA_MAX_ATTEMPTS  # the initial try plus retries
_BACKOFF_BASE_SECONDS = config.NVIDIA_BACKOFF_SECONDS
# An attempt with less than this left of the question's budget would only
# time out -- give up instead and let the caller's fallback answer.
_MIN_ATTEMPT_SECONDS = config.NVIDIA_MIN_ATTEMPT_SECONDS

# Deadline (time.monotonic()) for the question being answered; None = no budget.
_DEADLINE: ContextVar[float | None] = ContextVar("nvidia_deadline", default=None)


@contextmanager
def time_budget(seconds: float):
    """Every NVIDIA call inside this block shares one wall-clock budget."""
    token = _DEADLINE.set(time.monotonic() + seconds)
    try:
        yield
    finally:
        _DEADLINE.reset(token)


def remaining_budget() -> float | None:
    """Seconds left of the current question's budget, or None outside one."""
    deadline = _DEADLINE.get()
    return None if deadline is None else max(0.0, deadline - time.monotonic())


class _RetryableHTTPError(Exception):
    pass


def available() -> bool:
    return config.NVIDIA_ENABLED


def _send_once(payload: dict[str, Any], model: str, *, timeout_seconds: float) -> str | None:
    """One request against `model`. Returns the raw response text, or None --
    and raises only requests.exceptions.Timeout/ConnectionError or
    _RetryableHTTPError, the failure modes worth retrying (see
    _send_with_retry). Everything else (auth error, malformed request, empty
    response) is a None return, not an exception -- retrying those would
    just wait longer to fail the same way."""
    payload = dict(payload, model=model)
    response = requests.post(
        f"{config.NVIDIA_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {config.NVIDIA_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout_seconds,
    )
    if response.status_code in _RETRYABLE_STATUS_CODES:
        raise _RetryableHTTPError(f"[{response.status_code}] {response.text[:200]}")
    if response.status_code != 200:
        log.warning("NVIDIA chat failed [%s]: %s", response.status_code, response.text[:400])
        return None
    data = response.json()
    choices = data.get("choices") or []
    if not choices:
        log.warning("NVIDIA returned no choices: %s", json.dumps(data)[:400])
        return None
    return choices[0].get("message", {}).get("content")


def _send_with_retry(
    payload: dict[str, Any], model: str, *, max_attempts: int, timeout_seconds: float,
) -> str | None:
    """Retries `model` up to `max_attempts` times, with exponential backoff
    (1s, 2s, 4s, ...) between attempts, on a timeout, connection error, or
    retryable HTTP status. Backoff (rather than retrying immediately) gives
    a congested or rate-limited API actual time to recover instead of
    hammering it again right away."""
    for attempt in range(1, max_attempts + 1):
        left = remaining_budget()
        if left is not None and left < _MIN_ATTEMPT_SECONDS:
            log.warning("NVIDIA chat skipped on model %r: the question's time budget is spent (%.1fs left).",
                        model, left)
            return None
        try:
            return _send_once(payload, model,
                              timeout_seconds=timeout_seconds if left is None else min(timeout_seconds, left))
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, _RetryableHTTPError) as exc:
            if attempt == max_attempts:
                log.warning("NVIDIA chat failed after %s attempts on model %r: %s", attempt, model, exc)
                return None
            delay = _BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
            left = remaining_budget()
            if left is not None and left - delay < _MIN_ATTEMPT_SECONDS:
                log.warning("NVIDIA chat failed on model %r and no time budget is left to retry: %s", model, exc)
                return None
            log.warning(
                "NVIDIA chat transient error on model %r (attempt %s/%s) -- retrying in %.1fs: %s",
                model, attempt, max_attempts, delay, exc,
            )
            time.sleep(delay)
            continue
        except Exception as exc:
            log.warning("NVIDIA chat error on model %r: %s", model, exc)
            return None
    return None


def _send(
    prompt: str, *, system: str | None, temperature: float, json_mode: bool,
    max_attempts: int | None, timeout_seconds: float | None,
) -> str | None:
    """Shared transport for chat()/chat_json(). Tries NVIDIA_MODEL first with
    its full retry budget; if that comes back empty for any reason, falls
    back to NVIDIA_FALLBACK_MODEL (a smaller retry budget, since this is
    already a last resort) before giving up entirely -- an outage or a
    "model overloaded" run of 429/503s on the primary model doesn't
    necessarily mean the whole API is down, often just that specific model
    is congested or rate-limited right now.

    `max_attempts`/`timeout_seconds` default to the module-wide _MAX_ATTEMPTS/
    config.NVIDIA_TIMEOUT_SECONDS when omitted -- every existing caller
    (router, sql_planner) keeps today's behavior unchanged. A caller whose
    result is purely cosmetic (narrator's optional polish pass, see
    shared/agent/narrator.py::_llm_polish) can pass a smaller budget instead,
    since it already has a correct, always-available fallback (the
    deterministic template) and shouldn't be allowed to cost as much as a
    correctness-critical call."""
    if not config.NVIDIA_ENABLED:
        return None

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload: dict[str, Any] = {"messages": messages, "temperature": temperature}
    if json_mode:
        # Requires the word "json" to appear somewhere in the messages --
        # every JSON-mode prompt in this app already asks for a JSON object
        # explicitly, so this is never silently unmet.
        payload["response_format"] = {"type": "json_object"}

    attempts = max_attempts if max_attempts is not None else _MAX_ATTEMPTS
    timeout = timeout_seconds if timeout_seconds is not None else config.NVIDIA_TIMEOUT_SECONDS

    result = _send_with_retry(payload, config.NVIDIA_MODEL, max_attempts=attempts, timeout_seconds=timeout)
    if result is not None:
        return result

    fallback = config.NVIDIA_FALLBACK_MODEL
    if not fallback or fallback == config.NVIDIA_MODEL:
        return None
    log.warning(
        "NVIDIA primary model %r exhausted all retries -- falling back to %r.",
        config.NVIDIA_MODEL, fallback,
    )
    return _send_with_retry(payload, fallback, max_attempts=config.NVIDIA_MAX_ATTEMPTS, timeout_seconds=timeout)


def chat_json(
    prompt: str, *, system: str | None = None, temperature: float = 0.0,
    max_attempts: int | None = None, timeout_seconds: float | None = None,
) -> Any | None:
    """Single-turn completion constrained to a JSON object. Returns the
    parsed object, or None on any failure (no key configured, network
    error, non-200 response, unparseable output)."""
    text = _send(
        prompt, system=system, temperature=temperature, json_mode=True,
        max_attempts=max_attempts, timeout_seconds=timeout_seconds,
    )
    return _extract_json(text) if text else None
