"""Groq transport -- text-to-speech only.

Groq does not write any answer text in this app (backend/agent/narrator.py's
summary is a deterministic template; backend/agent/sql_planner.py's SQL comes
from NVIDIA NIM). Its one remaining job is turning already-decided text into
audio for the "Play" button.

`_extract_json` stays here as a small shared utility (NVIDIA's client,
backend/nvidia_llm.py, imports it to parse its own JSON responses) -- it has
nothing to do with Groq specifically, it just hasn't needed a home of its own
since nothing else currently uses it.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections import OrderedDict
from typing import Any

import requests

from shared import config

log = logging.getLogger(__name__)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def available() -> bool:
    return config.LLM_ENABLED


def _extract_json(text: str) -> Any | None:
    """Parse a JSON object out of a model response, tolerating code fences and
    surrounding prose."""
    if not text:
        return None
    candidate = text.strip()
    fenced = _JSON_FENCE_RE.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    # Fall back to the outermost {...} span.
    start, end = candidate.find("{"), candidate.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as exc:
            log.warning("LLM returned unparseable JSON: %s", exc)
    return None


def chat_json(prompt: str, *, system: str, timeout_seconds: float | None = None, max_tokens: int | None = None) -> Any | None:
    """One Groq chat call in JSON mode. Returns the parsed object, or None on
    any failure (callers have a fallback). The only retry is a single short
    wait when Groq's per-minute token limit says it frees up within seconds."""
    if not available():
        return None
    payload = {
        "model": config.GROQ_LLM_MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "temperature": config.GROQ_LLM_TEMPERATURE,
        "max_tokens": max_tokens or config.GROQ_LLM_MAX_TOKENS,
        "response_format": {"type": "json_object"},
    }
    if config.GROQ_LLM_MODEL.startswith("openai/gpt-oss"):
        # Reasoning model: keep its hidden "thinking" (billed as output tokens) short.
        payload["reasoning_effort"] = "low"
    def send():
        return requests.post(
            f"{config.GROQ_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {config.GROQ_API_KEY}", "Content-Type": "application/json"},
            json=payload,
            timeout=timeout_seconds or config.GROQ_LLM_TIMEOUT,
        )

    try:
        response = send()
        if response.status_code == 429:
            try:
                wait = float(response.headers.get("retry-after", "99"))
            except ValueError:
                wait = 99.0
            if wait <= config.GROQ_LLM_MAX_RETRY_WAIT:
                log.info("Groq chat rate-limited -- retrying once in %.1fs.", wait)
                time.sleep(wait + 0.2)
                response = send()
        if response.status_code != 200:
            log.warning("Groq chat failed [%s]: %s", response.status_code, response.text[:400])
            return None
        body = response.json()
        usage = body.get("usage") or {}
        log.info("Groq chat: %s prompt + %s completion tokens.", usage.get("prompt_tokens"), usage.get("completion_tokens"))
        return _extract_json(body["choices"][0]["message"]["content"])
    except Exception as exc:
        log.warning("Groq chat error: %s", exc)
        return None


_MD_RE = re.compile(r"\*\*|__|`|#+\s|\[(.*?)\]\(.*?\)")
_WS_RE = re.compile(r"\s+")

# Groq bills TTS per input character, so every Play click is kept as small as
# it can be: markdown/whitespace stripped, long summaries cut at a sentence
# boundary, and the audio for an exact text reused instead of re-synthesized.
_TTS_CACHE: OrderedDict[str, bytes] = OrderedDict()
_TTS_CACHE_SIZE = config.GROQ_TTS_CACHE_SIZE
_TTS_CACHE_LOCK = threading.Lock()


def _prepare_tts_text(text: str) -> str:
    clean = _WS_RE.sub(" ", _MD_RE.sub(lambda m: m.group(1) or "", text)).strip()
    limit = config.GROQ_TTS_MAX_CHARS
    if len(clean) <= limit:
        return clean
    cut = clean[:limit]
    # Prefer ending on a full sentence; fall back to the last word.
    end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    if end >= limit // 2:
        return cut[: end + 1]
    return cut.rsplit(" ", 1)[0] + "."


def text_to_speech(text: str) -> bytes | None:
    """Synthesize speech for the summary "Play" button. Returns WAV bytes."""
    if not config.GROQ_API_KEY:
        return None
    clean = _prepare_tts_text(text or "")
    if not clean:
        return None

    with _TTS_CACHE_LOCK:
        cached = _TTS_CACHE.get(clean)
        if cached is not None:
            _TTS_CACHE.move_to_end(clean)
            log.info("Groq TTS cache hit (%d chars) -- no API call.", len(clean))
            return cached

    audio = _synthesize(clean)
    if audio:
        with _TTS_CACHE_LOCK:
            _TTS_CACHE[clean] = audio
            while len(_TTS_CACHE) > _TTS_CACHE_SIZE:
                _TTS_CACHE.popitem(last=False)
    return audio


def _synthesize(clean: str) -> bytes | None:
    log.info("Groq TTS request: %d chars.", len(clean))
    payload: dict[str, Any] = {
        "model": config.GROQ_TTS_MODEL,
        "input": clean,
        "voice": config.GROQ_TTS_VOICE,
        "response_format": "wav",
    }
    try:
        payload["speed"] = float(config.GROQ_TTS_SPEED)
    except (TypeError, ValueError):
        pass

    try:
        response = requests.post(
            f"{config.GROQ_BASE_URL}/audio/speech",
            headers={
                "Authorization": f"Bearer {config.GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=config.GROQ_TTS_TIMEOUT,
        )
        if response.status_code != 200:
            log.warning("Groq TTS failed [%s]: %s", response.status_code, response.text[:400])
            return None
        return response.content
    except Exception as exc:
        log.warning("Groq TTS error: %s", exc)
        return None
