"""Groq text-to-speech for the answer's "Play" button.

Every answer is written through shared/llm_gateway.py; this module only turns
already-written text into audio.
"""

from __future__ import annotations

import logging
import re
import threading
from collections import OrderedDict
from typing import Any

import requests

from shared import config

log = logging.getLogger(__name__)


def available() -> bool:
    return config.LLM_ENABLED


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
