"""Groq text-to-speech for the answer's "Play" button.

Every answer is written through shared/llm_gateway.py; this module only turns
already-written text into audio.

The Orpheus voice model generates audio token by token like an LLM, and on
long inputs it loses its place: it skips sentences, repeats a row, or trails
off into gibberish (a 600-char answer came back as 86s of audio reading one
sub-division five times). So the text is (1) rewritten into plain speakable
English -- no markup, symbols, ISO dates or "a / b" paths -- and (2) sent as
short sentence-sized chunks, which the model reads faithfully, and the clips
are joined back into one WAV.
"""

from __future__ import annotations

import html
import logging
import re
import struct
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests

from shared import config

log = logging.getLogger(__name__)


def available() -> bool:
    return config.LLM_ENABLED


# --- Text -> speakable text -------------------------------------------------

_MONTHS = ["January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December"]

_CHAR_MAP = {
    " ": " ", " ": " ", " ": " ", "​": "",
    "‐": "-", "‑": "-", "‒": "-",
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "•": ". ", "·": ", ", "…": ".",
    "≥": " at least ", "≤": " at most ", "→": " to ", "←": " from ",
    "₹": " rupees ", "×": " times ", "−": "-",
}
_CHAR_RE = re.compile("|".join(map(re.escape, _CHAR_MAP)))

# Ordered rewrites: (pattern, replacement). Each fixes something the voice
# model either reads wrongly ("slash", "underscore") or silently drops.
_RULES: list[tuple[re.Pattern[str], Any]] = [
    # HTML / markdown markup.
    (re.compile(r"<[^>\n]{1,200}>"), " "),
    (re.compile(r"\[([^\]]*)\]\([^)]*\)"), r"\1"),                 # [text](link)
    (re.compile(r"\*\*|__|`+|~~"), ""),
    (re.compile(r"^\s{0,3}#{1,6}\s*", re.M), ""),                     # headings
    (re.compile(r"^\s*(?:[-*+]|\d{1,2}[.)])\s+", re.M), ""),          # list bullets
    (re.compile(r"^\s*\|?\s*:?-{3,}.*$", re.M), ""),                  # table rule rows
    # Square brackets are vocal directions to Orpheus ("[cheerful]"): never send them.
    (re.compile(r"\[([^\]]*)\]"), r"(\1)"),
    # A line break ends a sentence.
    (re.compile(r"(?<![.!?:;,])[ \t]*\n+"), ". "),
    (re.compile(r"\n+"), " "),
    # Dates: 2026-07-01 -> 1 July 2026; 2026-07 -> July 2026.
    (re.compile(r"\b(\d{4})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b"),
     lambda m: f"{int(m[3])} {_MONTHS[int(m[2]) - 1]} {m[1]}"),
    (re.compile(r"\b(\d{4})-(0[1-9]|1[0-2])\b(?!-)"),
     lambda m: f"{_MONTHS[int(m[2]) - 1]} {m[1]}"),
    # Comparisons and arrows.
    (re.compile(r"\s*>=\s*"), " at least "),
    (re.compile(r"\s*<=\s*"), " at most "),
    (re.compile(r"\s*-+>\s*"), " to "),
    (re.compile(r"\s+>\s+"), " more than "),
    (re.compile(r"\s+<\s+"), " less than "),
    # Ranges: "3-30 days" / "3–30" -> "3 to 30".
    (re.compile(r"(?<![\d.\w-])(\d{1,3})\s?[–-]\s?(\d{1,3})\b(?![-\w])"), r"\1 to \2"),
    (re.compile(r"\s*[–—]\s*"), ", "),
    # Signs and units.
    (re.compile(r"(?<![\w.])-(?=\d)"), "minus "),
    (re.compile(r"(?<=\d)\s?%"), " percent"),
    (re.compile(r"(?<=\d)\s?Cr\b"), " crore"),
    (re.compile(r"(?<=\d)\s?L\b"), " lakh"),
    (re.compile(r"~\s*(?=\d)"), "about "),
    (re.compile(r"\s*&\s*"), " and "),
    # "AGRA / EDC AGRA" is a path, not a fraction: pause instead of "slash".
    (re.compile(r"(?<=[A-Za-z)])\s*/\s*|\s*/\s*(?=[A-Za-z(])"), ", "),
    (re.compile(r"\s*\|\s*"), ", "),
    (re.compile(r"_+"), " "),
    (re.compile(r"[#*~^=\\{}<>]"), " "),
    # Tidy the punctuation the rules above leave behind.
    (re.compile(r"\(\s*\)"), ""),
    (re.compile(r"\s+([,.;:!?])"), r"\1"),
    (re.compile(r"([,;:])(?:\s*[,;:])+"), r"\1"),
    (re.compile(r"[,;:]\s*([.!?])"), r"\1"),
    (re.compile(r"([.!?])(?:\s*[.,;:])+"), r"\1"),
    (re.compile(r"\s+"), " "),
]


def speakable(text: str) -> str:
    """The text as the voice should say it: no markup, symbols spelled out."""
    out = html.unescape(text or "")
    out = _CHAR_RE.sub(lambda m: _CHAR_MAP[m.group(0)], out)
    for pattern, repl in _RULES:
        out = pattern.sub(repl, out)
    return out.strip(" ,;:")


def _cap(text: str, limit: int) -> str:
    """Groq bills per character: cut long answers at a sentence boundary."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    if end >= limit // 2:
        return cut[: end + 1]
    return cut.rsplit(" ", 1)[0] + "."


def _split_long(piece: str, size: int) -> list[str]:
    """A sentence longer than one chunk: break at commas, then at words."""
    out: list[str] = []
    current = ""
    for part in re.split(r"(?<=[,;:])\s+", piece):
        while len(part) > size:
            head = part[:size].rsplit(" ", 1)[0] or part[:size]
            out.append(head)
            part = part[len(head):].strip()
        if current and len(current) + 1 + len(part) > size:
            out.append(current)
            current = part
        else:
            current = f"{current} {part}".strip()
    if current:
        out.append(current)
    return out


def chunk_text(text: str, size: int) -> list[str]:
    """Whole sentences packed into chunks of at most `size` characters."""
    chunks: list[str] = []
    current = ""
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if not sentence:
            continue
        pieces = [sentence] if len(sentence) <= size else _split_long(sentence, size)
        for piece in pieces:
            if current and len(current) + 1 + len(piece) > size:
                chunks.append(current)
                current = piece
            else:
                current = f"{current} {piece}".strip()
    if current:
        chunks.append(current)
    return chunks


def _prepare_tts_text(text: str) -> str:
    return _cap(speakable(text), config.GROQ_TTS_MAX_CHARS)


# --- WAV joining --------------------------------------------------------------

def _wav_parts(wav: bytes) -> tuple[bytes, bytes] | None:
    """(fmt chunk body, PCM data). Groq streams its WAVs, so the RIFF and data
    sizes are 0xFFFFFFFF placeholders: the data runs to the end of the file."""
    if len(wav) < 12 or wav[:4] != b"RIFF" or wav[8:12] != b"WAVE":
        return None
    pos, fmt = 12, None
    while pos + 8 <= len(wav):
        cid, size = wav[pos:pos + 4], struct.unpack("<I", wav[pos + 4:pos + 8])[0]
        body = pos + 8
        if cid == b"data":
            end = len(wav) if size == 0xFFFFFFFF else min(len(wav), body + size)
            return (fmt, wav[body:end]) if fmt else None
        if cid == b"fmt ":
            fmt = wav[body:body + size]
        pos = body + size + (size & 1)
    return None


def _join_wavs(clips: list[bytes]) -> bytes | None:
    parts = [_wav_parts(c) for c in clips]
    if not parts or any(p is None for p in parts):
        return clips[0] if len(clips) == 1 else None
    fmt = parts[0][0]
    channels, rate, _, _, bits = struct.unpack("<HIIHH", fmt[2:16])
    frame = channels * bits // 8
    gap = b"\x00" * (int(rate * 0.12) * frame)  # a short breath between chunks
    pcm = gap.join(p[1][: len(p[1]) - len(p[1]) % frame] for p in parts)
    return (b"RIFF" + struct.pack("<I", 4 + 8 + len(fmt) + 8 + len(pcm)) + b"WAVE"
            + b"fmt " + struct.pack("<I", len(fmt)) + fmt
            + b"data" + struct.pack("<I", len(pcm)) + pcm)


# --- Synthesis ------------------------------------------------------------------

# Audio per chunk is cached, so replaying an answer (or one that shares
# sentences with it) costs nothing.
_TTS_CACHE: OrderedDict[str, bytes] = OrderedDict()
_TTS_CACHE_SIZE = config.GROQ_TTS_CACHE_SIZE
_TTS_CACHE_LOCK = threading.Lock()
_RETRY_IN_RE = re.compile(r"try again in ([\d.]+)\s*(ms|s)")


def _cached_chunk(chunk: str) -> bytes | None:
    with _TTS_CACHE_LOCK:
        audio = _TTS_CACHE.get(chunk)
        if audio is not None:
            _TTS_CACHE.move_to_end(chunk)
        return audio


def _store_chunk(chunk: str, audio: bytes) -> None:
    with _TTS_CACHE_LOCK:
        _TTS_CACHE[chunk] = audio
        while len(_TTS_CACHE) > _TTS_CACHE_SIZE:
            _TTS_CACHE.popitem(last=False)


def _speak_chunk(chunk: str) -> bytes | None:
    audio = _cached_chunk(chunk)
    if audio is None:
        audio = _synthesize(chunk)
        if audio:
            _store_chunk(chunk, audio)
    return audio


def text_to_speech(text: str) -> bytes | None:
    """Synthesize speech for the summary "Play" button. Returns WAV bytes."""
    if not config.GROQ_API_KEY:
        return None
    clean = _prepare_tts_text(text or "")
    if not clean:
        return None
    chunks = chunk_text(clean, max(60, config.GROQ_TTS_CHUNK_CHARS))
    log.info("Groq TTS: %d chars in %d chunk(s).", len(clean), len(chunks))
    with ThreadPoolExecutor(max_workers=min(4, len(chunks))) as pool:
        clips = list(pool.map(_speak_chunk, chunks))
    if any(c is None for c in clips):
        # A gap would silently drop part of the answer -- fail instead.
        return None
    return _join_wavs(clips)


def _synthesize(clean: str) -> bytes | None:
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

    for attempt in range(3):
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
        except Exception as exc:
            log.warning("Groq TTS error: %s", exc)
            return None
        if response.status_code == 200:
            return response.content
        if response.status_code == 429 and attempt < 2:
            # The per-minute character budget is small: wait it out briefly. The
            # daily one ("try again in 28m") fails at once -- the Play button
            # then reads the answer with the browser's own voice.
            m = _RETRY_IN_RE.search(response.text)
            wait = (float(m[1]) / (1000 if m[2] == "ms" else 1)) if m else 5.0
            if wait <= 10:
                time.sleep(wait + 0.5)
                continue
        log.warning("Groq TTS failed [%s]: %s", response.status_code, response.text[:400])
        return None
    return None
