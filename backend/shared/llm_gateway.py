"""LLM gateway: the one entry point for every LLM call the chatbot makes.

Model choice. Each task names an ordered model list (config.LLM_MODELS_SQL /
LLM_MODELS_ANSWER / _EXPLAIN / _TRIAGE). Groq's free tier gives EACH model its own tokens-per-minute
bucket (8,000), reported on every response in x-ratelimit-* headers. The
gateway remembers those numbers per model and sends a request to the first
model whose bucket can take it right now -- so a burst of questions spreads
over gpt-oss-120b, qwen3.8-27b and gpt-oss-20b, and slow NVIDIA (~25-40s) is
used only when every Groq bucket is spent.

Retries. A failed call is never resent blind to the same model:
  * 429 / 413 / 5xx / timeout / connection error -> the model is cooled down
    (429: for Groq's retry-after) and the NEXT model is tried at once;
  * Groq "json_validate_failed" -> the JSON is recovered from the error's
    failed_generation when possible, else the next model;
  * 401/403/other 4xx -> that provider is skipped for this call.
Every call respects the question's deadline (question_scope()).

Logging. Every attempt is appended to outputs/logs/llm_calls.jsonl (trace id,
task, model, tokens, latency, status, error) and collected on the current
question scope so the per-question record can show which models answered.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import random
import re
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

import requests

from shared import config
from shared import rate_limiter, tokens
from shared.logging_config import append_jsonl

log = logging.getLogger(__name__)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_DURATION_RE = re.compile(r"(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m(?!s))?(?:(\d+(?:\.\d+)?)s)?(?:(\d+(?:\.\d+)?)ms)?")

# A request is never started with less than this much of the question's
# deadline left (it would only be cut off mid-answer).
_MIN_CALL_SECONDS = config.LLM_MIN_CALL_SECONDS


def _rate_limits(provider: str, model: str) -> rate_limiter.Limits:
    return rate_limiter.Limits(rpm=config.LLM_RPM_LIMITS.get(provider, 0), tpm=config.LLM_TPM_LIMITS.get(provider, 0))


# Every LLM request is admitted through this one limiter (see _admit); nothing else
# in the app talks to a provider.
LIMITER = rate_limiter.SlidingWindowLimiter(rate_limiter.MemoryBackend(), config.LLM_RATE_WINDOW_SECONDS,
                                             _rate_limits)
_RESERVATION: contextvars.ContextVar = contextvars.ContextVar("llm_rate_reservation", default=None)
# Wait for a Groq bucket to refill instead of moving on, when it is this close.
_MAX_REFILL_WAIT = config.LLM_MAX_REFILL_WAIT


# Fields a reply must have, complete, for its task. A reply cut off by the model is
# rescued only when every one of them survived intact (a cut-off value is dropped,
# never repaired) -- otherwise it stays "bad_json" and the next model is tried.
_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "triage": ("kind", "question"), "sql": ("sql",), "repair": ("sql",),
    "answer": ("summary",), "explain": ("summary",),
}
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def _scan_object(text: str, start: int) -> tuple[int | None, int | None]:
    """From the '{' at `start`: (index of its matching '}', or None if the text ends
    first; index of the last ',' at the object's top level -- the end of the last
    complete member). String contents and escapes are skipped correctly."""
    depth, in_str, esc, last_member_end = 0, False, False, None
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                return i, last_member_end
        elif ch == "," and depth == 1:
            last_member_end = i
    return None, last_member_end


def _loads(candidate: str) -> Any | None:
    for attempt in (candidate, _TRAILING_COMMA_RE.sub(r"\1", candidate)):
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            continue
    return None


def extract_json(text: str | None, required: tuple[str, ...] | None = None) -> Any | None:
    """Parse a JSON object out of a model response, tolerating code fences,
    <think> blocks, surrounding prose, trailing commas and text after the object.
    With `required`, a reply truncated mid-object is rescued by keeping its complete
    members -- accepted only if every required field is among them."""
    if not text:
        return None
    candidate = _THINK_RE.sub("", text).strip()
    fenced = _JSON_FENCE_RE.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()
    data = _loads(candidate)
    if data is not None:
        return data
    start = candidate.find("{")
    if start == -1:
        return None
    end, last_member_end = _scan_object(candidate, start)
    if end is not None:  # one balanced object followed / preceded by prose
        return _loads(candidate[start: end + 1])
    if required and last_member_end is not None:
        salvaged = _loads(candidate[start:last_member_end] + "}")
        if isinstance(salvaged, dict) and all(str(salvaged.get(k) or "").strip() for k in required):
            log.info("llm: truncated reply rescued (%d complete fields)", len(salvaged))
            return salvaged
    return None


def estimate_tokens(text: str, provider: str | None = None) -> int:
    """Expected token count (tiktoken x per-provider factor learned from real usage --
    shared/tokens.py). Without a provider, the general factor is used."""
    return tokens.estimate(text or "", provider) + (8 if text else 0)


# --------------------------------------------------------------------------- #
# Per-question scope: deadline + collected call records
# --------------------------------------------------------------------------- #

@dataclass
class _Scope:
    trace_id: str
    deadline: float
    calls: list[dict] = field(default_factory=list)
    limit_hit: str | None = None       # set when a per-question cost / usage cap stopped a call


_SCOPE: ContextVar[_Scope | None] = ContextVar("llm_scope", default=None)


@contextmanager
def question_scope(trace_id: str, budget_seconds: float):
    scope = _Scope(trace_id=trace_id, deadline=time.monotonic() + budget_seconds)
    token = _SCOPE.set(scope)
    try:
        yield scope
    finally:
        _SCOPE.reset(token)


def limit_hit() -> str | None:
    """The cost / usage cap that stopped a call in the current question, if any."""
    scope = _SCOPE.get()
    return scope.limit_hit if scope else None


def remaining_seconds() -> float | None:
    scope = _SCOPE.get()
    return None if scope is None else max(0.0, scope.deadline - time.monotonic())


# --------------------------------------------------------------------------- #
# Live rate-limit state per model
# --------------------------------------------------------------------------- #

@dataclass
class _ModelState:
    limit_tokens: int = config.GROQ_DEFAULT_TPM
    remaining_tokens: int = config.GROQ_DEFAULT_TPM
    tokens_reset_at: float = 0.0       # monotonic time the bucket is full again
    remaining_requests: int = 1000
    requests_reset_at: float = 0.0
    cooldown_until: float = 0.0        # after a 429 / outage
    too_large_tokens: int | None = None  # a request this big was rejected (413)
    consecutive_429: int = 0           # drives exponential backoff when no Retry-After is given
    consecutive_timeouts: int = 0      # a model that keeps timing out is parked for longer

    def available_tokens(self, now: float) -> int:
        return self.limit_tokens if now >= self.tokens_reset_at else self.remaining_tokens


_STATE: dict[str, _ModelState] = {}
_STATE_LOCK = threading.Lock()

# Long rests (daily quota spent, repeated 429s / timeouts) survive a restart: they are kept
# in outputs/llm_cooldowns.json as wall-clock times, so a restarted server does not re-probe
# every exhausted model (each probe costs ~0.4-1.5s, a timeout ~12-25s) on its first questions.
_PARKED_PATH = os.path.join(config.PROJECT_ROOT, "outputs", "llm_cooldowns.json")
_PARKED: dict[str, dict] = {}
# Park lengths, from a replay of two days of llm_calls.jsonl: daily quota 1h (LLM_MAX_DAILY_
# LIMIT_COOLDOWN), 3 rate limits in a row 10 min, 2 timeouts in a row 30 min -- skips ~48% of
# the time lost to failed attempts (~2s/question) while a recovered model is back soon.
_REPEATED_429, _REPEATED_429_PARK = 3, 600
_REPEATED_TIMEOUTS, _REPEATED_TIMEOUT_PARK = 2, 1800


def _load_parked() -> None:
    try:
        with open(_PARKED_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return
    now = time.time()
    _PARKED.update({m: v for m, v in (data or {}).items() if isinstance(v, dict) and v.get("until", 0) > now})


def _park(model: str, seconds: float, reason: str, st: _ModelState | None = None) -> None:
    """Rest a model for `seconds` and persist the rest (JSON, wall clock)."""
    st = st or _state(model)
    st.cooldown_until = max(st.cooldown_until, time.monotonic() + seconds)
    now = time.time()
    with _STATE_LOCK:
        _PARKED[model] = {"until": round(now + seconds, 1), "reason": reason[:120]}
        live = {m: v for m, v in _PARKED.items() if v.get("until", 0) > now}
    try:
        os.makedirs(os.path.dirname(_PARKED_PATH), exist_ok=True)
        tmp = _PARKED_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(live, fh, indent=1)
        os.replace(tmp, _PARKED_PATH)
    except OSError:
        log.debug("could not save %s", _PARKED_PATH, exc_info=True)
    log.info("LLM %s parked for %.0f min: %s", model, seconds / 60, reason[:120])


def _state(model: str) -> _ModelState:
    with _STATE_LOCK:
        st = _STATE.get(model)
        if st is None:
            st = _STATE[model] = _ModelState()
            parked = _PARKED.get(model)
            if parked and parked.get("until", 0) > time.time():
                st.cooldown_until = time.monotonic() + (parked["until"] - time.time())
        return st


_load_parked()


def _parse_duration(text: str | None) -> float | None:
    """'2.925s' / '1m26.4s' / '7.66ms' -> seconds."""
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    m = _DURATION_RE.fullmatch(text.strip())
    if not m or not any(m.groups()):
        return None
    h, mi, s, ms = (float(g) if g else 0.0 for g in m.groups())
    return h * 3600 + mi * 60 + s + ms / 1000


def _update_from_headers(model: str, headers: Any) -> None:
    now = time.monotonic()
    st = _state(model)
    with _STATE_LOCK:
        try:
            if headers.get("x-ratelimit-limit-tokens"):
                st.limit_tokens = int(headers["x-ratelimit-limit-tokens"])
            if headers.get("x-ratelimit-remaining-tokens") is not None:
                st.remaining_tokens = int(headers["x-ratelimit-remaining-tokens"])
                st.tokens_reset_at = now + (_parse_duration(headers.get("x-ratelimit-reset-tokens")) or 60.0)
            if headers.get("x-ratelimit-remaining-requests") is not None:
                st.remaining_requests = int(headers["x-ratelimit-remaining-requests"])
                st.requests_reset_at = now + (_parse_duration(headers.get("x-ratelimit-reset-requests")) or 60.0)
        except (TypeError, ValueError):
            pass


def model_status() -> list[dict]:
    """Snapshot of every model's live budget (for /api/llm/status)."""
    now = time.monotonic()
    out = []
    for provider, model in dict.fromkeys(config.LLM_MODELS_TRIAGE + config.LLM_MODELS_SQL + config.LLM_MODELS_ANSWER + config.LLM_MODELS_EXPLAIN):
        st = _state(model)
        out.append({
            "provider": provider, "model": model, "enabled": _provider_enabled(provider),
            "tokens_per_minute": st.limit_tokens if provider == "groq" else None,
            "tokens_available_now": st.available_tokens(now) if provider == "groq" else None,
            "cooling_down_seconds": round(max(0.0, st.cooldown_until - now), 1),
        })
    return out


def _provider_enabled(provider: str) -> bool:
    if provider == "groq":
        return bool(config.GROQ_API_KEY) and config.LLM_ENABLED
    if provider == "nvidia":
        return config.NVIDIA_ENABLED
    base, key = config.PROVIDERS.get(provider, ("", ""))
    return bool(base and key)  # an extra OpenAI-compatible provider from .env


# --------------------------------------------------------------------------- #
# Transport
# --------------------------------------------------------------------------- #

@dataclass
class LLMResult:
    data: dict
    model: str
    provider: str
    prompt_tokens: int | None
    completion_tokens: int | None
    ms: float


class _Skip(Exception):
    """This model can't serve the request now -- try the next one."""


def decoding(task: str) -> tuple[float, float, int]:
    """(temperature, top_p, top_k) for a task from LLM_DECODING (deterministic by default)."""
    return config.LLM_DECODING.get(task) or config.LLM_DECODING.get("sql" if task == "repair" else "answer", (0.0, 1.0, 0))


# Decoding parameters a provider rejected (HTTP 400) -- not sent to it again.
_UNSUPPORTED_PARAMS: dict[str, set[str]] = {}
_DECODING: contextvars.ContextVar = contextvars.ContextVar("llm_decoding", default=None)


def _payload(provider: str, model: str, system: str, prompt: str, max_tokens: int, temperature: float,
             effort: str, top_p: float = 1.0, top_k: int = 0) -> dict:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    skip = _UNSUPPORTED_PARAMS.get(provider, set())
    if top_p < 1.0 and "top_p" not in skip:
        payload["top_p"] = top_p
    if top_k > 0 and provider in config.LLM_TOP_K_PROVIDERS and "top_k" not in skip:
        payload["top_k"] = top_k
    if "temperature" in skip:
        payload.pop("temperature", None)
    _DECODING.set({"temperature": payload.get("temperature"), "top_p": payload.get("top_p", 1.0),
                   "top_k": payload.get("top_k")})
    if provider == "groq":
        if model.startswith("openai/gpt-oss"):
            payload["reasoning_effort"] = effort
        elif "qwen" in model:
            payload["reasoning_format"] = "hidden"
    elif config.PROVIDER_REASONING_EFFORT.get(provider) and "gemma" not in model:
        # (Gemma models on the Gemini API reject any thinking level -- HTTP 400, measured.)
        # Extra provider's thinking level from .env (Gemini lite models reject "none"; "low" measured
        # 9/10 correct at ~2.5s on real SQL prompts).
        payload["reasoning_effort"] = config.PROVIDER_REASONING_EFFORT[provider]
    elif provider == "nvidia":
        # Thinking off (measured on real SQL prompts: 6/6 correct, 2.8s median vs
        # 5/6, 7.5s with thinking on) -- the verified examples carry the reasoning.
        payload["chat_template_kwargs"] = {"enable_thinking": config.NVIDIA_THINKING}
    return payload


def _send(provider: str, model: str, payload: dict, timeout: float) -> tuple[dict, Any]:
    base, key = config.PROVIDERS[provider]
    resp = requests.post(f"{base}/chat/completions",
                         headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                         json=payload, timeout=timeout)
    if provider == "groq":
        _update_from_headers(model, resp.headers)
    return resp.status_code, resp


def _cached_tokens(usage: dict) -> int | None:
    """Prompt tokens the provider served from its prompt (KV) cache, when it reports
    them (OpenAI-style usage.prompt_tokens_details.cached_tokens: NVIDIA, Groq; Gemini's
    OpenAI-compatible endpoint uses the same field). None = not reported."""
    details = (usage or {}).get("prompt_tokens_details") or {}
    value = details.get("cached_tokens") if isinstance(details, dict) else None
    return int(value) if isinstance(value, (int, float)) else None


def estimate_cost(provider: str, model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimated USD for one call from LLM_PRICING (model key first, then provider)."""
    p_in, p_out = config.LLM_PRICING.get(f"{provider}:{model}") or config.LLM_PRICING.get(provider) or (0.0, 0.0)
    return round(prompt_tokens * p_in / 1e6 + completion_tokens * p_out / 1e6, 6)


def _record(task: str, provider: str, model: str, status: str, ms: float, *, prompt_tokens=None,
            completion_tokens=None, error: str | None = None, cached_tokens: int | None = None) -> None:
    scope = _SCOPE.get()
    rec = {
        "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "trace_id": scope.trace_id if scope else None,
        "task": task, "provider": provider, "model": model, "status": status, "ms": round(ms, 1),
        "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "cached_tokens": cached_tokens,
        "decoding": _DECODING.get(),
    }
    if error:
        rec["error"] = error[:300]
    if prompt_tokens or completion_tokens:
        rec["cost_usd"] = estimate_cost(provider, model, prompt_tokens or 0, completion_tokens or 0)
    if status.startswith("ok"):
        st = _state(model)
        st.consecutive_429 = st.consecutive_timeouts = 0
    if prompt_tokens or completion_tokens:
        # Replace the limiter's estimate with what the provider actually counted.
        LIMITER.record_actual(_RESERVATION.get(), (prompt_tokens or 0) + (completion_tokens or 0))
    if scope is not None:
        scope.calls.append(rec)
    append_jsonl("llm_calls", rec)
    level = logging.INFO if status == "ok" else logging.WARNING
    log.log(level, "[%s] LLM %-8s %-28s %-14s %6.0fms tokens=%s+%s%s", rec["trace_id"], task, model, status, ms,
            prompt_tokens, completion_tokens, f" error={error[:160]}" if error else "")


def _try_model(task: str, provider: str, model: str, system: str, prompt: str, max_tokens: int,
               temperature: float, effort: str, top_p: float = 1.0, top_k: int = 0) -> LLMResult:
    st, timeout, max_tokens = _admit(provider, model, system, prompt, max_tokens)
    payload = _payload(provider, model, system, prompt, max_tokens, temperature, effort, top_p, top_k)
    t0 = time.perf_counter()
    try:
        status, resp = _send(provider, model, payload, timeout)
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
        ms = (time.perf_counter() - t0) * 1000
        _timed_out(model, st)
        _record(task, provider, model, "timeout", ms, error=str(exc))
        raise _Skip("timeout")
    ms = (time.perf_counter() - t0) * 1000

    if status == 200:
        body = resp.json()
        usage = body.get("usage") or {}
        choices = body.get("choices") or []
        content = (choices[0].get("message") or {}).get("content") if choices else None
        data = extract_json(content, _REQUIRED_FIELDS.get(task))
        if not isinstance(data, dict):
            _record(task, provider, model, "bad_json", ms, prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"), error=(content or "")[:200])
            raise _Skip("unparseable reply")
        _record(task, provider, model, "ok", ms, prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"), cached_tokens=_cached_tokens(usage))
        tokens.learn(provider, tokens.count(system) + tokens.count(prompt), usage.get("prompt_tokens"))
        return LLMResult(data, model, provider, usage.get("prompt_tokens"), usage.get("completion_tokens"), ms)
    _fail_status(task, provider, model, st, status, resp, ms, system, prompt, max_tokens)


def _timed_out(model: str, st: _ModelState) -> None:
    """A timeout rests the model LLM_TIMEOUT_COOLDOWN; timing out again right after (no
    success between) parks it like a spent quota -- each timeout costs the question
    12-25s (measured: gemini-3.1-flash-lite 16 timeouts = 200s in two hours)."""
    st.consecutive_timeouts += 1
    if st.consecutive_timeouts >= _REPEATED_TIMEOUTS:
        _park(model, _REPEATED_TIMEOUT_PARK, f"{st.consecutive_timeouts} timeouts in a row", st)
    else:
        st.cooldown_until = time.monotonic() + config.LLM_TIMEOUT_COOLDOWN


def _fail_status(task, provider, model, st, status, resp, ms, system, prompt, max_tokens):
    """Record a non-200 reply, cool the model down when it should rest, raise _Skip
    (or return a recovered LLMResult for Groq's json_validate_failed)."""
    text = resp.text[:600]
    if status == 429:
        full = resp.text or ""
        # Groq sends Retry-After; Gemini puts "retryDelay": "17s" in the body.
        delay = re.search(r'"retryDelay"\s*:\s*"([\d.]+)s"', full)
        st.consecutive_429 += 1
        LIMITER.record_429(model)
        # Provider's Retry-After / retryDelay wins; otherwise exponential backoff with jitter.
        retry_after = (_parse_duration(resp.headers.get("retry-after"))
                       or (float(delay.group(1)) if delay else None)
                       or rate_limiter.backoff_seconds(st.consecutive_429, config.LLM_RATE_BACKOFF_BASE_SECONDS,
                                                       config.LLM_MAX_RATE_LIMIT_COOLDOWN, random.random()))
        # A daily-quota 429 (Groq "tokens per day", Gemini "...PerDay...") won't clear in
        # minutes. Its Retry-After / retryDelay only says when ONE request would fit (Gemini
        # says ~30s, Groq's rolling day a few minutes), so honouring it re-probed spent models
        # every few minutes (measured: ~0.7s lost per question). Park it for the daily cap
        # instead; so too a model that 429s again and again with no success in between.
        daily = "per day" in full.lower() or "perday" in full.lower()
        _record(task, provider, model, "rate_limited", ms, error=text)
        if daily:
            _park(model, max(retry_after, config.LLM_MAX_DAILY_LIMIT_COOLDOWN), "daily quota spent", st)
        elif st.consecutive_429 >= _REPEATED_429:
            _park(model, max(retry_after, _REPEATED_429_PARK), f"{st.consecutive_429} rate limits in a row", st)
        else:
            st.cooldown_until = time.monotonic() + min(retry_after, config.LLM_MAX_RATE_LIMIT_COOLDOWN)
        raise _Skip("rate limited")
    if status == 413:
        st.too_large_tokens = estimate_tokens(system, provider) + estimate_tokens(prompt, provider) + max_tokens
        _record(task, provider, model, "too_large", ms, error=text)
        raise _Skip("request too large")
    if status == 400 and "json_validate_failed" in text:
        # The model answered but broke JSON mode; its text is often still usable.
        try:
            failed = resp.json().get("error", {}).get("failed_generation")
        except ValueError:
            failed = None
        data = extract_json(failed)
        if isinstance(data, dict):
            _record(task, provider, model, "ok_recovered", ms)
            return LLMResult(data, model, provider, None, None, ms)
        _record(task, provider, model, "bad_json", ms, error=text)
        raise _Skip("invalid JSON")
    if status == 400:
        rejected = [p for p in ("top_k", "top_p", "temperature") if p in text.lower()]
        if rejected:
            _UNSUPPORTED_PARAMS.setdefault(provider, set()).update(rejected)
            log.warning("LLM %s rejects %s -- not sent to %s again", model, rejected, provider)
            _record(task, provider, model, "unsupported_param", ms, error=text)
            raise _Skip(f"unsupported decoding parameter {rejected}")
    if status >= 500:
        st.cooldown_until = time.monotonic() + config.LLM_FAILURE_COOLDOWN
    _record(task, provider, model, f"http_{status}", ms, error=text)
    if status in (404, 410):
        # Model retired / unknown to the provider ("reached its end of life"): it will not come
        # back within minutes -- measured 51 wasted attempts in one hour before this.
        _park(model, config.LLM_MAX_DAILY_LIMIT_COOLDOWN, f"HTTP {status}: model gone / not found", st)
    raise _Skip(f"HTTP {status}")


def _admit(provider: str, model: str, system: str, prompt: str, max_tokens: int) -> tuple[_ModelState, float, int]:
    """Can this model take the request now? Returns (state, timeout, max_tokens) or raises _Skip."""
    now = time.monotonic()
    st = _state(model)
    left = remaining_seconds()
    if left is not None and left < _MIN_CALL_SECONDS:
        raise _Skip("question time budget spent")
    if st.cooldown_until > now:
        raise _Skip(f"cooling down {st.cooldown_until - now:.1f}s")
    # Hard context cap per call: never send more than MAX_CONTEXT_TOKENS (well under every
    # configured model's window); callers budget their prompts far below it.
    if estimate_tokens(system, provider) + estimate_tokens(prompt, provider) + max_tokens > config.MAX_CONTEXT_TOKENS:
        log.error("LLM %s: prompt over MAX_CONTEXT_TOKENS (%d) -- not sent", model, config.MAX_CONTEXT_TOKENS)
        raise _Skip("CONTEXT_LIMIT_EXCEEDED")
    scope = _SCOPE.get()
    if scope is not None:
        # Per-question usage caps: a runaway loop cannot keep spending calls / tokens.
        # 429s / 4xx rejections cost nothing (and are routine with long fallback chains): not counted.
        attempts = [c for c in scope.calls if not (c["status"] == "rate_limited" or c["status"].startswith("http_4"))]
        used = sum((c.get("prompt_tokens") or 0) + (c.get("completion_tokens") or 0) for c in scope.calls)
        if config.QUESTION_MAX_LLM_CALLS and len(attempts) >= config.QUESTION_MAX_LLM_CALLS:
            scope.limit_hit = f"COST_LIMIT_EXCEEDED: {len(attempts)} LLM calls this question"
            raise _Skip(scope.limit_hit)
        if config.QUESTION_MAX_TOKENS and used + estimate_tokens(system, provider) + estimate_tokens(prompt, provider) > config.QUESTION_MAX_TOKENS:
            scope.limit_hit = f"COST_LIMIT_EXCEEDED: ~{used} tokens used this question"
            raise _Skip(scope.limit_hit)
        spent = sum(c.get("cost_usd") or 0 for c in scope.calls)
        next_cost = estimate_cost(provider, model, estimate_tokens(system, provider) + estimate_tokens(prompt, provider), max_tokens)
        if config.QUESTION_MAX_COST_USD and spent + next_cost > config.QUESTION_MAX_COST_USD:
            scope.limit_hit = f"COST_LIMIT_EXCEEDED: ~${spent:.4f} spent this question"
            raise _Skip(scope.limit_hit)

    if provider == "groq":
        # Groq counts prompt + max_tokens against the minute bucket when admitting a request.
        need = estimate_tokens(system, provider) + estimate_tokens(prompt, provider) + max_tokens
        if need > st.limit_tokens or (st.too_large_tokens and need >= st.too_large_tokens):
            raise _Skip(f"request ~{need} tokens exceeds the {st.limit_tokens}/min limit")
        avail = st.available_tokens(now)
        if avail < need:
            wait = st.tokens_reset_at - now
            if wait <= _MAX_REFILL_WAIT and (left is None or left - wait > _MIN_CALL_SECONDS + 2):
                time.sleep(max(0.0, wait) + 0.1)
            else:
                raise _Skip(f"bucket has ~{avail} of ~{need} tokens (refills in {wait:.1f}s)")
        timeout = config.GROQ_CALL_TIMEOUT
    elif provider != "nvidia":
        timeout = config.EXTRA_PROVIDER_TIMEOUT
    else:
        timeout = config.NVIDIA_TIMEOUT_SECONDS
        # No per-minute bucket here; with thinking on, reasoning tokens count
        # against max_tokens -- a tight cap cut the JSON off mid-reply (measured).
        max_tokens = max(max_tokens, config.NVIDIA_MIN_MAX_TOKENS)
    # Sliding-window RPM / TPM for this model. Wait briefly for a slot; if none frees up
    # within LLM_RATE_MAX_WAIT_SECONDS (or the question's time budget) move on to the
    # next model instead of sending a request that would only come back 429.
    est = estimate_tokens(system, provider) + estimate_tokens(prompt, provider) + max_tokens
    max_wait = config.LLM_RATE_MAX_WAIT_SECONDS
    if left is not None:
        max_wait = max(0.0, min(max_wait, left - _MIN_CALL_SECONDS - 2))
    try:
        _RESERVATION.set(LIMITER.acquire(provider, model, est, max_wait))
    except rate_limiter.RateLimitTimeout as exc:
        raise _Skip(str(exc))
    left = remaining_seconds()
    if left is not None:
        timeout = min(timeout, left)
    return st, timeout, max_tokens


def chat_json(task: str, prompt: str, *, system: str, models: list[tuple[str, str]] | None = None,
              max_tokens: int = 800, temperature: float | None = None, effort: str = "low") -> LLMResult | None:
    """Ask the first model that can take this request now; on any failure move
    to the next model. Returns None only when every model failed or was
    unavailable (the caller shows an honest "try again" message)."""
    temp, top_p, top_k = decoding(task)
    temp = temp if temperature is None else temperature
    order = models or {"sql": config.LLM_MODELS_SQL, "repair": config.LLM_MODELS_SQL,
                       "triage": config.LLM_MODELS_TRIAGE, "explain": config.LLM_MODELS_EXPLAIN,
                       }.get(task, config.LLM_MODELS_ANSWER)
    skipped: list[str] = []
    for provider, model in order:
        if not _provider_enabled(provider):
            continue
        try:
            return _try_model(task, provider, model, system, prompt, max_tokens, temp, effort, top_p, top_k)
        except _Skip as why:
            skipped.append(f"{model}: {why}")
            continue
        except Exception as exc:  # never let one model's bug kill the question
            log.warning("LLM %s on %s failed unexpectedly: %s", task, model, exc, exc_info=True)
            skipped.append(f"{model}: {exc}")
    log.warning("LLM %s: no model could answer -- %s", task, "; ".join(skipped))
    return None


# --------------------------------------------------------------------------- #
# Streaming (the answer writer): the "summary" string is passed on as it is
# written, so the user sees the answer appear word by word.
# --------------------------------------------------------------------------- #

class _SummaryStream:
    """Incrementally extracts the value of the "summary" key from streamed JSON
    text and hands each newly decoded piece to on_text."""

    _ESC = {"\"": "\"", "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}

    def __init__(self, on_text):
        self.on_text = on_text
        self.buf = ""
        self.pos = None      # index in buf where the summary value starts
        self.done = False

    def feed(self, chunk: str) -> None:
        self.buf += chunk
        if self.done:
            return
        if self.pos is None:
            m = re.search(r'"summary"\s*:\s*"', self.buf)
            if not m:
                return
            self.pos = m.end()
        out, i = [], self.pos
        while i < len(self.buf):
            ch = self.buf[i]
            if ch == "\\":
                if i + 1 >= len(self.buf):
                    break
                nxt = self.buf[i + 1]
                if nxt == "u":
                    if i + 6 > len(self.buf):
                        break
                    try:
                        out.append(chr(int(self.buf[i + 2:i + 6], 16)))
                    except ValueError:
                        pass
                    i += 6
                    continue
                out.append(self._ESC.get(nxt, nxt))
                i += 2
                continue
            if ch == '"':
                self.done = True
                i += 1
                break
            out.append(ch)
            i += 1
        self.pos = i
        if out:
            self.on_text("".join(out))


def _try_model_stream(task, provider, model, system, prompt, max_tokens, temperature, effort, on_text,
                      top_p: float = 1.0, top_k: int = 0) -> LLMResult:
    st, timeout, max_tokens = _admit(provider, model, system, prompt, max_tokens)
    payload = {**_payload(provider, model, system, prompt, max_tokens, temperature, effort, top_p, top_k), "stream": True}
    base, key = config.PROVIDERS[provider]
    url = f"{base}/chat/completions"
    t0 = time.perf_counter()
    stream = _SummaryStream(on_text)
    sent_any = False
    usage: dict = {}
    try:
        with requests.post(url, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                           json=payload, timeout=timeout, stream=True) as resp:
            if provider == "groq":
                _update_from_headers(model, resp.headers)
            if resp.status_code != 200:
                ms = (time.perf_counter() - t0) * 1000
                return _fail_status(task, provider, model, st, resp.status_code, resp, ms, system, prompt, max_tokens)
            resp.encoding = "utf-8"
            for line in resp.iter_lines(chunk_size=32, decode_unicode=True):  # small reads: pass text on promptly
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except ValueError:
                    continue
                usage = obj.get("usage") or (obj.get("x_groq") or {}).get("usage") or usage
                for choice in obj.get("choices") or []:
                    piece = (choice.get("delta") or {}).get("content")
                    if piece:
                        before = stream.pos
                        stream.feed(piece)
                        sent_any = sent_any or stream.pos != before
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
        ms = (time.perf_counter() - t0) * 1000
        _timed_out(model, st)
        _record(task, provider, model, "timeout", ms, error=str(exc))
        if sent_any:
            raise _StreamBroken("timeout after text was sent")
        raise _Skip("timeout")
    ms = (time.perf_counter() - t0) * 1000
    data = extract_json(stream.buf, _REQUIRED_FIELDS.get(task))
    if not isinstance(data, dict):
        _record(task, provider, model, "bad_json", ms, error=stream.buf[:200])
        if sent_any:
            raise _StreamBroken("unparseable reply after text was sent")
        raise _Skip("unparseable reply")
    _record(task, provider, model, "ok", ms, prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"), cached_tokens=_cached_tokens(usage))
    tokens.learn(provider, tokens.count(system) + tokens.count(prompt), usage.get("prompt_tokens"))
    return LLMResult(data, model, provider, usage.get("prompt_tokens"), usage.get("completion_tokens"), ms)


class _StreamBroken(Exception):
    """The stream failed after part of the answer reached the user."""


def chat_json_stream(task: str, prompt: str, *, system: str, on_text, models: list[tuple[str, str]] | None = None,
                     max_tokens: int = 800, temperature: float | None = None, effort: str = "low") -> LLMResult | None:
    """chat_json, but the reply's "summary" text is passed to on_text as it is
    generated. Falls through the model list like chat_json while nothing has
    been sent; once text reached the user, a failure ends the call (None)."""
    temp, top_p, top_k = decoding(task)
    temp = temp if temperature is None else temperature
    order = models or config.LLM_MODELS_ANSWER
    skipped: list[str] = []
    for provider, model in order:
        if not _provider_enabled(provider):
            continue
        try:
            return _try_model_stream(task, provider, model, system, prompt, max_tokens, temp, effort, on_text, top_p, top_k)
        except _Skip as why:
            skipped.append(f"{model}: {why}")
        except _StreamBroken as why:
            log.warning("LLM %s stream on %s broke: %s", task, model, why)
            return None
        except Exception as exc:
            log.warning("LLM %s stream on %s failed unexpectedly: %s", task, model, exc, exc_info=True)
            skipped.append(f"{model}: {exc}")
    log.warning("LLM %s: no model could stream -- %s", task, "; ".join(skipped))
    return None
