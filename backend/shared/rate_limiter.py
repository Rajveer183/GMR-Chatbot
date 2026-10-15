"""Sliding-window rate limiter for every LLM API call (RPM and TPM per model).

Who controls it: application code, in the one place every LLM request passes --
shared/llm_gateway._admit(). Agents, graph nodes and prompts never see it and
cannot bypass it (no node holds a provider client).

    Agent / graph node -> llm_gateway.chat_json -> _admit -> LIMITER.acquire -> provider HTTP

Algorithm (per key = model id, limits from its provider's settings)
    window holds (timestamp, tokens) of requests in the last `window` seconds
    acquire(est_tokens):
        drop entries older than now - window
        if count < rpm and sum(tokens) + est_tokens <= tpm: record now, ALLOW
        else wait = time until enough oldest entries expire; if wait <= max_wait,
             sleep and retry, else DENY (the gateway then tries the next model)
    record_actual(): replace the estimate with the provider's reported usage.
The check-and-record is one atomic step (a lock), so concurrent requests can never
both take the last slot. Windows live in per-process memory -- correct for one
uvicorn process; several workers / replicas would need a shared store instead.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Protocol

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Limits:
    rpm: int = 0        # requests per window; 0 = unlimited
    tpm: int = 0        # tokens per window; 0 = unlimited


@dataclass
class Reservation:
    key: str
    entry_id: str
    tokens: int
    waited: float = 0.0


class RateLimitTimeout(Exception):
    """The window did not free up within the allowed wait."""


class Backend(Protocol):
    def try_reserve(self, key: str, now: float, window: float, limits: Limits, tokens: int) -> tuple[str | None, float]:
        """Atomically: reserve and return (entry_id, 0), or return (None, seconds until it could fit)."""

    def adjust(self, key: str, entry_id: str, tokens: int) -> None: ...

    def usage(self, key: str, now: float, window: float) -> tuple[int, int]: ...


# --------------------------------------------------------------------------- #
# in-memory backend
# --------------------------------------------------------------------------- #

class MemoryBackend:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._windows: dict[str, deque[list]] = {}   # key -> deque of [ts, tokens, id]

    def _trim(self, dq: deque, now: float, window: float) -> None:
        while dq and dq[0][0] <= now - window:
            dq.popleft()

    def try_reserve(self, key, now, window, limits, tokens):
        with self._lock:
            dq = self._windows.setdefault(key, deque())
            self._trim(dq, now, window)
            used = sum(e[1] for e in dq)
            rpm_ok = not limits.rpm or len(dq) < limits.rpm
            tpm_ok = not limits.tpm or used + tokens <= limits.tpm or not dq  # a lone oversized request may go
            if rpm_ok and tpm_ok:
                entry_id = uuid.uuid4().hex
                dq.append([now, tokens, entry_id])
                return entry_id, 0.0
            return None, self._wait(dq, now, window, limits, tokens, used)

    @staticmethod
    def _wait(dq, now, window, limits, tokens, used) -> float:
        waits = []
        if limits.rpm and len(dq) >= limits.rpm:
            waits.append(dq[len(dq) - limits.rpm][0] + window - now)   # when enough oldest requests expire
        if limits.tpm and used + tokens > limits.tpm:
            freed, need = 0, used + tokens - limits.tpm
            for ts, t, _id in dq:
                freed += t
                if freed >= need:
                    waits.append(ts + window - now)
                    break
        return max(0.0, max(waits, default=0.0)) + 0.01

    def adjust(self, key, entry_id, tokens):
        with self._lock:
            for e in self._windows.get(key, ()):
                if e[2] == entry_id:
                    e[1] = tokens
                    return

    def usage(self, key, now, window):
        with self._lock:
            dq = self._windows.get(key, deque())
            self._trim(dq, now, window)
            return len(dq), sum(e[1] for e in dq)


# --------------------------------------------------------------------------- #
# limiter
# --------------------------------------------------------------------------- #

@dataclass
class Stats:
    allowed: int = 0
    delayed: int = 0
    denied: int = 0
    wait_seconds: float = 0.0
    provider_429: int = 0
    tokens_estimated: int = 0
    tokens_actual: int = 0
    by_key: dict[str, dict] = field(default_factory=dict)


class SlidingWindowLimiter:
    def __init__(self, backend: Backend, window_seconds: float, limits_for, clock=time.time, sleep=time.sleep):
        """limits_for(provider, model) -> Limits. clock/sleep are injectable for tests."""
        self.backend, self.window, self.limits_for = backend, float(window_seconds), limits_for
        self.clock, self.sleep = clock, sleep
        self.stats = Stats()
        self._stats_lock = threading.Lock()

    def acquire(self, provider: str, model: str, est_tokens: int, max_wait: float) -> Reservation:
        """Block up to `max_wait` seconds for a slot; RateLimitTimeout if none frees up."""
        limits = self.limits_for(provider, model)
        key = model
        waited = 0.0
        while True:
            entry_id, wait = self.backend.try_reserve(key, self.clock(), self.window, limits, est_tokens)
            if entry_id is not None:
                self._count(key, allowed=1, delayed=1 if waited else 0, wait=waited, est=est_tokens)
                if waited:
                    log.info("rate limit: %s delayed %.2fs", model, waited)
                return Reservation(key, entry_id, est_tokens, waited)
            if waited + wait > max_wait:
                self._count(key, denied=1, wait=waited)
                log.info("rate limit: %s window full (next slot in %.1fs, max wait %.1fs) -- denied",
                         model, wait, max_wait)
                raise RateLimitTimeout(f"{model}: rate window full, next slot in {wait:.1f}s")
            self.sleep(wait)
            waited += wait

    def record_actual(self, reservation: Reservation | None, actual_tokens: int | None) -> None:
        if reservation is None or not actual_tokens:
            return
        self.backend.adjust(reservation.key, reservation.entry_id, int(actual_tokens))
        with self._stats_lock:
            self.stats.tokens_actual += int(actual_tokens)

    def record_429(self, model: str) -> None:
        self._count(model, p429=1)

    def utilization(self, provider: str, model: str) -> dict:
        limits = self.limits_for(provider, model)
        count, tokens = self.backend.usage(model, self.clock(), self.window)
        return {"requests": count, "rpm_limit": limits.rpm or None, "tokens": tokens, "tpm_limit": limits.tpm or None,
                "rpm_used_pct": round(count * 100 / limits.rpm, 1) if limits.rpm else None,
                "tpm_used_pct": round(tokens * 100 / limits.tpm, 1) if limits.tpm else None}

    def _count(self, key, allowed=0, delayed=0, denied=0, wait=0.0, est=0, p429=0) -> None:
        with self._stats_lock:
            s = self.stats
            s.allowed += allowed
            s.delayed += delayed
            s.denied += denied
            s.wait_seconds += wait
            s.tokens_estimated += est
            s.provider_429 += p429
            k = s.by_key.setdefault(key, {"allowed": 0, "delayed": 0, "denied": 0, "provider_429": 0})
            k["allowed"] += allowed
            k["delayed"] += delayed
            k["denied"] += denied
            k["provider_429"] += p429


def backoff_seconds(consecutive: int, base: float, cap: float, rand: float) -> float:
    """Exponential backoff with jitter for repeated 429s that carry no Retry-After:
    base * 2^(n-1), capped, times (1 + up to 25% jitter) so replicas don't retry in step."""
    return min(cap, base * (2 ** max(0, consecutive - 1))) * (1 + 0.25 * rand)
