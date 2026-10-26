"""Unit tests for the LLM sliding-window rate limiter (shared/rate_limiter.py) and
its integration in shared/llm_gateway (429 handling, Retry-After, backoff).

    python -m unittest discover -s tests -v        (from backend/)

A fake clock makes window tests instant. The Redis test runs only when the
`redis` package is installed and TEST_REDIS_URL points at a server.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from shared import llm_gateway, rate_limiter  # noqa: E402
from shared.rate_limiter import Limits, MemoryBackend, RateLimitTimeout, SlidingWindowLimiter  # noqa: E402


class FakeClock:
    def __init__(self, t: float = 1000.0):
        self.t = t
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.slept.append(s)
        self.t += s


def limiter(rpm=3, tpm=0, window=60, backend=None, clock=None):
    clock = clock or FakeClock()
    lim = SlidingWindowLimiter(backend or MemoryBackend(), window, lambda p, m: Limits(rpm, tpm),
                               clock=clock, sleep=clock.sleep)
    return lim, clock


class WindowTests(unittest.TestCase):
    def test_below_limit(self):
        lim, clock = limiter(rpm=3)
        for _ in range(2):
            lim.acquire("gemini", "m", 10, max_wait=0)
        self.assertEqual((lim.stats.allowed, lim.stats.delayed, clock.slept), (2, 0, []))

    def test_exactly_at_limit(self):
        lim, _ = limiter(rpm=3)
        for _ in range(3):
            lim.acquire("gemini", "m", 10, max_wait=0)
        self.assertEqual(lim.utilization("gemini", "m")["requests"], 3)

    def test_exceeding_limit_is_denied_without_wait(self):
        lim, _ = limiter(rpm=3)
        for _ in range(3):
            lim.acquire("gemini", "m", 10, max_wait=0)
        with self.assertRaises(RateLimitTimeout):
            lim.acquire("gemini", "m", 10, max_wait=0)
        self.assertEqual(lim.stats.denied, 1)

    def test_exceeding_limit_waits_for_oldest_to_expire(self):
        lim, clock = limiter(rpm=2, window=60)
        lim.acquire("gemini", "m", 1, max_wait=0)          # t=1000
        clock.t += 50
        lim.acquire("gemini", "m", 1, max_wait=0)          # t=1050
        res = lim.acquire("gemini", "m", 1, max_wait=15)   # oldest expires at 1060 -> wait ~10s
        self.assertAlmostEqual(res.waited, 10.01, places=2)
        self.assertEqual(lim.stats.delayed, 1)

    def test_old_timestamps_expire(self):
        lim, clock = limiter(rpm=1, window=60)
        lim.acquire("gemini", "m", 1, max_wait=0)
        clock.t += 61
        lim.acquire("gemini", "m", 1, max_wait=0)          # no wait: the first one left the window
        self.assertEqual(clock.slept, [])

    def test_max_wait_timeout(self):
        lim, clock = limiter(rpm=1, window=60)
        lim.acquire("gemini", "m", 1, max_wait=0)
        with self.assertRaises(RateLimitTimeout):
            lim.acquire("gemini", "m", 1, max_wait=5)     # slot frees in 60s > 5s
        self.assertEqual(clock.slept, [])                  # never sleeps when the wait is too long

    def test_models_are_independent(self):
        lim, _ = limiter(rpm=1)
        lim.acquire("gemini", "a", 1, max_wait=0)
        lim.acquire("gemini", "b", 1, max_wait=0)          # other model, own window

    def test_tpm_limit_and_actual_usage(self):
        lim, clock = limiter(rpm=0, tpm=1000)
        r = lim.acquire("gemini", "m", 900, max_wait=0)    # estimate reserves 900
        with self.assertRaises(RateLimitTimeout):
            lim.acquire("gemini", "m", 200, max_wait=0)
        lim.record_actual(r, 300)                          # provider says it used 300
        lim.acquire("gemini", "m", 600, max_wait=0)        # 300 + 600 <= 1000 now fits
        self.assertEqual(lim.utilization("gemini", "m")["tokens"], 900)

    def test_unlimited_when_zero(self):
        lim, _ = limiter(rpm=0, tpm=0)
        for _ in range(500):
            lim.acquire("nvidia", "m", 10_000, max_wait=0)


class ConcurrencyTests(unittest.TestCase):
    def test_concurrent_requests_never_exceed_limit(self):
        lim = SlidingWindowLimiter(MemoryBackend(), 60, lambda p, m: Limits(rpm=10))
        granted, denied = [], []
        barrier = threading.Barrier(50)

        def worker():
            barrier.wait()  # all 50 threads hit the limiter at the same moment
            try:
                granted.append(lim.acquire("gemini", "m", 1, max_wait=0))
            except RateLimitTimeout:
                denied.append(1)

        threads = [threading.Thread(target=worker) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual((len(granted), len(denied)), (10, 40))


class _Resp:
    def __init__(self, status=429, headers=None, text=""):
        self.status_code, self.headers, self.text = status, headers or {}, text


class GatewayTests(unittest.TestCase):
    """The provider-429 path in shared/llm_gateway._fail_status."""

    def setUp(self):
        self.st = llm_gateway._ModelState()
        # keep test 429s out of outputs/logs/llm_calls.jsonl
        self._append = llm_gateway.append_jsonl
        llm_gateway.append_jsonl = lambda *a, **k: None
        # ...and parked test models out of outputs/llm_cooldowns.json
        import tempfile
        self._parked_path = llm_gateway._PARKED_PATH
        llm_gateway._PARKED_PATH = os.path.join(tempfile.mkdtemp(), "llm_cooldowns.json")

    def tearDown(self):
        llm_gateway.append_jsonl = self._append
        llm_gateway._PARKED_PATH = self._parked_path
        for key in ("test-model", "gemini:test-model", "nvidia:test-model"):
            llm_gateway._PARKED.pop(key, None)

    def _fail(self, resp):
        with self.assertRaises(llm_gateway._Skip):
            llm_gateway._fail_status("sql", "gemini", "test-model", self.st, 429, resp, 5.0, "", "", 100)
        return self.st.cooldown_until - time.monotonic()

    def test_retry_after_header_is_respected(self):
        self.assertAlmostEqual(self._fail(_Resp(headers={"retry-after": "7"})), 7, delta=0.5)

    def test_gemini_retry_delay_body_is_respected(self):
        self.assertAlmostEqual(self._fail(_Resp(text='{"error": {"details": [{"retryDelay": "17s"}]}}')), 17, delta=0.5)

    def test_backoff_grows_with_jitter_without_retry_after(self):
        waits = [self._fail(_Resp()) for _ in range(llm_gateway._REPEATED_429 - 1)]
        base = llm_gateway.config.LLM_RATE_BACKOFF_BASE_SECONDS
        for n, w in enumerate(waits):
            expected = min(llm_gateway.config.LLM_MAX_RATE_LIMIT_COOLDOWN, base * 2 ** n)
            self.assertGreaterEqual(w, expected - 0.5)
            self.assertLessEqual(w, expected * 1.25 + 0.5)
        self.assertEqual(self.st.consecutive_429, llm_gateway._REPEATED_429 - 1)

    def test_repeated_429s_park_the_model(self):
        for _ in range(llm_gateway._REPEATED_429 - 1):
            self._fail(_Resp())
        self.assertGreaterEqual(self._fail(_Resp()), llm_gateway._REPEATED_429_PARK - 1)
        self.assertIn("gemini:test-model", llm_gateway._PARKED)

    def test_daily_quota_parks_past_a_short_retry_delay(self):
        # Gemini: retryDelay "30s" on a spent daily quota -- still parked for the daily cap.
        wait = self._fail(_Resp(text='{"error": {"details": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel"},'
                                     ' {"retryDelay": "30s"}]}}'))
        self.assertGreaterEqual(wait, llm_gateway.config.LLM_MAX_DAILY_LIMIT_COOLDOWN - 1)

    def test_retired_model_is_parked(self):
        with self.assertRaises(llm_gateway._Skip):
            llm_gateway._fail_status("sql", "nvidia", "test-model", self.st, 410,
                                     _Resp(410, text='{"status":410,"detail":"reached its end of life"}'), 5.0, "", "", 100)
        self.assertGreaterEqual(self.st.cooldown_until - time.monotonic(), llm_gateway.config.LLM_MAX_DAILY_LIMIT_COOLDOWN - 1)

    def test_repeated_timeouts_park_the_model(self):
        llm_gateway._timed_out("test-model", self.st)
        first = self.st.cooldown_until - time.monotonic()
        llm_gateway._timed_out("test-model", self.st)
        self.assertLessEqual(first, llm_gateway.config.LLM_TIMEOUT_COOLDOWN + 0.5)
        self.assertGreaterEqual(self.st.cooldown_until - time.monotonic(), llm_gateway._REPEATED_TIMEOUT_PARK - 1)

    def test_429_is_counted(self):
        before = llm_gateway.LIMITER.stats.provider_429
        self._fail(_Resp(headers={"retry-after": "1"}))
        self.assertEqual(llm_gateway.LIMITER.stats.provider_429, before + 1)

    def test_backoff_formula(self):
        self.assertEqual(rate_limiter.backoff_seconds(1, 10, 120, 0.0), 10)
        self.assertEqual(rate_limiter.backoff_seconds(3, 10, 120, 0.0), 40)
        self.assertEqual(rate_limiter.backoff_seconds(9, 10, 120, 0.0), 120)   # capped
        self.assertEqual(rate_limiter.backoff_seconds(1, 10, 120, 1.0), 12.5)  # max jitter

    def test_cooling_model_is_skipped_before_limiter(self):
        st = llm_gateway._state(llm_gateway._key("gemini", "cooling-model"))
        st.cooldown_until = time.monotonic() + 30
        with self.assertRaises(llm_gateway._Skip):
            llm_gateway._admit("gemini", "cooling-model", "", "hi", 10)

    def test_full_window_skips_to_next_model(self):
        model = "window-test-model"
        for _ in range(llm_gateway.config.LLM_RPM_LIMITS["gemini"]):
            llm_gateway._admit("gemini", model, "", "hi", 10)
        with self.assertRaises(llm_gateway._Skip) as ctx:  # next slot is ~60s away > LLM_RATE_MAX_WAIT_SECONDS
            llm_gateway._admit("gemini", model, "", "hi", 10)
        self.assertIn("rate window full", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
