"""Redis connection manager -- the one place every other module reaches Redis
through (shared/session_store.py, gmr_agents/answer_cache.py,
gmr_agents/memory/episodic.py, gmr_agents/main.py's per-session rate limit).

Who controls it: application code. Connects once (a pooled client, built on
first use) and reuses the pool for every call -- callers never construct their
own `redis.Redis(...)`.

Redis is explicitly NOT a single point of failure here:
    * `available()` is the only health check anyone should call before touching
      Redis. It caches its verdict for _CHECK_INTERVAL seconds, so a down Redis
      costs one connect-timeout, not one per request -- and a recovered Redis is
      noticed again within that same window, with no reconnect loop in between.
    * Every helper below (get/set/incr/...) catches its own Redis exceptions,
      logs a warning ONCE per down/up transition (not once per call -- that would
      flood the log), and returns None/False instead of raising. Every caller in
      this codebase already treats None as "fall back to the local store" per
      their own docstrings.
    * REDIS_HOST="" (left blank in .env) disables Redis outright, same effect as
      it being unreachable -- no code path requires Redis to be configured.
"""

from __future__ import annotations

import logging
import threading
import time

from shared import config

log = logging.getLogger(__name__)

try:
    import redis
except Exception:  # redis-py not installed: behave exactly like "Redis unavailable"
    redis = None

_CHECK_INTERVAL = 5.0  # seconds between health re-checks of a Redis we found down (or up)

_LOCK = threading.Lock()
_CLIENT: "redis.Redis | None" = None
_LAST_CHECK = 0.0
_LAST_OK = False


def _build_client():
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    with _LOCK:
        if _CLIENT is None:
            pool = redis.ConnectionPool(
                host=config.REDIS_HOST, port=config.REDIS_PORT, db=config.REDIS_DB,
                decode_responses=True,
                socket_connect_timeout=0.3, socket_timeout=0.5,  # never let a down Redis stall a request
            )
            _CLIENT = redis.Redis(connection_pool=pool)
        return _CLIENT


def available() -> bool:
    """True if Redis answered a PING recently. Cheap to call on every request --
    it only actually pings Redis once every _CHECK_INTERVAL seconds."""
    global _LAST_CHECK, _LAST_OK
    if redis is None or not config.REDIS_HOST:
        return False
    now = time.monotonic()
    if now - _LAST_CHECK < _CHECK_INTERVAL:
        return _LAST_OK
    _LAST_CHECK = now
    try:
        ok = bool(_build_client().ping())
    except Exception as exc:
        if _LAST_OK:  # log only the down transition, not every failed check while it stays down
            log.warning("Redis unreachable -- falling back to local storage (%s)", exc)
        _LAST_OK = False
        return False
    if ok and not _LAST_OK:
        log.info("Redis reachable again at %s:%s", config.REDIS_HOST, config.REDIS_PORT)
    _LAST_OK = ok
    return ok


def client():
    """The shared client, or None if Redis is not configured/available right now.
    Every call site must treat None as "use the fallback", never raise it."""
    return _build_client() if available() else None


# --------------------------------------------------------------------------- #
# small helpers every caller needs -- each one is best-effort and never raises
# --------------------------------------------------------------------------- #

def get(key: str) -> str | None:
    c = client()
    if c is None:
        return None
    try:
        return c.get(key)
    except Exception:
        log.debug("redis GET %s failed", key, exc_info=True)
        return None


def set(key: str, value: str, *, ttl: int | None = None) -> bool:
    c = client()
    if c is None:
        return False
    try:
        c.set(key, value, ex=ttl or None)
        return True
    except Exception:
        log.debug("redis SET %s failed", key, exc_info=True)
        return False


def delete(*keys: str) -> None:
    c = client()
    if c is None or not keys:
        return
    try:
        c.delete(*keys)
    except Exception:
        log.debug("redis DEL %s failed", keys, exc_info=True)


def rate_limit_allow(key: str, limit: int, window_seconds: int) -> bool | None:
    """Fixed-window counter: True = allowed, False = over `limit` requests in the
    current `window_seconds` window, None = Redis unavailable (caller should
    treat None as "allow" -- requirement: bypass rate limiting when Redis is down,
    never block real traffic because a cache layer is unreachable).

    INCR then EXPIRE-if-new is two round trips but no Lua script / WATCH needed:
    the only race is two requests both incrementing from 0 in the same instant,
    which at worst lets the window start a few ms later than the very first hit --
    acceptable for a soft per-session abuse guard, not a billing meter."""
    c = client()
    if c is None:
        return None
    try:
        count = c.incr(key)
        if count == 1:
            c.expire(key, window_seconds)
        return count <= limit
    except Exception:
        log.debug("redis rate limit check %s failed", key, exc_info=True)
        return None
