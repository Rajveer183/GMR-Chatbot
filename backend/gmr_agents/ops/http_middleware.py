"""HTTP-layer middleware: request id, W3C trace context, timing, slow-request log, compression.

Everything here sits OUTSIDE the answer pipeline: no LLM call, no SQL, no change to any
response body -- only response headers (and gzip transfer encoding when the client asks
for it). Pure ASGI (not BaseHTTPMiddleware), so the streamed /api/ai-query/stream answer
is passed through chunk by chunk exactly as before.

    X-Request-ID     the caller's own id (<= 64 safe chars) or a new one; logged with the request
    traceparent      W3C trace context: the caller's trace id is kept, a new span id is issued,
                     so the backend joins a distributed trace without any tracing library
    Server-Timing    app;dur=<ms until the response headers were sent> -- browser devtools show it

    GET /api/metrics/http   p50 / p95 / max per route over the last HTTP_TIMING_WINDOW requests
    outputs/logs/slow_requests.jsonl   every request slower than HTTP_SLOW_REQUEST_MS

Compression: GZip for JSON / text bodies >= HTTP_GZIP_MIN_BYTES when the client sends
Accept-Encoding: gzip. Never for the ndjson stream (it would hold events back), audio,
Excel or other already-compressed files.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import re
import secrets
import threading
import time

from starlette.datastructures import Headers, MutableHeaders
from starlette.middleware.gzip import GZipMiddleware

from shared import config
from shared.logging_config import append_jsonl

log = logging.getLogger(__name__)

_REQUEST_ID = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")
_TRACEPARENT = re.compile(r"^00-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$")

# Never gzip these: the stream must flush every event, the rest is already compressed.
GZIP_EXCLUDED_TYPES = (
    "application/x-ndjson", "text/event-stream",
    "application/gzip", "application/x-gzip", "application/zip",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/pdf", "audio/*", "video/*", "image/png", "image/jpeg", "image/gif", "image/webp",
    "font/woff", "font/woff2",
)

_lock = threading.Lock()
_timings: dict[str, collections.deque] = {}


def _request_id(headers: Headers) -> str:
    given = (headers.get("x-request-id") or "").strip()
    return given if _REQUEST_ID.match(given) else secrets.token_hex(8)


def _traceparent(headers: Headers) -> str:
    """Child of the caller's trace when it sent a valid traceparent, else a new trace."""
    m = _TRACEPARENT.match((headers.get("traceparent") or "").strip().lower())
    trace_id = m.group(1) if m and m.group(1) != "0" * 32 else secrets.token_hex(16)
    flags = m.group(3) if m else "01"
    return f"00-{trace_id}-{secrets.token_hex(8)}-{flags}"


def _route_of(scope) -> str:
    route = scope.get("route")
    path = getattr(route, "path", None)
    return path or "unmatched"


def _record(route: str, method: str, ms: float, status: int) -> None:
    with _lock:
        q = _timings.get(f"{method} {route}")
        if q is None:
            q = _timings[f"{method} {route}"] = collections.deque(maxlen=max(10, config.HTTP_TIMING_WINDOW))
        q.append((ms, status))


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return round(s[min(len(s) - 1, int(round(p * (len(s) - 1))))], 1)


def http_timing_summary() -> dict:
    """Per route: count, error count, p50 / p95 / max total time (ms), slowest first by p95."""
    with _lock:
        snap = {k: list(v) for k, v in _timings.items()}
    rows = []
    for key, items in snap.items():
        ms = [m for m, _ in items]
        rows.append({"route": key, "count": len(ms), "errors": sum(1 for _, s in items if s >= 500),
                     "p50_ms": _pct(ms, 0.50), "p95_ms": _pct(ms, 0.95), "max_ms": round(max(ms), 1)})
    rows.sort(key=lambda r: r["p95_ms"], reverse=True)
    return {"window_per_route": config.HTTP_TIMING_WINDOW, "slow_threshold_ms": config.HTTP_SLOW_REQUEST_MS,
            "routes": rows}


class RequestContextMiddleware:
    """Adds X-Request-ID / traceparent / Server-Timing headers and records per-route timings."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        start = time.perf_counter()
        headers = Headers(scope=scope)
        rid = _request_id(headers)
        tp = _traceparent(headers)
        state = {"status": 500, "headers_ms": None}

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                state["status"] = message["status"]
                state["headers_ms"] = (time.perf_counter() - start) * 1000
                out = MutableHeaders(scope=message)
                out["X-Request-ID"] = rid
                out["traceparent"] = tp
                out.append("Server-Timing", f"app;dur={state['headers_ms']:.1f}")
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            try:
                total = (time.perf_counter() - start) * 1000
                route, method = _route_of(scope), scope.get("method", "")
                _record(route, method, total, state["status"])
                if config.HTTP_SLOW_REQUEST_MS and total >= config.HTTP_SLOW_REQUEST_MS:
                    rec = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "request_id": rid,
                           "trace": tp.split("-")[1], "method": method, "route": route,
                           "path": scope.get("path"), "status": state["status"], "total_ms": round(total, 1),
                           "headers_ms": round(state["headers_ms"], 1) if state["headers_ms"] is not None else None}
                    # File write off the event loop.
                    asyncio.get_running_loop().run_in_executor(None, append_jsonl, "slow_requests", rec)
            except Exception:  # timing must never break a request
                log.debug("request timing failed", exc_info=True)


def install(app) -> None:
    """GZip inside, request context outermost (so its timing includes compression)."""
    if config.HTTP_GZIP_ENABLED:
        app.add_middleware(GZipMiddleware, minimum_size=max(1, config.HTTP_GZIP_MIN_BYTES), compresslevel=5,
                           exclude_content_types=GZIP_EXCLUDED_TYPES)
    app.add_middleware(RequestContextMiddleware)
