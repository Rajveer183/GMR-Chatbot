"""HTTP middleware (gmr_agents/ops/http_middleware.py): headers, gzip, stream untouched. No LLM, no DB."""

from __future__ import annotations

import gzip
import json

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from gmr_agents.ops import http_middleware


def _app() -> FastAPI:
    app = FastAPI()
    http_middleware.install(app)

    @app.get("/big")
    def big():
        return {"rows": [{"zone": "Varanasi I", "meters": i} for i in range(500)]}

    @app.get("/small")
    def small():
        return {"ok": True}

    @app.get("/stream")
    def stream():
        def lines():
            for i in range(300):
                yield json.dumps({"type": "summary_delta", "text": "x" * 20, "i": i}) + "\n"
        return StreamingResponse(lines(), media_type="application/x-ndjson")

    @app.get("/boom")
    def boom():
        raise RuntimeError("fail")

    return app


def test_headers_on_every_response():
    c = TestClient(_app())
    r = c.get("/small")
    assert r.status_code == 200 and r.json() == {"ok": True}
    assert len(r.headers["x-request-id"]) == 16
    assert r.headers["traceparent"].startswith("00-") and len(r.headers["traceparent"]) == 55
    assert r.headers["server-timing"].startswith("app;dur=")


def test_caller_ids_are_kept():
    c = TestClient(_app())
    tp = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    r = c.get("/small", headers={"X-Request-ID": "ui-123", "traceparent": tp})
    assert r.headers["x-request-id"] == "ui-123"
    trace = r.headers["traceparent"].split("-")
    assert trace[1] == "4bf92f3577b34da6a3ce929d0e0e4736" and trace[2] != "00f067aa0ba902b7" and trace[3] == "01"
    bad = c.get("/small", headers={"X-Request-ID": "bad id\nx", "traceparent": "garbage"})
    assert bad.headers["x-request-id"] != "bad id\nx" and bad.headers["traceparent"].startswith("00-")


def test_big_json_gzipped_body_unchanged():
    c = TestClient(_app())
    plain = c.get("/big", headers={"Accept-Encoding": "identity"})
    zipped = c.get("/big", headers={"Accept-Encoding": "gzip"})
    assert "content-encoding" not in plain.headers
    assert zipped.headers.get("content-encoding") == "gzip"
    assert zipped.json() == plain.json()  # client decompresses -> identical payload


def test_small_and_stream_not_gzipped():
    c = TestClient(_app())
    assert "content-encoding" not in c.get("/small", headers={"Accept-Encoding": "gzip"}).headers
    with c.stream("GET", "/stream", headers={"Accept-Encoding": "gzip"}) as r:
        assert "content-encoding" not in r.headers
        assert r.headers["content-type"].startswith("application/x-ndjson")
        events = [json.loads(line) for line in r.iter_lines() if line]
    assert len(events) == 300 and events[-1]["i"] == 299


def test_errors_still_recorded():
    c = TestClient(_app(), raise_server_exceptions=False)
    assert c.get("/boom").status_code == 500
    rows = {r["route"]: r for r in http_middleware.http_timing_summary()["routes"]}
    assert rows["GET /boom"]["errors"] >= 1


def test_real_app_has_middleware_and_metrics_route():
    from gmr_agents.main import app

    c = TestClient(app)  # no `with`: lifespan (catalog warm-up) does not run
    r = c.get("/api/metrics/http")
    assert r.status_code == 200 and "routes" in r.json()
    assert "x-request-id" in r.headers and "server-timing" in r.headers
    assert gzip  # imported for clarity of intent
