"""Gateway fallback when every model is cooling down, and per-provider model state."""

import time

from shared import config, llm_gateway as g


def _order():
    return [("groq", "test/busy-model"), ("nvidia", "test/busy-model")]


def test_state_is_per_provider():
    g._state(g._key("groq", "test/busy-model")).cooldown_until = time.monotonic() + 3600
    assert g._state(g._key("nvidia", "test/busy-model")).cooldown_until <= time.monotonic()


def test_waits_for_a_model_seconds_away(monkeypatch):
    monkeypatch.setattr(config, "LLM_BUSY_WAIT_SECONDS", 5.0)
    monkeypatch.setattr(g, "_provider_enabled", lambda p: True)
    calls = []

    def fake_try(task, provider, model, *a, **k):
        calls.append(provider)
        st = g._state(g._key(provider, model))
        if st.cooldown_until > time.monotonic():
            raise g._Skip("cooling down")
        return g.LLMResult({"ok": 1}, model, provider, None, None, 1.0)

    monkeypatch.setattr(g, "_try_model", fake_try)
    now = time.monotonic()
    g._state(g._key("groq", "test/busy-model")).cooldown_until = now + 3600
    g._state(g._key("nvidia", "test/busy-model")).cooldown_until = now + 0.3
    res = g.chat_json("triage", "q", system="s", models=_order())
    assert res is not None and res.provider == "nvidia"
    assert calls == ["groq", "nvidia", "groq", "nvidia"]


def test_no_wait_when_every_model_is_parked_long(monkeypatch):
    monkeypatch.setattr(config, "LLM_BUSY_WAIT_SECONDS", 5.0)
    monkeypatch.setattr(g, "_provider_enabled", lambda p: True)
    monkeypatch.setattr(g, "_try_model", lambda *a, **k: (_ for _ in ()).throw(g._Skip("cooling down")))
    now = time.monotonic()
    for p, m in _order():
        g._state(g._key(p, m)).cooldown_until = now + 3600
    t0 = time.monotonic()
    assert g.chat_json("triage", "q", system="s", models=_order()) is None
    assert time.monotonic() - t0 < 1.0
