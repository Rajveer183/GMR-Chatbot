"""Advanced chat history (17 Nov): pin, rename, archive, folders, full-text search, and
continuing an archived chat -- all in SessionStore, persisted to the same JSON file."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.session_store import SessionStore  # noqa: E402


@pytest.fixture()
def store(tmp_path, monkeypatch):
    # Strictly increasing timestamps (the Windows clock can repeat within a test).
    import itertools

    from shared import session_store

    tick = itertools.count()
    monkeypatch.setattr(session_store, "_now", lambda: f"2026-11-17T00:00:00.{next(tick):06d}+00:00")
    return SessionStore(str(tmp_path / "sessions.json"))


def _chat(store, question, answer):
    sid = store.create_session()["session_id"]
    store.append_message(sid, role="user", content=question)
    store.append_message(sid, role="assistant", content=answer)
    return sid


def test_pin_archive_do_not_reorder_rename_does(store):
    a = _chat(store, "installed meters in agra", "Agra installed 1,200 meters.")
    b = _chat(store, "SAT percentage", "SAT is 82%.")
    order = [s["session_id"] for s in store.list_sessions()]
    assert order == [b, a]

    s = store.patch_session(a, pinned=True)
    assert s["pinned"] and s["pinned_at"]
    store.patch_session(a, archived=True)
    assert [x["session_id"] for x in store.list_sessions()] == [b, a]  # updated_at untouched

    store.patch_session(a, pinned=False)
    assert store.get_session(a)["pinned"] is False
    store.patch_session(a, "Agra install review")
    assert store.list_sessions()[0]["session_name"] == "Agra install review"


def test_folders_move_and_delete_keeps_chats(store):
    a = _chat(store, "stock by warehouse", "Lucknow holds the most.")
    f = store.create_folder("  Supply  ")
    assert f["name"] == "Supply" and f["session_count"] == 0
    assert store.patch_session(a, folder_id=f["folder_id"])["folder_id"] == f["folder_id"]
    assert store.list_folders()[0]["session_count"] == 1
    assert store.rename_folder(f["folder_id"], "Supply chain")["name"] == "Supply chain"

    with pytest.raises(KeyError):
        store.patch_session(a, folder_id="nope")

    assert store.delete_folder(f["folder_id"]) is True
    assert store.list_folders() == []
    assert store.get_session(a)["folder_id"] is None  # chat kept, back in the main list
    assert store.delete_folder(f["folder_id"]) is False


def test_search_matches_all_words_in_any_message(store):
    a = _chat(store, "installed meters in agra", "Agra installed 1,200 meters last month.")
    _chat(store, "SAT percentage", "SAT is 82% across projects.")
    hits = store.search("agra 1,200")
    assert [h["session_id"] for h in hits] == [a]
    assert "1,200" in hits[0]["match_snippet"] and hits[0]["match_role"] == "assistant"
    assert store.search("agra sat") == []  # every word must match
    assert store.search("   ") == []
    store.patch_session(a, archived=True)
    assert store.search("agra")[0]["archived"] is True  # archived chats stay findable


def test_new_question_unarchives(store):
    a = _chat(store, "q1", "a1")
    store.patch_session(a, archived=True)
    store.append_message(a, role="assistant", content="late answer")
    assert store.get_session(a)["archived"] is True
    store.append_message(a, role="user", content="and for last month?")
    assert store.get_session(a)["archived"] is False


def test_persisted_and_reloaded(store, tmp_path):
    a = _chat(store, "q1", "a1")
    f = store.create_folder("Ops")
    store.patch_session(a, pinned=True, folder_id=f["folder_id"])
    store.flush()
    raw = json.loads(open(store.path, encoding="utf-8").read())
    assert raw["folders"][0]["name"] == "Ops"

    again = SessionStore(store.path)
    s = again.list_sessions()[0]
    assert s["pinned"] and s["folder_id"] == f["folder_id"]
    assert again.list_folders()[0]["session_count"] == 1
    # deleting the folder re-serialises the moved chat, not a stale cached copy
    again.delete_folder(f["folder_id"])
    again.flush()
    assert "folder_id" not in json.loads(open(store.path, encoding="utf-8").read())["sessions"][0]


def test_old_file_without_folders_loads(tmp_path):
    p = tmp_path / "old.json"
    p.write_text(json.dumps({"sessions": [{"session_id": "x1", "session_name": "Old", "messages": []}]}), encoding="utf-8")
    s = SessionStore(str(p))
    assert s.list_folders() == []
    row = s.list_sessions()[0]
    assert row["pinned"] is False and row["archived"] is False and row["folder_id"] is None


def test_api_patch_and_folder_endpoints(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from gmr_agents import main

    monkeypatch.setattr(main, "sessions", SessionStore(str(tmp_path / "api.json")))
    c = TestClient(main.app)
    sid = c.post("/api/sessions", json={}).json()["session_id"]
    fid = c.post("/api/folders", json={"name": "KPIs"}).json()["folder_id"]

    r = c.patch(f"/api/sessions/{sid}", json={"pinned": True, "folder_id": fid}).json()
    assert r["pinned"] is True and r["folder_id"] == fid
    r = c.patch(f"/api/sessions/{sid}", json={"archived": True}).json()
    assert r["pinned"] is True and r["archived"] is True and r["folder_id"] == fid  # untouched fields kept
    r = c.patch(f"/api/sessions/{sid}", json={"folder_id": None}).json()
    assert r["folder_id"] is None
    assert c.patch(f"/api/sessions/{sid}", json={"folder_id": "missing"}).status_code == 404
    assert c.patch(f"/api/sessions/{sid}", json={"pinned": None}).status_code == 400
    assert c.get("/api/sessions/search", params={"q": "nothing"}).json() == []
    assert c.get("/api/folders").json()[0]["name"] == "KPIs"
    assert c.delete(f"/api/folders/{fid}").json()["deleted"] is True
