"""Temporary chats (17 Nov): in memory only -- never on disk, listed or searched."""
import json
from datetime import datetime, timedelta, timezone

from shared.session_store import SessionStore


def test_temporary_chat_not_saved(tmp_path):
    path = tmp_path / "sessions.json"
    store = SessionStore(str(path))
    keep = store.create_session("Kept")
    temp = store.create_session("New Session", temporary=True)
    assert temp["temporary"] is True
    store.append_message(temp["session_id"], role="user", content="secret question")
    store.append_message(temp["session_id"], role="assistant", content="answer")
    store.append_message(keep["session_id"], role="user", content="normal question")
    store.flush()

    on_disk = json.loads(path.read_text(encoding="utf-8"))
    ids = [s["session_id"] for s in on_disk["sessions"]]
    assert keep["session_id"] in ids and temp["session_id"] not in ids
    assert "secret" not in path.read_text(encoding="utf-8")

    assert [s["session_id"] for s in store.list_sessions()] == [keep["session_id"]]
    assert store.search("secret") == []
    # Follow-ups still see the temporary chat's history.
    assert store.recent_questions(temp["session_id"]) == ["secret question"]
    assert store.get_session(temp["session_id"])["temporary"] is True

    assert store.delete_session(temp["session_id"]) is True
    assert store.get_session(temp["session_id"]) is None


def test_idle_temporary_chat_is_dropped(tmp_path):
    store = SessionStore(str(tmp_path / "sessions.json"))
    temp = store.create_session(temporary=True)
    gone = store.purge_idle_temporary(now=datetime.now(timezone.utc) + timedelta(days=2))
    assert gone == [temp["session_id"]]
    assert store.get_session(temp["session_id"]) is None
