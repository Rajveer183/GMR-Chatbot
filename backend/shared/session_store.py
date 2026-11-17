"""Chat session persistence.

Every message -- including the assistant's full structured response (summary,
kpis, charts, tables, insights) -- is written to a JSON file, so reloading the
page restores each answer exactly as it was first rendered. This file (and its
background writer) stays the source of truth: it is what GET /api/sessions/{id}
and eval/question_bank/eval_followups.py read, and what still works if Redis is
never configured or goes down mid-session.

Redis mirror (optional, shared/redis_client.py): every write also mirrors a
LEAN copy of the session -- just the last REDIS_HISTORY_TTL-bounded
REDIS_HISTORY_MAX_MESSAGES messages, without each assistant message's full
structured `response` (kpis/charts/tables/insights can be tens of KB; the LLM
context never needs them, only `recent_questions`/`recent_turns` do) -- under
`chat:{session_id}`, TTL-refreshed on every write, sent after the store's lock
is released (a slow Redis never holds up other requests). Nothing in this
process reads it: `recent_questions` / `recent_turns` -- what feeds the LLM
prompt -- read the in-memory sessions, which are always current. The mirror
went stale whenever one SET was skipped (Redis marked down for up to 5 s, or a
0.5 s timeout): the next follow-up then resolved against the turn BEFORE the
latest one -- and it was capped at REDIS_HISTORY_MAX_MESSAGES, so
recent_turns(sid, 50) returned at most 10 turns.

The sessions are held in memory and the file is rewritten in the background,
shortly after a change (one write for a question and its answer), and once
more at shutdown. Measured live with a 5 MB file (710 sessions): reading it
~5 times and rewriting it twice per question added ~3 s to every answer
whose own pipeline took 0.2 s.

Each session's JSON text is kept between writes and only the sessions that
changed are serialised again; the file is the same bytes json.dumps would
write. Measured 28 Oct with a 47 MB file (5,357 sessions): a full json.dumps
took ~560 ms per write, holding the lock every request needs.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from shared import config, redis_client

log = logging.getLogger(__name__)

# Changes within this window share one write (a question and its answer).
_WRITE_DELAY_SECONDS = 0.5

# An archived chat that is not unarchived (or continued) within this many days is deleted.
ARCHIVE_RETENTION_DAYS = int(os.environ.get("GMR_ARCHIVE_RETENTION_DAYS", "30"))
# How often list_sessions sweeps for expired archived chats.
_PURGE_INTERVAL_SECONDS = 3600


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _snippet(text: str, at: int, width: int = 90) -> str:
    """About `width` characters of `text` around position `at`, on one line."""
    start = max(0, at - width // 3)
    piece = " ".join(text[start:start + width].split())
    return ("\u2026" if start else "") + piece + ("\u2026" if start + width < len(text) else "")


class SessionStore:
    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.RLock()
        self._io_lock = threading.Lock()
        self._data: dict | None = None
        # session_id -> its json.dumps text, valid until the session changes (_dirty)
        self._frags: dict[str, str] = {}
        self._dirty: set[str] = set()
        self._pending = threading.Event()
        self._writer: threading.Thread | None = None
        self._last_purge = 0.0
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            self._write_file(json.dumps({"sessions": []}))
        atexit.register(self.flush)

    # -- disk ---------------------------------------------------------------

    def _load(self) -> dict:
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict) and isinstance(data.get("sessions"), list):
                return data
        except (json.JSONDecodeError, OSError) as exc:
            log.error("Session store unreadable (%s) -- starting fresh.", exc)
        return {"sessions": []}

    def _read(self) -> dict:
        """The in-memory sessions (loaded from the file once). Callers hold
        self._lock while they change them, then call _write."""
        with self._lock:
            if self._data is None:
                self._data = self._load()
            return self._data

    def _write(self, data: dict, changed: str | None = None) -> None:
        """Record a change (to session `changed`, or to any session when None); the file is
        written by the background writer."""
        with self._lock:
            self._data = data
            if changed is None:
                self._frags.clear()
            elif changed:
                self._dirty.add(changed)
        self._pending.set()
        if self._writer is None or not self._writer.is_alive():
            self._writer = threading.Thread(target=self._write_loop, name="session-store-writer", daemon=True)
            self._writer.start()

    def _write_loop(self) -> None:
        while True:
            self._pending.wait()
            time.sleep(_WRITE_DELAY_SECONDS)
            self.flush()

    def flush(self) -> None:
        """Write the current sessions to the file now, if anything changed."""
        if not self._pending.is_set():
            return
        with self._lock:
            self._pending.clear()
            if self._data is None:
                return
            # Snapshot: requests may change _data next. Unchanged sessions reuse their text.
            for sid in self._dirty:
                self._frags.pop(sid, None)
            self._dirty.clear()
            parts = []
            for s in self._data["sessions"]:
                sid = s.get("session_id")
                if sid not in self._frags:
                    self._frags[sid] = json.dumps(s, ensure_ascii=False, default=str)
                parts.append(self._frags[sid])
            if len(self._frags) > len(parts):  # deleted sessions
                live = {s.get("session_id") for s in self._data["sessions"]}
                self._frags = {k: v for k, v in self._frags.items() if k in live}
            folders = json.dumps(self._data.get("folders") or [], ensure_ascii=False, default=str)
        # Same bytes as json.dumps({"sessions": [...], "folders": [...]}) with its default separators.
        text = '{"sessions": [' + ", ".join(parts) + '], "folders": ' + folders + "}"
        try:
            self._write_file(text)
        except OSError as exc:
            log.error("Session store write failed (%s) -- will retry on the next change.", exc)
            self._pending.set()

    def _write_file(self, text: str) -> None:
        with self._io_lock:
            tmp = f"{self.path}.{os.getpid()}.{threading.get_ident()}.tmp"  # per writer: see knowledge_store._save
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(text)
            os.replace(tmp, self.path)

    # -- Redis mirror (lean, capped, TTL'd -- see module docstring) ---------

    @staticmethod
    def _mirror_key(session_id: str) -> str:
        return f"chat:{session_id}"

    @staticmethod
    def _lean_message(m: dict) -> dict:
        """A message stripped of the bulky structured response (kpis/charts/
        tables) -- only what recent_questions/recent_turns actually read."""
        lean = {"role": m.get("role"), "content": m.get("content")}
        if m.get("normalized"):
            lean["normalized"] = m["normalized"]
        if m.get("context"):
            lean["context"] = m["context"]
        debug = (m.get("response") or {}).get("debug") or {}
        if debug.get("followup") or debug.get("resolved_question"):
            lean["debug"] = {"followup": debug.get("followup"), "resolved_question": debug.get("resolved_question")}
        return lean

    def _mirror_payload(self, session: dict) -> str | None:
        """The lean mirror text, built while the caller holds self._lock (a snapshot);
        None when Redis is unavailable."""
        if not redis_client.available():
            return None
        messages = session.get("messages", [])[-config.REDIS_HISTORY_MAX_MESSAGES:]
        return json.dumps([self._lean_message(m) for m in messages], ensure_ascii=False, default=str)

    def _mirror_to_redis(self, session_id: str, payload: str | None) -> None:
        """Best-effort, called after self._lock is released; a Redis write failure never
        blocks or fails the request (the JSON file above is already the durable copy)."""
        if payload is not None:
            redis_client.set(self._mirror_key(session_id), payload, ttl=config.REDIS_HISTORY_TTL)

    def _messages(self, session_id: str) -> list[dict] | None:
        """The session's messages from the in-memory store (a copy), or None."""
        with self._lock:
            for session in self._read()["sessions"]:
                if session.get("session_id") == session_id:
                    return list(session.get("messages", []))
        return None

    # -- sessions -----------------------------------------------------------

    def create_session(self, session_name: str = "New Session") -> dict:
        with self._lock:
            data = self._read()
            session = {
                "session_id": uuid.uuid4().hex[:16],
                "session_name": session_name or "New Session",
                "created_at": _now(),
                "updated_at": _now(),
                "messages": [],
            }
            data["sessions"].insert(0, session)
            self._write(data, session["session_id"])
        return {
            "session_id": session["session_id"],
            "session_name": session["session_name"],
            "created_at": session["created_at"],
            "updated_at": session["updated_at"],
            "messages": [],
            "last_question": "",
        }

    def purge_expired_archives(self, now: datetime | None = None) -> list[str]:
        """Delete archived chats archived more than ARCHIVE_RETENTION_DAYS ago; returns their ids.
        An archived chat with no archived_at (archived before the date was kept) starts its
        30 days now rather than being deleted on sight."""
        now = now or datetime.now(timezone.utc)
        cutoff = now - timedelta(days=ARCHIVE_RETENTION_DAYS)
        expired: list[str] = []
        with self._lock:
            data = self._read()
            keep = []
            for s in data["sessions"]:
                if s.get("archived"):
                    if not s.get("archived_at"):
                        s["archived_at"] = now.isoformat()
                        self._dirty.add(s["session_id"])
                    else:
                        try:
                            at = datetime.fromisoformat(s["archived_at"])
                        except (TypeError, ValueError):
                            at = now
                        if at.tzinfo is None:
                            at = at.replace(tzinfo=timezone.utc)
                        if at <= cutoff:
                            expired.append(s["session_id"])
                            continue
                keep.append(s)
            if expired or self._dirty:
                data["sessions"] = keep
                self._write(data, "")
            self._last_purge = time.monotonic()
        for sid in expired:
            redis_client.delete(self._mirror_key(sid))
        if expired:
            log.info("Deleted %d archived chat(s) older than %d days.", len(expired), ARCHIVE_RETENTION_DAYS)
        return expired

    def list_sessions(self) -> list[dict]:
        if time.monotonic() - self._last_purge > _PURGE_INTERVAL_SECONDS or not self._last_purge:
            self.purge_expired_archives()
        with self._lock:
            data = self._read()
            sessions = sorted(
                data["sessions"], key=lambda s: s.get("updated_at", ""), reverse=True
            )
            return [self._summarize(s) for s in sessions]

    def get_session(self, session_id: str) -> dict | None:
        with self._lock:
            data = self._read()
            for session in data["sessions"]:
                if session.get("session_id") == session_id:
                    return {
                        "session_id": session["session_id"],
                        "session_name": session.get("session_name", "New Session"),
                        "created_at": session.get("created_at", ""),
                        "updated_at": session.get("updated_at", ""),
                        # A copy: the in-memory list keeps growing while this one is sent.
                        "messages": list(session.get("messages", [])),
                        "last_question": self._last_question(session),
                        "pinned": bool(session.get("pinned")),
                        "archived": bool(session.get("archived")),
                        "archived_at": session.get("archived_at"),
                        "delete_at": self._delete_at(session),
                        "folder_id": session.get("folder_id"),
                    }
        return None

    def patch_session(self, session_id: str, session_name: str | None = None, **changes: Any) -> dict | None:
        """Rename and/or organise a chat. `changes` may hold pinned (bool), archived (bool) and
        folder_id (str, or None to take it out of its folder); keys left out are untouched.
        Only a rename counts as activity (moves it up the list) -- pinning, archiving and
        filing do not reorder the history. KeyError when folder_id names no folder."""
        with self._lock:
            data = self._read()
            for session in data["sessions"]:
                if session.get("session_id") == session_id:
                    if "folder_id" in changes:
                        fid = changes["folder_id"] or None
                        if fid is not None and not any(f.get("folder_id") == fid for f in data.get("folders") or []):
                            raise KeyError(fid)
                        if fid is None:
                            session.pop("folder_id", None)
                        else:
                            session["folder_id"] = fid
                    if session_name:
                        session["session_name"] = session_name
                        session["updated_at"] = _now()
                    if "pinned" in changes:
                        session["pinned"] = bool(changes["pinned"])
                        if session["pinned"]:
                            session["pinned_at"] = _now()
                        else:
                            session.pop("pinned_at", None)
                    if "archived" in changes:
                        if changes["archived"] and not session.get("archived"):
                            session["archived_at"] = _now()  # starts the auto-delete clock
                        elif not changes["archived"]:
                            session.pop("archived_at", None)
                        session["archived"] = bool(changes["archived"])
                    self._write(data, session_id)
                    return self._summarize(session)
        return None

    # -- folders -------------------------------------------------------------

    def list_folders(self) -> list[dict]:
        with self._lock:
            data = self._read()
            counts: dict[str, int] = {}
            for s in data["sessions"]:
                if s.get("folder_id"):
                    counts[s["folder_id"]] = counts.get(s["folder_id"], 0) + 1
            return [{**f, "pinned": bool(f.get("pinned")), "pinned_at": f.get("pinned_at"),
                     "session_count": counts.get(f["folder_id"], 0)} for f in data.get("folders") or []]

    def create_folder(self, name: str) -> dict:
        folder = {"folder_id": uuid.uuid4().hex[:12], "name": (name or "").strip()[:60] or "New folder",
                  "created_at": _now()}
        with self._lock:
            data = self._read()
            data.setdefault("folders", []).append(folder)
            self._write_meta(data)
        return {**folder, "pinned": False, "pinned_at": None, "session_count": 0}

    def update_folder(self, folder_id: str, name: str | None = None, pinned: bool | None = None) -> dict | None:
        """Rename and/or pin a folder (None = leave as is). A pinned folder is listed under
        Pinned as well as under Folders."""
        with self._lock:
            data = self._read()
            for f in data.get("folders") or []:
                if f.get("folder_id") == folder_id:
                    if name is not None:
                        f["name"] = name.strip()[:60] or f["name"]
                    if pinned is not None:
                        f["pinned"] = bool(pinned)
                        if pinned:
                            f["pinned_at"] = _now()
                        else:
                            f.pop("pinned_at", None)
                    self._write_meta(data)
                    count = sum(1 for s in data["sessions"] if s.get("folder_id") == folder_id)
                    return {**f, "pinned": bool(f.get("pinned")), "pinned_at": f.get("pinned_at"), "session_count": count}
        return None

    def rename_folder(self, folder_id: str, name: str) -> dict | None:
        return self.update_folder(folder_id, name=name)

    def delete_folder(self, folder_id: str) -> bool:
        """Remove the folder; its chats are kept and go back to the main list."""
        with self._lock:
            data = self._read()
            folders = data.get("folders") or []
            if not any(f.get("folder_id") == folder_id for f in folders):
                return False
            data["folders"] = [f for f in folders if f.get("folder_id") != folder_id]
            for s in data["sessions"]:
                if s.get("folder_id") == folder_id:
                    s.pop("folder_id", None)
                    self._dirty.add(s["session_id"])
            self._write_meta(data)
        return True

    def _write_meta(self, data: dict) -> None:
        """A folder-list change: no session's text changes (unless the caller marked it
        dirty), so the next flush re-serialises only the small folder list."""
        self._write(data, "")

    # -- search --------------------------------------------------------------

    def search(self, query: str, limit: int = 50) -> list[dict]:
        """Chats whose title or any question/answer contains every word of `query`
        (case-insensitive), most recent first, each with a snippet of the best-matching message."""
        words = [w for w in (query or "").lower().split() if w]
        if not words:
            return []
        hits: list[tuple[str, dict]] = []
        # Scan a snapshot outside the lock (~0.1 s over 7.5k chats) so a search never holds up
        # a question being answered; a message appended meanwhile just may not be matched.
        with self._lock:
            snapshot = list(self._read()["sessions"])
        for session in snapshot:
            name = session.get("session_name") or ""
            texts = [(m.get("role"), m.get("content") or "") for m in session.get("messages", [])]
            blob = (name + "\n" + "\n".join(t for _, t in texts)).lower()
            if not all(w in blob for w in words):
                continue
            snippet, role, best = "", None, 0
            for r, text in texts:  # the message holding the most query words (first wins a tie)
                low = text.lower()
                n = sum(1 for w in words if w in low)
                if n > best:
                    first = min(low.index(w) for w in words if w in low)
                    snippet, role, best = _snippet(text, first), r, n
                    if n == len(words):
                        break
            hits.append((session.get("updated_at", ""),
                         {**self._summarize(session), "match_snippet": snippet, "match_role": role}))
        hits.sort(key=lambda h: h[0], reverse=True)
        return [h for _, h in hits[:limit]]

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            data = self._read()
            before = len(data["sessions"])
            data["sessions"] = [
                s for s in data["sessions"] if s.get("session_id") != session_id
            ]
            if len(data["sessions"]) == before:
                return False
            self._write(data, session_id)
        redis_client.delete(self._mirror_key(session_id))
        return True

    # -- messages -----------------------------------------------------------

    def append_message(
        self,
        session_id: str,
        *,
        role: str,
        content: str,
        response: dict[str, Any] | None = None,
        via_followup: bool | None = None,
        normalized: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict | None:
        message = {
            "id": uuid.uuid4().hex[:16],
            "role": role,
            "content": content,
            "timestamp": _now(),
        }
        # English rewrite of a typo/Hinglish/Hindi question -- what follow-up
        # resolution reads back, so it never has to translate it again.
        if normalized and normalized != content:
            message["normalized"] = normalized
        if response is not None:
            message["response"] = response
        if via_followup is not None:
            message["via_followup"] = bool(via_followup)
        # Compact memory of what this answer was about (plan filters, result
        # keys, SQL) -- what gmr_agents/followup_context.py reads back.
        if context:
            message["context"] = context

        with self._lock:
            data = self._read()
            for session in data["sessions"]:
                if session.get("session_id") == session_id:
                    session.setdefault("messages", []).append(message)
                    session["updated_at"] = _now()
                    if role == "user" and session.get("archived"):
                        session["archived"] = False  # continuing an archived chat restores it
                        session.pop("archived_at", None)
                    # Name a brand-new session after its first question so the
                    # sidebar isn't a wall of "New Session".
                    if role == "user" and session.get("session_name") in (
                        None,
                        "",
                        "New Session",
                    ):
                        session["session_name"] = content.strip()[:60] or "New Session"
                    self._write(data, session_id)
                    payload = self._mirror_payload(session)
                    break
            else:
                return None
        self._mirror_to_redis(session_id, payload)
        return message

    def truncate_messages(self, session_id: str, keep: int) -> dict | None:
        """Keep the first `keep` messages, drop the rest (Regenerate / Edit-and-resubmit in the UI).
        The session as get_session returns it, or None when it does not exist."""
        with self._lock:
            data = self._read()
            for session in data["sessions"]:
                if session.get("session_id") == session_id:
                    messages = session.setdefault("messages", [])
                    if keep < len(messages):
                        del messages[max(keep, 0):]
                        session["updated_at"] = _now()
                        self._write(data, session_id)
                    payload = self._mirror_payload(session)
                    break
            else:
                return None
        self._mirror_to_redis(session_id, payload)
        return self.get_session(session_id)

    def recent_questions(self, session_id: str, limit: int = 6) -> list[str]:
        """Prior user questions, oldest first -- context for resolving
        follow-ups like "and for the previous year?". A question that was
        itself a follow-up comes back as the standalone question it was
        answered as, so "same for SAT" -> "and agra?" still has a full base."""
        messages = self._messages(session_id)
        if not messages:
            return []
        questions: list[str] = []
        for m in messages:
            if m.get("role") == "user" and m.get("content"):
                questions.append(m.get("normalized") or m["content"])
            elif m.get("role") == "assistant" and questions:
                debug = m.get("debug") or (m.get("response") or {}).get("debug") or {}
                if debug.get("followup") and debug.get("resolved_question"):
                    questions[-1] = debug["resolved_question"]
        return questions[-limit:]

    def recent_turns(self, session_id: str, limit: int = 3) -> list[dict]:
        """The last `limit` question/answer pairs, oldest first:
        {question, resolved_question, summary, context}, from the in-memory store."""
        messages = self._messages(session_id)
        if not messages:
            return []
        turns: list[dict] = []
        for m in messages:
            if m.get("role") == "user" and m.get("content"):
                turns.append({"question": m.get("normalized") or m["content"]})
            elif m.get("role") == "assistant" and turns and "summary" not in turns[-1]:
                debug = m.get("debug") or (m.get("response") or {}).get("debug") or {}
                turns[-1].update(
                    resolved_question=debug.get("resolved_question") or turns[-1]["question"],
                    summary=m.get("content", ""),
                    context=m.get("context") or {},
                )
        return [t for t in turns if "summary" in t][-limit:]

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _last_question(session: dict) -> str:
        for message in reversed(session.get("messages", [])):
            if message.get("role") == "user":
                return message.get("content", "")
        return ""

    def _summarize(self, session: dict) -> dict:
        return {
            "session_id": session["session_id"],
            "session_name": session.get("session_name", "New Session"),
            "created_at": session.get("created_at", ""),
            "updated_at": session.get("updated_at", ""),
            "message_count": len(session.get("messages", [])),
            "last_question": self._last_question(session),
            "pinned": bool(session.get("pinned")),
            "pinned_at": session.get("pinned_at"),
            "archived": bool(session.get("archived")),
            "archived_at": session.get("archived_at"),
            "delete_at": self._delete_at(session),
            "folder_id": session.get("folder_id"),
        }

    @staticmethod
    def _delete_at(session: dict) -> str | None:
        """When an archived chat will be auto-deleted (None when it is not archived)."""
        if not session.get("archived") or not session.get("archived_at"):
            return None
        try:
            at = datetime.fromisoformat(session["archived_at"])
        except (TypeError, ValueError):
            return None
        return (at + timedelta(days=ARCHIVE_RETENTION_DAYS)).isoformat()
