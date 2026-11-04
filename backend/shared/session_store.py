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
from datetime import datetime, timezone
from typing import Any

from shared import config, redis_client

log = logging.getLogger(__name__)

# Changes within this window share one write (a question and its answer).
_WRITE_DELAY_SECONDS = 0.5


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
            else:
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
        # Same bytes as json.dumps({"sessions": [...]}) with its default separators.
        text = '{"sessions": [' + ", ".join(parts) + "]}"
        try:
            self._write_file(text)
        except OSError as exc:
            log.error("Session store write failed (%s) -- will retry on the next change.", exc)
            self._pending.set()

    def _write_file(self, text: str) -> None:
        with self._io_lock:
            tmp = f"{self.path}.tmp"
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

    def list_sessions(self) -> list[dict]:
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
                    }
        return None

    def patch_session(self, session_id: str, session_name: str | None) -> dict | None:
        with self._lock:
            data = self._read()
            for session in data["sessions"]:
                if session.get("session_id") == session_id:
                    if session_name:
                        session["session_name"] = session_name
                    session["updated_at"] = _now()
                    self._write(data, session_id)
                    return self._summarize(session)
        return None

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
        }
