import React, { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Archive, Folder, MessageCircle, X } from "lucide-react";
import { apiClient } from "../api/client";
import type { ChatFolder, SessionSearchHit, SessionSummary } from "../api/client";
import type { UIKey } from "../i18n/uiStrings";

// The chat-history search, as a centred dialog (like ChatGPT's): with nothing typed it lists
// "Last opened" (the chat you were on before this one) and "Recent chats"; typing runs the
// full-text search over every question and answer (server-side, debounced) and lists the hits
// with the matching snippet. Arrow keys move, Enter opens, Escape / outside click closes.

type T = (key: UIKey, vars?: Record<string, string | number>) => string;

const OPENED_KEY = "gmr-chat.recentlyOpened";
const OPENED_MAX = 10;
/** Result rows drawn per step. */
const ROW_STEP = 50;

function readOpened(): string[] {
  try {
    const raw = window.localStorage.getItem(OPENED_KEY);
    const v = raw ? JSON.parse(raw) : [];
    return Array.isArray(v) ? v.filter((x) => typeof x === "string") : [];
  } catch {
    return [];
  }
}

/** Remember that a chat was opened (most recent first) -- feeds "Last opened". */
export function recordOpened(sessionId: string | null) {
  if (!sessionId) return;
  const next = [sessionId, ...readOpened().filter((id) => id !== sessionId)].slice(0, OPENED_MAX);
  try {
    window.localStorage.setItem(OPENED_KEY, JSON.stringify(next));
  } catch {
    // Non-fatal -- "Last opened" just isn't shown.
  }
}

/** "3:45 PM" today / "Yesterday" / "12 Nov" -- shown at the right of a row. */
function shortDate(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  const now = new Date();
  const days = Math.floor((new Date(now.toDateString()).getTime() - new Date(d.toDateString()).getTime()) / 86400000);
  if (days <= 0) return d.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
  if (days === 1) return "Yesterday";
  return d.toLocaleDateString(undefined, { day: "numeric", month: "short", year: d.getFullYear() === now.getFullYear() ? undefined : "numeric" });
}

/** The text with every query word wrapped in <mark>. */
function Highlight({ text, query }: { text: string; query: string }) {
  const words = query.toLowerCase().split(/\s+/).filter(Boolean);
  if (!words.length || !text) return <>{text}</>;
  const escaped = words.map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const parts = text.split(new RegExp(`(${escaped.join("|")})`, "gi"));
  return (
    <>
      {parts.map((p, i) => (words.includes(p.toLowerCase()) ? <mark key={i}>{p}</mark> : <React.Fragment key={i}>{p}</React.Fragment>))}
    </>
  );
}

interface ChatSearchDialogProps {
  open: boolean;
  sessions: SessionSummary[];
  folders: ChatFolder[];
  activeId: string | null;
  t: T;
  /** query = the search that found it, so the chat can scroll to the matching message */
  onOpen: (sessionId: string, query?: string) => void;
  onClose: () => void;
}

type Row = { s: SessionSummary; hit?: SessionSearchHit };

export const ChatSearchDialog: React.FC<ChatSearchDialogProps> = ({ open, sessions, folders, activeId, t, onOpen, onClose }) => {
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<SessionSearchHit[] | null>(null);
  const [searching, setSearching] = useState(false);
  const [cursor, setCursor] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  // Fresh each time it opens.
  useEffect(() => {
    if (!open) return;
    setQuery("");
    setHits(null);
    setCursor(0);
    window.setTimeout(() => inputRef.current?.focus(), 0);
  }, [open]);

  // Full-text search: debounced, last request wins.
  const seqRef = useRef(0);
  useEffect(() => {
    const q = query.trim();
    setCursor(0);
    if (!q) {
      seqRef.current++;
      setHits(null);
      setSearching(false);
      return;
    }
    const seq = ++seqRef.current;
    setSearching(true);
    const timer = window.setTimeout(() => {
      apiClient
        .searchSessions(q)
        .then((res) => seq === seqRef.current && setHits(res))
        .catch(() => seq === seqRef.current && setHits(null))
        .finally(() => seq === seqRef.current && setSearching(false));
    }, 250);
    return () => window.clearTimeout(timer);
  }, [query]);

  const q = query.trim();
  const folderName = (id: string | null | undefined) => (id ? folders.find((f) => f.folder_id === id)?.name : undefined);

  // Nothing typed: "Last opened" + "Recent chats" (archived chats are left out).
  const { lastOpened, recent } = useMemo(() => {
    const live = sessions.filter((s) => !s.archived);
    const byId = new Map(live.map((s) => [s.session_id, s]));
    const lastId = open ? readOpened().find((id) => id !== activeId && byId.has(id)) : undefined;
    const lastOpened = lastId ? byId.get(lastId)! : null;
    const recent = [...live]
      .filter((s) => s.session_id !== lastId)
      .sort((a, b) => b.updated_at.localeCompare(a.updated_at));
    return { lastOpened, recent };
  }, [sessions, activeId, open]);

  // Typed: server hits; until they arrive, titles/last questions filtered locally.
  const results: SessionSearchHit[] = useMemo(() => {
    if (!q) return [];
    if (hits) {
      const byId = new Map(sessions.map((s) => [s.session_id, s]));
      return hits.filter((h) => byId.has(h.session_id)).map((h) => ({ ...h, ...byId.get(h.session_id)! }));
    }
    const words = q.toLowerCase().split(/\s+/);
    return sessions
      .filter((s) => words.every((w) => `${s.session_name} ${s.last_question}`.toLowerCase().includes(w)))
      .map((s) => ({ ...s, match_snippet: s.last_question, match_role: "user" as const }));
  }, [q, hits, sessions]);

  const rows: Row[] = q
    ? results.map((h) => ({ s: h, hit: h }))
    : [...(lastOpened ? [{ s: lastOpened }] : []), ...recent.map((s) => ({ s }))];

  // Rows are drawn in steps (thousands of chats / hundreds of hits froze the dialog for
  // seconds when all were drawn at once): the next step comes when the end of the list
  // scrolls into view, or when the arrow keys move past it.
  const [limit, setLimit] = useState(ROW_STEP);
  useEffect(() => setLimit(ROW_STEP), [q, open]);
  const drawn = Math.min(rows.length, Math.max(limit, cursor + 1));
  const moreRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = moreRef.current;
    if (!el || drawn >= rows.length || typeof IntersectionObserver === "undefined") return;
    const io = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) setLimit((n) => Math.max(n, drawn) + ROW_STEP);
      },
      { root: listRef.current, rootMargin: "200px" }
    );
    io.observe(el);
    return () => io.disconnect();
  }, [drawn, rows.length, open]);

  // Keep the highlighted row in view.
  useEffect(() => {
    listRef.current?.querySelector<HTMLElement>(`[data-row="${cursor}"]`)?.scrollIntoView({ block: "nearest" });
  }, [cursor]);

  if (!open) return null;

  const choose = (row: Row | undefined) => {
    if (!row) return;
    onOpen(row.s.session_id, row.hit ? q : undefined);
    onClose();
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Escape") {
      e.preventDefault();
      onClose();
    } else if (e.key === "ArrowDown") {
      e.preventDefault();
      setCursor((c) => Math.min(rows.length - 1, c + 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setCursor((c) => Math.max(0, c - 1));
    } else if (e.key === "Enter") {
      e.preventDefault();
      choose(rows[cursor]);
    }
  };

  const renderRow = (row: Row, index: number) => {
    const { s, hit } = row;
    const folder = folderName(s.folder_id);
    return (
      <button
        key={`${hit ? "hit" : "row"}:${s.session_id}`}
        type="button"
        role="option"
        aria-selected={index === cursor}
        data-row={index}
        className={`chat-search-row${index === cursor ? " active" : ""}${s.session_id === activeId ? " current" : ""}`}
        onMouseMove={() => index !== cursor && setCursor(index)}
        onClick={() => choose(row)}
      >
        <span className="chat-search-row-icon">{s.archived ? <Archive size={18} /> : <MessageCircle size={18} />}</span>
        <span className="chat-search-row-body">
          <span className="chat-search-row-title">
            {hit ? <Highlight text={s.session_name || "New Session"} query={q} /> : s.session_name && s.session_name !== "New Session" ? s.session_name : t("newSession")}
          </span>
          {hit && hit.match_snippet ? (
            <span className="chat-search-row-snippet">
              <Highlight text={hit.match_snippet} query={q} />
            </span>
          ) : null}
        </span>
        <span className="chat-search-row-meta">
          {folder ? (
            <span className="chat-search-row-folder">
              <Folder size={11} /> {folder}
            </span>
          ) : null}
          {hit && s.archived ? <span>{t("archivedTag")}</span> : null}
          <span className="chat-search-row-date">{shortDate(s.updated_at)}</span>
        </span>
      </button>
    );
  };

  let index = 0;
  return createPortal(
    <div className="chat-search-overlay" onMouseDown={onClose} role="presentation">
      <div
        className="chat-search-dialog"
        role="dialog"
        aria-modal="true"
        aria-label={t("searchHistory")}
        onMouseDown={(e) => e.stopPropagation()}
        onKeyDown={onKeyDown}
      >
        <div className="chat-search-head">
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={t("searchChats")}
            aria-label={t("searchHistory")}
            aria-controls="chat-search-list"
          />
          <button type="button" className="chat-search-close" onClick={onClose} aria-label={t("close")} title={t("close")}>
            <X size={20} />
          </button>
        </div>

        <div className="chat-search-list" id="chat-search-list" role="listbox" ref={listRef}>
          {q ? (
            <section className="chat-search-group">
              <div className="chat-search-group-label">
                {searching && !hits ? t("searching") : t("searchResults", { n: results.length })}
              </div>
              {results.length === 0 && !searching ? (
                <div className="chat-search-empty">{t("noChatsMatch", { q })}</div>
              ) : (
                rows.slice(0, drawn).map((r) => renderRow(r, index++))
              )}
            </section>
          ) : (
            <>
              {lastOpened ? (
                <section className="chat-search-group">
                  <div className="chat-search-group-label">{t("lastOpened")}</div>
                  {renderRow(rows[0], index++)}
                </section>
              ) : null}
              <section className="chat-search-group">
                <div className="chat-search-group-label">{t("recentChats")}</div>
                {recent.length === 0 ? (
                  <div className="chat-search-empty">{t("noSessions")}</div>
                ) : (
                  rows.slice(index, drawn).map((r) => renderRow(r, index++))
                )}
              </section>
            </>
          )}
          {drawn < rows.length ? <div ref={moreRef} className="chat-list-more" aria-hidden="true" /> : null}
        </div>
      </div>
    </div>,
    document.body
  );
};
