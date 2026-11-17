import React, { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import {
  Archive, ArchiveRestore, Check, ChevronDown, ChevronRight, Folder, FolderInput, FolderMinus, FolderOpen,
  FolderPlus, MoreHorizontal, Pencil, Pin, PinOff, Search, Trash2, X,
} from "lucide-react";
import { apiClient } from "../api/client";
import type { ChatFolder, SessionPatch, SessionSearchHit, SessionSummary } from "../api/client";
import type { UIKey } from "../i18n/uiStrings";
import { ConfirmDialog } from "./ConfirmDialog";

// The sidebar's chat history: pinned chats, folders, recent chats, archived chats, and a
// full-text search over every question and answer (server-side, debounced). All changes go
// through PATCH /api/sessions/{id} and /api/folders; the parent reloads the lists afterwards.

type T = (key: UIKey, vars?: Record<string, string | number>) => string;

interface ChatHistoryProps {
  sessions: SessionSummary[];
  folders: ChatFolder[];
  activeId: string | null;
  t: T;
  /** query = the search that found it, so the chat can scroll to the matching message */
  onOpen: (sessionId: string, query?: string) => void;
  onUpdate: (sessionId: string, patch: SessionPatch) => Promise<void>;
  onDelete: (sessionId: string) => void;
  onCreateFolder: (name: string) => Promise<ChatFolder | null>;
  /** Rename and/or pin a folder. */
  onUpdateFolder: (folderId: string, patch: { name?: string; pinned?: boolean }) => Promise<void>;
  onDeleteFolder: (folderId: string) => Promise<void>;
}

const OPEN_FOLDERS_KEY = "gmr-chat.openFolders";
const SHOW_ARCHIVED_KEY = "gmr-chat.showArchived";
const CLOSED_SECTIONS_KEY = "gmr-chat.closedSections";

type Section = "pinned" | "folders";

/** A sidebar section label ("Pinned ˅"); the chevron collapses the section. */
function SectionHead({ label, open, onToggle, action, drop }: {
  label: string; open?: boolean; onToggle?: () => void; action?: React.ReactNode; drop?: Record<string, unknown> & { className?: string };
}) {
  const { className: dropClass = "", ...dropRest } = drop ?? {};
  return (
    <div className={`chat-section-head ${dropClass}`} {...dropRest}>
      {onToggle ? (
        <button type="button" className="chat-section-toggle" onClick={onToggle} aria-expanded={open}>
          <span>{label}</span>
          {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        </button>
      ) : (
        <span className="chat-section-label">{label}</span>
      )}
      {action}
    </div>
  );
}

function readJSON<V>(key: string, fallback: V): V {
  try {
    const raw = window.localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as V) : fallback;
  } catch {
    return fallback;
  }
}

function writeJSON(key: string, value: unknown) {
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Non-fatal -- the view state just isn't remembered after a reload.
  }
}

/** "Today" / "Yesterday" / "Mon 12 Nov" -- shown under a chat's title in search results. */
function shortDate(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  const now = new Date();
  const days = Math.floor((new Date(now.toDateString()).getTime() - new Date(d.toDateString()).getTime()) / 86400000);
  if (days <= 0) return d.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
  if (days === 1) return "Yesterday";
  return d.toLocaleDateString(undefined, { day: "numeric", month: "short", year: d.getFullYear() === now.getFullYear() ? undefined : "numeric" });
}

/** The snippet with every query word wrapped in <mark>. */
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

interface MenuState {
  session: SessionSummary;
  place: string;
  top: number;
  left: number;
}

export const ChatHistory: React.FC<ChatHistoryProps> = ({
  sessions, folders, activeId, t, onOpen, onUpdate, onDelete, onCreateFolder, onUpdateFolder, onDeleteFolder,
}) => {
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<SessionSearchHit[] | null>(null);
  const [searching, setSearching] = useState(false);
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");
  // A chat or folder can be listed twice (under Pinned and in its usual place); the rename
  // input opens only in the row it was started from.
  const [renamePlace, setRenamePlace] = useState<string>("");
  const [renamingFolderId, setRenamingFolderId] = useState<string | null>(null);
  const [newFolderFor, setNewFolderFor] = useState<string | null | undefined>(undefined); // undefined = closed; string = move that chat in
  const [newFolderName, setNewFolderName] = useState("");
  const [openFolders, setOpenFolders] = useState<string[]>(() => readJSON<string[]>(OPEN_FOLDERS_KEY, []));
  const [showArchived, setShowArchived] = useState<boolean>(() => readJSON<boolean>(SHOW_ARCHIVED_KEY, false));
  const [closedSections, setClosedSections] = useState<Section[]>(() => readJSON<Section[]>(CLOSED_SECTIONS_KEY, []));
  const sectionOpen = (s: Section) => !closedSections.includes(s);
  const toggleSection = (s: Section) =>
    setClosedSections((prev) => {
      const next = prev.includes(s) ? prev.filter((x) => x !== s) : [...prev, s];
      writeJSON(CLOSED_SECTIONS_KEY, next);
      return next;
    });
  const [menu, setMenu] = useState<MenuState | null>(null);
  const [confirmFolder, setConfirmFolder] = useState<ChatFolder | null>(null);
  const [dropTarget, setDropTarget] = useState<string | null>(null);
  const menuRef = useRef<HTMLDivElement>(null);

  // Full-text search: debounced, last request wins.
  const searchSeq = useRef(0);
  useEffect(() => {
    const q = query.trim();
    if (!q) {
      setHits(null);
      setSearching(false);
      return;
    }
    const seq = ++searchSeq.current;
    setSearching(true);
    const timer = window.setTimeout(() => {
      apiClient
        .searchSessions(q)
        .then((res) => seq === searchSeq.current && setHits(res))
        .catch(() => seq === searchSeq.current && setHits(null))
        .finally(() => seq === searchSeq.current && setSearching(false));
    }, 250);
    return () => window.clearTimeout(timer);
  }, [query]);

  // The search results also follow renames/pins made while searching.
  useEffect(() => {
    if (!hits) return;
    const byId = new Map(sessions.map((s) => [s.session_id, s]));
    setHits((prev) => prev && prev.filter((h) => byId.has(h.session_id)).map((h) => ({ ...h, ...byId.get(h.session_id)! })));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessions]);

  // Close the row menu on outside click / Escape / scroll.
  useEffect(() => {
    if (!menu) return;
    const close = (e: Event) => {
      if (e instanceof KeyboardEvent && e.key !== "Escape") return;
      if (e.type === "mousedown" && menuRef.current?.contains(e.target as Node)) return;
      setMenu(null);
    };
    window.addEventListener("mousedown", close);
    window.addEventListener("keydown", close);
    window.addEventListener("resize", close);
    document.addEventListener("scroll", close, true);
    return () => {
      window.removeEventListener("mousedown", close);
      window.removeEventListener("keydown", close);
      window.removeEventListener("resize", close);
      document.removeEventListener("scroll", close, true);
    };
  }, [menu]);

  // Keep the menu inside the window.
  useLayoutEffect(() => {
    const el = menuRef.current;
    if (!menu || !el) return;
    const r = el.getBoundingClientRect();
    const top = Math.max(8, Math.min(menu.top, window.innerHeight - r.height - 8));
    const left = Math.max(8, Math.min(menu.left, window.innerWidth - r.width - 8));
    if (top !== menu.top || left !== menu.left) setMenu({ ...menu, top, left });
  }, [menu]);

  const toggleFolder = (id: string) =>
    setOpenFolders((prev) => {
      const next = prev.includes(id) ? prev.filter((f) => f !== id) : [...prev, id];
      writeJSON(OPEN_FOLDERS_KEY, next);
      return next;
    });

  const toggleArchived = () =>
    setShowArchived((v) => {
      writeJSON(SHOW_ARCHIVED_KEY, !v);
      return !v;
    });

  const { pinned, byFolder, recent, archived } = useMemo(() => {
    const live = sessions.filter((s) => !s.archived);
    const folderIds = new Set(folders.map((f) => f.folder_id));
    // A pinned chat that is in a folder is listed in both places -- it is the same chat,
    // so whatever happens to it (rename, new answers) shows in both rows.
    const byFolder = new Map<string, SessionSummary[]>();
    for (const s of live) {
      if (s.folder_id && folderIds.has(s.folder_id)) {
        byFolder.set(s.folder_id, [...(byFolder.get(s.folder_id) ?? []), s]);
      }
    }
    type PinnedEntry = { kind: "chat"; at: string; chat: SessionSummary } | { kind: "folder"; at: string; folder: ChatFolder };
    const pinned: PinnedEntry[] = [
      ...live.filter((s) => s.pinned).map((chat) => ({ kind: "chat" as const, at: chat.pinned_at ?? "", chat })),
      ...folders.filter((f) => f.pinned).map((folder) => ({ kind: "folder" as const, at: folder.pinned_at ?? "", folder })),
    ].sort((a, b) => b.at.localeCompare(a.at));
    return {
      pinned,
      byFolder,
      recent: live.filter((s) => !s.pinned && !(s.folder_id && folderIds.has(s.folder_id))),
      archived: sessions.filter((s) => s.archived),
    };
  }, [sessions, folders]);

  const folderName = (id: string | null | undefined) => folders.find((f) => f.folder_id === id)?.name;

  const commitRename = async () => {
    const id = renamingId;
    setRenamingId(null);
    const name = renameValue.trim();
    if (!id || !name || sessions.find((s) => s.session_id === id)?.session_name === name) return;
    await onUpdate(id, { session_name: name });
  };

  const commitFolderRename = async () => {
    const id = renamingFolderId;
    setRenamingFolderId(null);
    const name = renameValue.trim();
    if (id && name && folderName(id) !== name) await onUpdateFolder(id, { name });
  };

  const commitNewFolder = async () => {
    const target = newFolderFor;
    setNewFolderFor(undefined);
    const name = newFolderName.trim();
    setNewFolderName("");
    if (!name) return;
    const folder = await onCreateFolder(name);
    if (!folder) return;
    setOpenFolders((prev) => {
      const next = prev.includes(folder.folder_id) ? prev : [...prev, folder.folder_id];
      writeJSON(OPEN_FOLDERS_KEY, next);
      return next;
    });
    if (target) await onUpdate(target, { folder_id: folder.folder_id });
  };

  const openMenu = (e: React.MouseEvent, session: SessionSummary, place: string) => {
    e.stopPropagation();
    const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
    const same = menu?.session.session_id === session.session_id && menu.place === place;
    setMenu(same ? null : { session, place, top: r.bottom + 4, left: r.left - 150 });
  };

  const act = (fn: () => void) => () => {
    setMenu(null);
    fn();
  };

  const onDrop = (e: React.DragEvent, folderId: string | null) => {
    e.preventDefault();
    setDropTarget(null);
    const id = e.dataTransfer.getData("text/x-gmr-session");
    const s = sessions.find((x) => x.session_id === id);
    if (s && (s.folder_id ?? null) !== folderId) onUpdate(id, { folder_id: folderId });
  };

  const dropProps = (key: string, folderId: string | null): Record<string, any> => ({
    onDragOver: (e: React.DragEvent) => {
      if (!e.dataTransfer.types.includes("text/x-gmr-session")) return;
      e.preventDefault();
      setDropTarget(key);
    },
    onDragLeave: () => setDropTarget((d) => (d === key ? null : d)),
    onDrop: (e: React.DragEvent) => onDrop(e, folderId),
  });

  const renderItem = (s: SessionSummary, hit?: SessionSearchHit, place = "list") => {
    const renaming = renamingId === s.session_id && renamePlace === place;
    return (
    <div
      key={`${place}:${s.session_id}`}
      className={`chat-session-item${activeId === s.session_id ? " active" : ""}${hit ? " is-hit" : ""}${s.archived ? " is-archived" : ""}${s.pinned ? " is-pinned" : ""}`}
      title={s.session_name}
      draggable={!hit && !renaming}
      onDragStart={(e) => {
        e.dataTransfer.setData("text/x-gmr-session", s.session_id);
        e.dataTransfer.effectAllowed = "move";
      }}
    >
      {renaming ? (
        <input
          className="chat-session-rename"
          value={renameValue}
          onChange={(e) => setRenameValue(e.target.value)}
          onBlur={commitRename}
          onKeyDown={(e) => {
            if (e.key === "Enter") commitRename();
            if (e.key === "Escape") setRenamingId(null);
          }}
          maxLength={80}
          autoFocus
        />
      ) : (
        <button className="chat-session-name" type="button" onClick={() => onOpen(s.session_id, hit ? query.trim() : undefined)}>
          <span className="chat-session-title-row">
            <span className="chat-session-title">{hit ? <Highlight text={s.session_name || "New Session"} query={query} /> : s.session_name || "New Session"}</span>
          </span>
          {hit ? (
            <>
              {hit.match_snippet ? (
                <span className="chat-session-snippet">
                  <Highlight text={hit.match_snippet} query={query} />
                </span>
              ) : null}
              <span className="chat-session-meta">
                {shortDate(s.updated_at)}
                {s.folder_id && folderName(s.folder_id) ? ` · ${folderName(s.folder_id)}` : ""}
                {s.archived ? ` · ${t("archivedTag")}` : ""}
              </span>
            </>
          ) : null}
        </button>
      )}
      <div className="chat-session-actions">
        <button
          className={`chat-icon-btn chat-session-quickpin${s.pinned ? " on" : ""}`}
          type="button"
          onClick={() => onUpdate(s.session_id, { pinned: !s.pinned })}
          title={t(s.pinned ? "unpin" : "pin")}
          aria-label={t(s.pinned ? "unpin" : "pin")}
        >
          {/* A pinned chat keeps a filled pin on screen until it is unpinned (hover shows "unpin"). */}
          <Pin size={13} className="pin-on" fill={s.pinned ? "currentColor" : "none"} />
          {s.pinned ? <PinOff size={13} className="pin-off" /> : null}
        </button>
        <button
          className="chat-icon-btn"
          type="button"
          onClick={(e) => openMenu(e, s, place)}
          title={t("moreActions")}
          aria-label={t("moreActions")}
          aria-haspopup="menu"
          aria-expanded={menu?.session.session_id === s.session_id && menu.place === place}
        >
          <MoreHorizontal size={13} />
        </button>
      </div>
    </div>
    );
  };

  /** A folder row and (when open) its chats. place "pinned" = its copy under Pinned, which
   * opens and closes separately from the one under Folders. */
  const renderFolder = (f: ChatFolder, place: "folders" | "pinned") => {
    const openKey = place === "pinned" ? `pin:${f.folder_id}` : f.folder_id;
    const open = openFolders.includes(openKey);
    const items = byFolder.get(f.folder_id) ?? [];
    const dropKey = `${place}:${f.folder_id}`;
    const renaming = renamingFolderId === f.folder_id && renamePlace === place;
    return (
      <div key={`${place}:${f.folder_id}`} className="chat-folder">
        <div
          className={`chat-folder-row${dropTarget === dropKey ? " drop" : ""}${f.pinned ? " is-pinned" : ""}`}
          {...dropProps(dropKey, f.folder_id)}
        >
          {renaming ? (
            <>
              <Folder size={13} />
              <input
                className="chat-session-rename"
                value={renameValue}
                onChange={(e) => setRenameValue(e.target.value)}
                onBlur={commitFolderRename}
                onKeyDown={(e) => {
                  if (e.key === "Enter") commitFolderRename();
                  if (e.key === "Escape") setRenamingFolderId(null);
                }}
                maxLength={60}
                autoFocus
              />
            </>
          ) : (
            <button type="button" className="chat-folder-toggle" onClick={() => toggleFolder(openKey)} aria-expanded={open}>
              {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
              {open ? <FolderOpen size={13} /> : <Folder size={13} />}
              <span className="chat-folder-name">{f.name}</span>
              <span className="chat-folder-count">{items.length}</span>
            </button>
          )}
          <div className="chat-session-actions">
            <button
              className={`chat-icon-btn chat-session-quickpin${f.pinned ? " on" : ""}`}
              type="button"
              onClick={() => onUpdateFolder(f.folder_id, { pinned: !f.pinned })}
              title={t(f.pinned ? "unpin" : "pin")}
              aria-label={t(f.pinned ? "unpin" : "pin")}
            >
              <Pin size={12} className="pin-on" fill={f.pinned ? "currentColor" : "none"} />
              {f.pinned ? <PinOff size={12} className="pin-off" /> : null}
            </button>
            <button
              className="chat-icon-btn"
              type="button"
              onClick={() => {
                setRenameValue(f.name);
                setRenamePlace(place);
                setRenamingFolderId(f.folder_id);
              }}
              title={t("renameFolder")}
              aria-label={t("renameFolder")}
            >
              <Pencil size={11} />
            </button>
            <button className="chat-icon-btn danger" type="button" onClick={() => setConfirmFolder(f)} title={t("deleteFolder")} aria-label={t("deleteFolder")}>
              <Trash2 size={11} />
            </button>
          </div>
        </div>
        {open ? (
          <div className="chat-folder-items">
            {items.length ? items.map((s) => renderItem(s, undefined, dropKey)) : <div className="chat-folders-hint">{t("folderEmpty")}</div>}
          </div>
        ) : null}
      </div>
    );
  };

  const newFolderInput = (
    <div className="chat-folder-row new">
      <FolderPlus size={13} />
      <input
        className="chat-session-rename"
        value={newFolderName}
        placeholder={t("folderName")}
        onChange={(e) => setNewFolderName(e.target.value)}
        onBlur={commitNewFolder}
        onKeyDown={(e) => {
          if (e.key === "Enter") commitNewFolder();
          if (e.key === "Escape") {
            setNewFolderFor(undefined);
            setNewFolderName("");
          }
        }}
        maxLength={60}
        autoFocus
      />
    </div>
  );

  const q = query.trim();
  // Until the server answers, filter titles/last questions locally so typing feels instant.
  const localHits: SessionSearchHit[] = useMemo(() => {
    if (!q) return [];
    const words = q.toLowerCase().split(/\s+/);
    return sessions
      .filter((s) => words.every((w) => `${s.session_name} ${s.last_question}`.toLowerCase().includes(w)))
      .map((s) => ({ ...s, match_snippet: s.last_question, match_role: "user" as const }));
  }, [q, sessions]);
  const results = hits ?? localHits;

  return (
    <div className="chat-history">
      <div className="chat-session-search">
        <Search size={13} />
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Escape" && setQuery("")}
          placeholder={t("searchHistory")}
          aria-label={t("searchHistory")}
        />
        {query ? (
          <button type="button" className="chat-session-search-clear" onClick={() => setQuery("")} aria-label={t("clearSearch")} title={t("clearSearch")}>
            <X size={12} />
          </button>
        ) : null}
      </div>

      <div className="chat-sessions-list">
        {q ? (
          <>
            <SectionHead label={searching && !hits ? t("searching") : t("searchResults", { n: results.length })} />
            {results.length === 0 && !searching ? (
              <div className="chat-section-empty">{t("noChatsMatch", { q })}</div>
            ) : (
              results.map((h) => renderItem(h, h, "search"))
            )}
          </>
        ) : (
          <>
            {pinned.length ? (
              <section className="chat-section">
                <SectionHead label={t("pinnedChats")} open={sectionOpen("pinned")} onToggle={() => toggleSection("pinned")} />
                {sectionOpen("pinned")
                  ? pinned.map((p) => (p.kind === "chat" ? renderItem(p.chat, undefined, "pinned") : renderFolder(p.folder, "pinned")))
                  : null}
              </section>
            ) : null}

            <section className="chat-section">
            <SectionHead
              label={t("folders")}
              open={sectionOpen("folders")}
              onToggle={() => toggleSection("folders")}
              action={
                <button
                  type="button"
                  className="chat-icon-btn chat-section-action"
                  onClick={() => {
                    if (!sectionOpen("folders")) toggleSection("folders");
                    setNewFolderName("");
                    setNewFolderFor(null);
                  }}
                  title={t("newFolder")}
                  aria-label={t("newFolder")}
                >
                  <FolderPlus size={14} />
                </button>
              }
            />
            {sectionOpen("folders") ? (
            <>
            {newFolderFor === null ? newFolderInput : null}
            {folders.length === 0 && newFolderFor === undefined ? (
              <div className="chat-section-empty" title={t("foldersHint")}>{t("noFolders")}</div>
            ) : null}
            {folders.map((f) => renderFolder(f, "folders"))}
            </>
            ) : null}
            </section>

            <section className="chat-section">
              <SectionHead
                label={t("recentSessions")}
                drop={{ ...dropProps("recent", null), className: dropTarget === "recent" ? "drop" : "" }}
              />
              {recent.length === 0 ? <div className="chat-section-empty">{t("noSessions")}</div> : recent.map((s) => renderItem(s))}
            </section>

            {archived.length ? (
              <section className="chat-section">
                <SectionHead label={t("archivedChats", { n: archived.length })} open={showArchived} onToggle={toggleArchived} />
                {showArchived ? archived.map((s) => renderItem(s)) : null}
              </section>
            ) : null}
          </>
        )}
      </div>

      {menu ? (
        <div className="chat-row-menu" ref={menuRef} role="menu" style={{ top: menu.top, left: menu.left }}>
          <button type="button" role="menuitem" onClick={act(() => onUpdate(menu.session.session_id, { pinned: !menu.session.pinned }))}>
            {menu.session.pinned ? <PinOff size={13} /> : <Pin size={13} />} {t(menu.session.pinned ? "unpin" : "pin")}
          </button>
          <button
            type="button"
            role="menuitem"
            onClick={act(() => {
              setRenameValue(menu.session.session_name);
              setRenamePlace(menu.place);
              setRenamingId(menu.session.session_id);
              setQuery("");
            })}
          >
            <Pencil size={13} /> {t("rename")}
          </button>
          <div className="chat-row-menu-label">
            <FolderInput size={12} /> {t("moveTo")}
          </div>
          {folders.map((f) => (
            <button key={f.folder_id} type="button" role="menuitemradio" aria-checked={menu.session.folder_id === f.folder_id} className="indent"
                    onClick={act(() => onUpdate(menu.session.session_id, { folder_id: f.folder_id }))}>
              {menu.session.folder_id === f.folder_id ? <Check size={12} /> : <Folder size={12} />} {f.name}
            </button>
          ))}
          {menu.session.folder_id ? (
            <button type="button" role="menuitem" className="indent" onClick={act(() => onUpdate(menu.session.session_id, { folder_id: null }))}>
              <FolderMinus size={12} /> {t("removeFromFolder")}
            </button>
          ) : null}
          <button
            type="button"
            role="menuitem"
            className="indent"
            onClick={act(() => {
              setQuery("");
              setNewFolderName("");
              setNewFolderFor(menu.session.session_id);
            })}
          >
            <FolderPlus size={12} /> {t("newFolder")}…
          </button>
          <div className="chat-row-menu-sep" />
          <button type="button" role="menuitem" onClick={act(() => onUpdate(menu.session.session_id, { archived: !menu.session.archived }))}>
            {menu.session.archived ? <ArchiveRestore size={13} /> : <Archive size={13} />} {t(menu.session.archived ? "unarchive" : "archive")}
          </button>
          <button type="button" role="menuitem" className="danger" onClick={act(() => onDelete(menu.session.session_id))}>
            <Trash2 size={13} /> {t("del")}
          </button>
        </div>
      ) : null}

      {typeof newFolderFor === "string" ? (
        <div className="chat-newfolder-for">
          <div className="chat-section-label">{t("newFolderFor",{ name: sessions.find((s) => s.session_id === newFolderFor)?.session_name ?? "" })}</div>
          {newFolderInput}
        </div>
      ) : null}

      <ConfirmDialog
        open={confirmFolder !== null}
        title={t("deleteFolder")}
        message={t("deleteFolderMessage", { name: confirmFolder?.name ?? "" })}
        confirmLabel={t("del")}
        cancelLabel={t("cancel")}
        onCancel={() => setConfirmFolder(null)}
        onConfirm={async () => {
          const f = confirmFolder;
          setConfirmFolder(null);
          if (f) await onDeleteFolder(f.folder_id);
        }}
      />
    </div>
  );
};
