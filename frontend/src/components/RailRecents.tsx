import React, { useEffect, useMemo, useRef, useState } from "react";
import { MessageCircle } from "lucide-react";
import type { SessionSummary } from "../api/client";
import type { UIKey } from "../i18n/uiStrings";

type T = (key: UIKey, vars?: Record<string, string | number>) => string;

// Collapsed sidebar rail: the chat-bubble icon opens a "Recents" flyout (like ChatGPT's)
// listing the non-archived chats, newest first. Clicking one opens it and closes the flyout.

interface RailRecentsProps {
  sessions: SessionSummary[];
  activeId: string | null;
  t: T;
  onOpen: (sessionId: string) => void;
}

export const RailRecents: React.FC<RailRecentsProps> = ({ sessions, activeId, t, onOpen }) => {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  const recents = useMemo(
    () =>
      sessions
        .filter((s) => !s.archived)
        .slice()
        .sort((a, b) => (b.updated_at || "").localeCompare(a.updated_at || "")),
    [sessions],
  );

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", onDown);
    window.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      window.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <div className="chat-rail-recents" ref={ref}>
      <button
        type="button"
        className="chat-sidebar-icon chat-sidebar-rail-btn"
        aria-haspopup="menu"
        aria-expanded={open}
        aria-pressed={open}
        aria-label={t("recentSessions")}
        title={t("recentSessions")}
        onClick={() => setOpen((v) => !v)}
      >
        <MessageCircle size={18} />
      </button>
      {open ? (
        <div className="chat-rail-flyout" role="menu" aria-label={t("recentSessions")}>
          <div className="chat-rail-flyout-head">{t("recentSessions")}</div>
          <div className="chat-rail-flyout-list">
            {recents.length === 0 ? (
              <div className="chat-rail-flyout-empty">—</div>
            ) : (
              recents.map((s) => (
                <button
                  key={s.session_id}
                  type="button"
                  role="menuitem"
                  className={`chat-rail-flyout-item${s.session_id === activeId ? " active" : ""}`}
                  title={s.session_name}
                  onClick={() => {
                    setOpen(false);
                    onOpen(s.session_id);
                  }}
                >
                  {s.session_name || s.last_question}
                </button>
              ))
            )}
          </div>
        </div>
      ) : null}
    </div>
  );
};
