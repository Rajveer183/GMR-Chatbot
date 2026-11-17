import React, { useEffect, useRef, useState } from "react";
import { MessageCircle } from "lucide-react";
import type { UIKey } from "../i18n/uiStrings";

type T = (key: UIKey, vars?: Record<string, string | number>) => string;

// Collapsed sidebar rail: the chat-bubble icon opens a flyout (like ChatGPT's "Recents")
// holding the full chat history -- pinned chats, folders, recent chats and every row action
// (rename, pin, archive, move, delete) -- so the collapsed sidebar loses no feature.

interface RailRecentsProps {
  t: T;
  /** Renders the history; call close() after a chat is opened. */
  renderHistory: (close: () => void) => React.ReactNode;
}

export const RailRecents: React.FC<RailRecentsProps> = ({ t, renderHistory }) => {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      // Escape inside a rename box, row menu or confirm dialog belongs to that, not the flyout.
      const target = e.target as HTMLElement | null;
      if (target && /^(INPUT|TEXTAREA)$/.test(target.tagName)) return;
      if (ref.current?.querySelector(".chat-row-menu, [role='dialog'], [role='alertdialog']")) return;
      setOpen(false);
    };
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
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-pressed={open}
        aria-label={t("recentSessions")}
        data-tip={t("recentSessions")}
        onClick={() => setOpen((v) => !v)}
      >
        <MessageCircle size={18} />
      </button>
      {open ? (
        <div className="chat-rail-flyout" aria-label={t("recentSessions")}>
          {renderHistory(() => setOpen(false))}
        </div>
      ) : null}
    </div>
  );
};
