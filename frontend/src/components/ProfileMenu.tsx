import React, { useEffect, useMemo, useRef, useState } from "react";
import { Archive, ArchiveRestore, Check, ChevronRight, Languages, Settings, SunMoon, Trash2, X } from "lucide-react";
import type { SessionSummary } from "../api/client";
import type { UIKey } from "../i18n/uiStrings";
import { UI_LANGUAGE_OPTIONS, type LanguageCode } from "../i18n/languages";
import { readThemePref, setThemePref, type ThemePref } from "../utils/theme";

const THEME_OPTIONS: { value: ThemePref; label: UIKey }[] = [
  { value: "system", label: "themeSystem" },
  { value: "light", label: "themeLight" },
  { value: "dark", label: "themeDark" },
];

type T = (key: UIKey, vars?: Record<string, string | number>) => string;

// Sidebar footer: a Settings button whose menu opens the archived chats, the theme picker and
// the sidebar language (the sidebar keeps its own language; the answer language never moves it).
// Archived chats live only here -- they are deleted by the backend 30 days after archiving
// unless unarchived (or continued) first.

interface ProfileMenuProps {
  archivedCount: number;
  t: T;
  onOpenArchived: () => void;
  uiLanguage: LanguageCode;
  onUiLanguage: (language: LanguageCode) => void;
}

export const ProfileMenu: React.FC<ProfileMenuProps> = ({ archivedCount, t, onOpenArchived, uiLanguage, onUiLanguage }) => {
  const [open, setOpen] = useState(false);
  // which flyout is showing: theme or language (one at a time)
  const [sub, setSub] = useState<"theme" | "language" | null>(null);
  const themeOpen = sub === "theme";
  const langOpen = sub === "language";
  const [theme, setTheme] = useState<ThemePref>(readThemePref);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) {
      setSub(null);
      return;
    }
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
    <div className="chat-sidebar-footer chat-profile" ref={ref}>
      {open ? (
        <div className="chat-profile-menu" role="menu">
          <button
            type="button"
            role="menuitem"
            onMouseEnter={() => setSub(null)}
            onClick={() => {
              setOpen(false);
              onOpenArchived();
            }}
          >
            <Archive size={14} /> {t("archivedChatsTitle")}
            {archivedCount ? <span className="chat-profile-count">{archivedCount}</span> : null}
          </button>
          <div className="chat-profile-sub" onMouseEnter={() => setSub("theme")}>
            <button
              type="button"
              role="menuitem"
              aria-haspopup="menu"
              aria-expanded={themeOpen}
              className={themeOpen ? "is-active" : undefined}
              onClick={() => setSub((v) => (v === "theme" ? null : "theme"))}
            >
              <SunMoon size={14} /> {t("theme")}
              <ChevronRight size={14} className="chat-profile-chevron" />
            </button>
            {themeOpen ? (
              <div className="chat-profile-submenu" role="menu">
                {THEME_OPTIONS.map((o) => (
                  <button
                    key={o.value}
                    type="button"
                    role="menuitemradio"
                    aria-checked={theme === o.value}
                    onClick={() => {
                      setTheme(o.value);
                      setThemePref(o.value);
                      setOpen(false);
                    }}
                  >
                    {t(o.label)}
                    {theme === o.value ? <Check size={14} className="chat-profile-check" /> : null}
                  </button>
                ))}
              </div>
            ) : null}
          </div>
          <div className="chat-profile-sub" onMouseEnter={() => setSub("language")}>
            <button
              type="button"
              role="menuitem"
              aria-haspopup="menu"
              aria-expanded={langOpen}
              className={langOpen ? "is-active" : undefined}
              onClick={() => setSub((v) => (v === "language" ? null : "language"))}
            >
              <Languages size={14} /> {t("menuLanguage")}
              <ChevronRight size={14} className="chat-profile-chevron" />
            </button>
            {langOpen ? (
              <div className="chat-profile-submenu chat-profile-submenu-lang" role="menu">
                {UI_LANGUAGE_OPTIONS.map((o) => (
                  <button
                    key={o.value}
                    type="button"
                    role="menuitemradio"
                    aria-checked={uiLanguage === o.value}
                    onClick={() => {
                      onUiLanguage(o.value);
                      setOpen(false);
                    }}
                  >
                    {o.label}
                    {uiLanguage === o.value ? <Check size={14} className="chat-profile-check" /> : null}
                  </button>
                ))}
              </div>
            ) : null}
          </div>
        </div>
      ) : null}
      <button
        type="button"
        className="chat-profile-btn"
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
      >
        <Settings size={16} className="chat-profile-icon" aria-hidden="true" />
        <span className="chat-profile-name">{t("settings")}</span>
      </button>
    </div>
  );
};

interface ArchivedChatsDialogProps {
  open: boolean;
  sessions: SessionSummary[];
  t: T;
  /** true while another dialog (the delete confirmation) is on top -- Escape is its, not ours */
  blocked?: boolean;
  onClose: () => void;
  onOpenChat: (sessionId: string) => void;
  onUnarchive: (sessionId: string) => void;
  onDelete: (sessionId: string) => void;
}

const DAY_MS = 86_400_000;

export const ArchivedChatsDialog: React.FC<ArchivedChatsDialogProps> = ({
  open, sessions, t, blocked, onClose, onOpenChat, onUnarchive, onDelete,
}) => {
  useEffect(() => {
    if (!open || blocked) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, blocked, onClose]);

  const rows = useMemo(
    () =>
      sessions
        .filter((s) => s.archived)
        .sort((a, b) => (b.archived_at ?? "").localeCompare(a.archived_at ?? "")),
    [sessions]
  );

  if (!open) return null;

  const fmt = (iso?: string | null) =>
    iso ? new Date(iso).toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" }) : "";
  const deletesLabel = (iso?: string | null) => {
    if (!iso) return "";
    const days = Math.ceil((new Date(iso).getTime() - Date.now()) / DAY_MS);
    return days <= 0 ? t("deletesToday") : t(days === 1 ? "deletesInOne" : "deletesIn", { n: days });
  };

  return (
    <div className="confirm-overlay" onClick={onClose} role="presentation">
      <div
        className="confirm-dialog archived-dialog"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-labelledby="archived-dialog-title"
      >
        <div className="confirm-dialog-header">
          <div>
            <h3 id="archived-dialog-title" className="confirm-dialog-title">{t("archivedChatsTitle")}</h3>
            <p className="confirm-dialog-message">{t("archivedChatsHint", { days: 30 })}</p>
          </div>
          <button type="button" className="confirm-dialog-close" onClick={onClose} aria-label={t("dismiss")}>
            <X size={18} />
          </button>
        </div>
        <div className="archived-list">
          {rows.length === 0 ? (
            <div className="archived-empty">{t("noArchived")}</div>
          ) : (
            rows.map((s) => (
              <div className="archived-row" key={s.session_id}>
                <button
                  type="button"
                  className="archived-name"
                  title={s.last_question || s.session_name}
                  onClick={() => {
                    onClose();
                    onOpenChat(s.session_id);
                  }}
                >
                  <span className="archived-title">{s.session_name}</span>
                  <span className="archived-meta">
                    {s.archived_at ? t("archivedOn", { date: fmt(s.archived_at) }) : ""}
                    {s.delete_at ? <span className="archived-expiry"> · {deletesLabel(s.delete_at)}</span> : null}
                  </span>
                </button>
                <button
                  type="button"
                  className="chat-icon-btn"
                  onClick={() => onUnarchive(s.session_id)}
                  title={t("unarchive")}
                  aria-label={`${t("unarchive")}: ${s.session_name}`}
                >
                  <ArchiveRestore size={15} />
                </button>
                <button
                  type="button"
                  className="chat-icon-btn danger"
                  onClick={() => onDelete(s.session_id)}
                  title={t("delChat")}
                  aria-label={`${t("delChat")}: ${s.session_name}`}
                >
                  <Trash2 size={15} />
                </button>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};
