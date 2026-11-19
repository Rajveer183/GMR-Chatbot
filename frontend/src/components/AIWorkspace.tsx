import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { PanelLeft, Search, SquarePen, MessageCircleDashed, Pencil, X, User, Download, History, RefreshCw, ArchiveRestore } from "lucide-react";
import { apiClient, QueryStoppedError } from "../api/client";
import type { FeedbackRating } from "../api/client";
import type { AIQueryResponse, AIStreamEvent, ChatFolder, DatasetSummary, FilterRequest, SessionPatch, SessionSummary, SessionDetail, SessionMessage, TranslatedAnswer } from "../api/client";
import { LanguageSelect } from "./LanguageSelect";
import { langTag, readStoredLanguage, screenLanguage, sessionLabel, speechLocale, statusText, storeLanguage, uiText } from "../i18n/languages";
import type { LanguagePreference } from "../i18n/languages";
import { ResultRenderer } from "./ResultRenderer";
import { LoadingStatus } from "./LoadingStatus";
import { TypingIndicator } from "./TypingIndicator";
import { ConfirmDialog } from "./ConfirmDialog";
import { ChatHistory } from "./ChatHistory";
import { RailRecents } from "./RailRecents";
import { ChatSearchDialog, recordOpened } from "./ChatSearchDialog";
import { ArchivedChatsDialog, ProfileMenu } from "./ProfileMenu";
import { EmptyState } from "./EmptyState";
import { CopyButton } from "./CopyButton";
import { SourcesButton } from "./SourcesNote";
import { ChatInput } from "./ChatInput";
import { modeTransition, originOf, runNow } from "../utils/modeTransition";
import type { Wrap } from "../utils/modeTransition";
import type { ChatInputHandle } from "./ChatInput";
import gmrLogo from "../assets/gmr_logo.png";
import { downloadSnapshotHtml, prepareSnapshotFonts, renderSnapshotPdf, renderSnapshotPng, triggerDownload } from "../utils/snapshot";
import type { SnapshotPart } from "../utils/snapshot";

const LOADING_STATUS_MESSAGES = [
  "Analyzing with backend tools...",
  "Resolving entity matches...",
  "Calculating metrics from workbook...",
  "Preparing charts and tables...",
];

// The result table has no pagination controls, so every query asks for the
// whole result set as a single page and the table scrolls instead. This is not
// unbounded: the executor already caps how many rows a plan may return
// (plan["limit"], hard-capped at 500 in backend/agent/executor.py), so this
// only has to be larger than that cap.
const ALL_ROWS = 1000;

type ScrollIntent = "none" | "new-response" | "session-open" | "search-hit";

// "Continue previous analysis": a chat whose last message is older than this opens with a
// banner saying follow-ups still use its context, plus a one-click re-run on today's data.
const CONTINUE_AFTER_MS = 30 * 60 * 1000;
const SHOW_CONTINUE_BANNER = false;

function longDate(iso: string | undefined): string {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  return d.toLocaleString(undefined, { day: "numeric", month: "short", year: "numeric", hour: "numeric", minute: "2-digit" });
}

type SnapshotFormat = "png" | "pdf" | "html";
const SNAPSHOT_FORMATS: SnapshotFormat[] = ["pdf", "png", "html"];
const SNAPSHOT_LABEL_KEYS = { pdf: "exportPdf", png: "exportPng", html: "exportWebPage" } as const;

/** Edit-and-resubmit: drop the chat from message `truncateTo` on (in the backend too) before asking. */
interface RunOptions {
  truncateTo?: number;
}

function newRequestId(): string {
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

// The session that was open when the page was last used, so a reload lands
// back in the same chat. localStorage can throw (private mode, blocked site
// data) -- that just means falling back to the most recent session.
const ACTIVE_SESSION_KEY = "gmr-chat.activeSessionId";

function readStoredSessionId(): string | null {
  try {
    return window.localStorage.getItem(ACTIVE_SESSION_KEY);
  } catch {
    return null;
  }
}

function storeSessionId(sessionId: string | null) {
  try {
    if (sessionId) window.localStorage.setItem(ACTIVE_SESSION_KEY, sessionId);
    else window.localStorage.removeItem(ACTIVE_SESSION_KEY);
  } catch {
    // Non-fatal -- the session just won't be restored after a reload.
  }
}

// Temporary chats not yet confirmed deleted on the server. Kept so a refresh or a closed tab
// that lost its DELETE still clears them on the next load (and they never show in the list).
const PENDING_TEMP_KEY = "gmr-chat.pendingTemporaryChats";

function readPendingTemp(): string[] {
  try {
    const v = JSON.parse(window.localStorage.getItem(PENDING_TEMP_KEY) || "[]");
    return Array.isArray(v) ? v.filter((x) => typeof x === "string") : [];
  } catch {
    return [];
  }
}

function writePendingTemp(ids: string[]) {
  try {
    if (ids.length) window.localStorage.setItem(PENDING_TEMP_KEY, JSON.stringify(ids));
    else window.localStorage.removeItem(PENDING_TEMP_KEY);
  } catch {
    // Non-fatal -- the server still drops idle temporary chats on its own.
  }
}

// Sidebar collapsed to an icon rail (the menu button next to the logo); remembered across reloads.
const SIDEBAR_KEY = "gmr-chat.sidebarCollapsed";

function readSidebarCollapsed(): boolean {
  try {
    return window.localStorage.getItem(SIDEBAR_KEY) === "1";
  } catch {
    return false;
  }
}

function storeSidebarCollapsed(collapsed: boolean) {
  try {
    window.localStorage.setItem(SIDEBAR_KEY, collapsed ? "1" : "0");
  } catch {
    // Non-fatal -- the sidebar just opens expanded after a reload.
  }
}

// Expanded sidebar width, dragged by the handle on its right edge (like ChatGPT); remembered across reloads.
const SIDEBAR_WIDTH_KEY = "gmr-chat.sidebarWidth";
const SIDEBAR_WIDTH_DEFAULT = 260;
const SIDEBAR_WIDTH_MIN = 200;
const SIDEBAR_WIDTH_MAX = 480;

function clampSidebarWidth(w: number): number {
  return Math.round(Math.min(SIDEBAR_WIDTH_MAX, Math.max(SIDEBAR_WIDTH_MIN, w)));
}

function readSidebarWidth(): number {
  try {
    const v = Number(window.localStorage.getItem(SIDEBAR_WIDTH_KEY));
    return v ? clampSidebarWidth(v) : SIDEBAR_WIDTH_DEFAULT;
  } catch {
    return SIDEBAR_WIDTH_DEFAULT;
  }
}

function storeSidebarWidth(w: number) {
  try {
    window.localStorage.setItem(SIDEBAR_WIDTH_KEY, String(w));
  } catch {
    // Non-fatal -- the sidebar just opens at the default width after a reload.
  }
}

function scrollToElementStart(element: HTMLElement | null | undefined) {
  element?.scrollIntoView({ behavior: "smooth", block: "start" });
}

// Renders as "10:42 AM" for a message sent today, and "Jul 28, 10:42 AM"
// for anything older -- so the meta row stays useful once a session spans
// more than one day without cluttering today's messages with a full date.
function formatMessageTime(iso: string | undefined): string {
  if (!iso) return "";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "";
  const now = new Date();
  const sameDay =
    date.getFullYear() === now.getFullYear() && date.getMonth() === now.getMonth() && date.getDate() === now.getDate();
  const time = date.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
  if (sameDay) return time;
  const day = date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
  return `${day}, ${time}`;
}

interface AIWorkspaceProps {
  filters: FilterRequest;
}

export const AIWorkspace: React.FC<AIWorkspaceProps> = ({ filters }) => {
  const [sidebarCollapsed, setSidebarCollapsed] = useState<boolean>(readSidebarCollapsed);
  const toggleSidebar = () =>
    setSidebarCollapsed((c) => {
      storeSidebarCollapsed(!c);
      return !c;
    });
  const [sidebarWidth, setSidebarWidth] = useState<number>(readSidebarWidth);
  const layoutRef = useRef<HTMLDivElement>(null);
  const setAndStoreSidebarWidth = (w: number) => {
    const next = clampSidebarWidth(w);
    setSidebarWidth(next);
    storeSidebarWidth(next);
  };
  // Drag writes the CSS variable straight onto the layout element (once per animation frame) so
  // the big chat tree is not re-rendered on every pointer move; React state is set once on release.
  const startSidebarResize = (e: React.PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    const layout = layoutRef.current;
    if (!layout) return;
    e.preventDefault();
    const startX = e.clientX;
    const startW = sidebarWidth;
    let latest = startW;
    let frame = 0;
    layout.classList.add("sidebar-resizing");
    const onMove = (ev: PointerEvent) => {
      latest = clampSidebarWidth(startW + ev.clientX - startX);
      if (frame) return;
      frame = window.requestAnimationFrame(() => {
        frame = 0;
        layout.style.setProperty("--chat-sidebar-width", `${latest}px`);
      });
    };
    const onUp = () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("pointercancel", onUp);
      if (frame) window.cancelAnimationFrame(frame);
      layout.style.setProperty("--chat-sidebar-width", `${latest}px`);
      layout.classList.remove("sidebar-resizing");
      setAndStoreSidebarWidth(latest);
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    window.addEventListener("pointercancel", onUp);
  };
  const onSidebarResizeKey = (e: React.KeyboardEvent<HTMLDivElement>) => {
    const step = e.shiftKey ? 40 : 16;
    if (e.key === "ArrowLeft") setAndStoreSidebarWidth(sidebarWidth - step);
    else if (e.key === "ArrowRight") setAndStoreSidebarWidth(sidebarWidth + step);
    else if (e.key === "Home") setAndStoreSidebarWidth(SIDEBAR_WIDTH_MIN);
    else if (e.key === "End") setAndStoreSidebarWidth(SIDEBAR_WIDTH_MAX);
    else return;
    e.preventDefault();
  };
  const [loading, setLoading] = useState(false);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [activeSession, setActiveSession] = useState<SessionDetail | null>(null);
  const [error, setError] = useState("");
  /** PNG export progress, shown in a fixed notice so it is seen wherever the chat is scrolled. */
  const [exportNotice, setExportNotice] = useState<
    { kind: "working" } | { kind: "error"; text: string } | null
  >(null);
  const [pendingQuery, setPendingQuery] = useState<string | null>(null);
  const [loadingStatus, setLoadingStatus] = useState<string | null>(null);
  // Streaming answer in progress: the server's live status line, and the
  // answer so far (numbers first, then the summary growing word by word).
  const [streamStatus, setStreamStatus] = useState<string | null>(null);
  const [streamDraft, setStreamDraft] = useState<AIQueryResponse | null>(null);
  const [folders, setFolders] = useState<ChatFolder[]>([]);
  // session ids whose "continue" banner was closed (this page load only)
  const [dismissedBanners, setDismissedBanners] = useState<Set<string>>(() => new Set());
  // words of the history search that opened the chat -- scroll to the message they match
  const jumpQueryRef = useRef<string | null>(null);
  const [flashMessageId, setFlashMessageId] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState<{ open: boolean; sessionId: string | null; sessionName: string | null }>({
    open: false,
    sessionId: null,
    sessionName: null,
  });
  const [showArchivedDialog, setShowArchivedDialog] = useState(false);
  const [historySearchOpen, setHistorySearchOpen] = useState(false);
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const chatMessagesRef = useRef<HTMLDivElement>(null);
  const chatThreadRef = useRef<HTMLDivElement>(null);
  const assistantResponseRefs = useRef<Map<string, HTMLDivElement>>(new Map());
  const userMessageRefs = useRef<Map<string, HTMLDivElement>>(new Map());
  const pendingQueryRef = useRef<HTMLDivElement>(null);
  const loadingPlaceholderRef = useRef<HTMLDivElement>(null);
  const errorMessageRef = useRef<HTMLDivElement>(null);
  const scrollIntentRef = useRef<ScrollIntent>("none");
  const chatInputRef = useRef<ChatInputHandle>(null);
  const prevLoadingRef = useRef(false);
  const [enteringMessageId, setEnteringMessageId] = useState<string | null>(null);
  // Answer language (header dropdown). New questions are answered in it; changing it
  // re-presents the latest answer through POST /api/translate -- the stored answer is never
  // changed, so switching back and forth always starts from the original.
  const [responseLanguage, setResponseLanguage] = useState<LanguagePreference>(readStoredLanguage);
  // "<trace_id>:<language>" -> its translated text, or the request's state
  const [translations, setTranslations] = useState<Record<string, TranslatedAnswer | "pending" | "failed">>({});
  // message index -> the language that answer is shown in (absent / "auto" = as written)
  const [shownLanguage, setShownLanguage] = useState<Record<number, LanguagePreference>>({});
  const translationsInFlight = useRef<Set<string>>(new Set());

  const loadFolders = async () => {
    try {
      setFolders(await apiClient.listFolders());
    } catch {
      // Non-fatal -- an older backend has no folders; the history still lists every chat.
    }
  };

  // Temporary chat (dashed-bubble toggle in the sidebar): the open one's id -- state so the toggle
  // and header re-render, ref for handlers. Never listed, and discarded once the chat is left.
  const tempSessionRef = useRef<string | null>(null);
  const [tempId, setTempId] = useState<string | null>(null);

  const loadSessions = async () => {
    // A temporary chat never appears in the list, even on a server that does not know the flag.
    const hidden = new Set([...readPendingTemp(), ...(tempSessionRef.current ? [tempSessionRef.current] : [])]);
    const list = (await apiClient.listSessions()).filter((s) => !hidden.has(s.session_id));
    setSessions(list);
    return list;
  };

  // trace_id -> this session's thumbs vote (loaded with the session; failures just show no vote)
  const [votes, setVotes] = useState<Record<string, FeedbackRating>>({});
  const [exportOpen, setExportOpen] = useState(false);
  // The answer card whose Export menu is open (message id).
  const [answerExportOpen, setAnswerExportOpen] = useState<string | null>(null);
  // The question being edited in place (message index + draft text).
  const [editing, setEditing] = useState<{ index: number; text: string } | null>(null);
  // The answer running now -- what the Stop button aborts / cancels.
  const stopRef = useRef<{ controller: AbortController; requestId: string } | null>(null);
  // Insights drawer (observability, alerts, quality, freshness, investigations) -- read-only reports.
  const recordVote = useCallback(
    (traceId: string, rating: FeedbackRating) =>
      setVotes((v) => {
        const next = { ...v };
        if (rating === "clear") delete next[traceId];
        else next[traceId] = rating;
        return next;
      }),
    []
  );

  const openSession = async (sessionId: string, scrollToLatest = true, jumpTo?: string, wrap: Wrap = runNow) => {
    if (jumpTo) {
      jumpQueryRef.current = jumpTo;
      scrollIntentRef.current = "search-hit";
    } else if (scrollToLatest) {
      scrollIntentRef.current = "session-open";
    }
    const detail = await apiClient.getSession(sessionId);
    wrap(() => {
      leaveTemporary(sessionId);
      setActiveSession(detail);
    });
  };

  // Temporary chats are kept in server memory only; leaving one (another chat, new chat, toggle
  // off, refresh) deletes it.
  const discardTemporary = (id: string) =>
    apiClient.discardTemporarySession(id).then((ok) => {
      if (ok) writePendingTemp(readPendingTemp().filter((x) => x !== id));
    });
  const leaveTemporary = (nextId?: string) => {
    const leaving = tempSessionRef.current;
    if (leaving && leaving !== nextId) {
      tempSessionRef.current = null;
      setTempId(null);
      void discardTemporary(leaving);
    }
  };
  useEffect(() => {
    const onHide = () => {
      if (tempSessionRef.current) apiClient.discardTemporarySession(tempSessionRef.current);
    };
    window.addEventListener("pagehide", onHide);
    return () => window.removeEventListener("pagehide", onHide);
  }, []);

  const createNewSession = async (temporary = false, wrap: Wrap = runNow) => {
    const created = await apiClient.createSession("New Session", temporary);
    if (!temporary) await loadSessions();
    // One synchronous switch, so the mode transition (wrap) animates a single change.
    wrap(() => {
      leaveTemporary(created.session_id);
      if (temporary) {
        tempSessionRef.current = created.session_id;
        setTempId(created.session_id);
        writePendingTemp([...readPendingTemp(), created.session_id]);
      }
      setActiveSession(created);
      setError("");
    });
    chatInputRef.current?.clear();
    scrollIntentRef.current = "none";
    window.setTimeout(() => chatInputRef.current?.focus(), 0);
  };

  const refreshDatasets = async () => {
    try {
      const { datasets: list } = await apiClient.listDatasets();
      setDatasets(list);
    } catch {
      // Non-fatal -- the chip row just won't update; querying still works.
    }
  };

  useEffect(() => {
    if (!loading) {
      setLoadingStatus(null);
      return;
    }
    let idx = 0;
    setLoadingStatus(LOADING_STATUS_MESSAGES[0]);
    const timer = window.setInterval(() => {
      idx = Math.min(idx + 1, LOADING_STATUS_MESSAGES.length - 1);
      setLoadingStatus(LOADING_STATUS_MESSAGES[idx]);
    }, 1800);
    return () => window.clearInterval(timer);
  }, [loading]);

  useEffect(() => {
    const intent = scrollIntentRef.current;
    if (intent === "none") return;

    if (intent === "new-response") {
      if (loading && pendingQuery) {
        window.requestAnimationFrame(() => {
          // Prefer scrolling to the pending query bubble itself so the full
          // question is visible with the thinking indicator following below
          // it -- scrolling the placeholder card to the top instead left
          // only the bottom line of a two-line question peeking into view.
          scrollToElementStart(pendingQueryRef.current || loadingPlaceholderRef.current);
        });
        return;
      }

      if (!loading) {
        if (error) {
          window.requestAnimationFrame(() => {
            scrollToElementStart(errorMessageRef.current);
            scrollIntentRef.current = "none";
          });
          return;
        }

        if (activeSession?.messages?.length) {
          const messages = activeSession.messages;
          const last = messages[messages.length - 1];
          if (last.role === "assistant") {
            // Scroll to the user question that produced this answer, not
            // the answer card itself -- otherwise the question bubble right
            // above it gets pushed mostly out of view.
            const precedingUser = [...messages.slice(0, -1)].reverse().find((m) => m.role === "user");
            const target = precedingUser ? userMessageRefs.current.get(precedingUser.id) : undefined;
            window.requestAnimationFrame(() => {
              scrollToElementStart(target || assistantResponseRefs.current.get(last.id));
              scrollIntentRef.current = "none";
            });
          } else {
            scrollIntentRef.current = "none";
          }
        } else {
          scrollIntentRef.current = "none";
        }
      }
      return;
    }

    if (intent === "search-hit" && activeSession?.messages?.length) {
      const words = (jumpQueryRef.current || "").toLowerCase().split(/\s+/).filter(Boolean);
      jumpQueryRef.current = null;
      const messages = activeSession.messages;
      const has = (m: SessionMessage, all: boolean) => {
        const text = (m.content || "").toLowerCase();
        return all ? words.every((w) => text.includes(w)) : words.some((w) => text.includes(w));
      };
      const match = messages.find((m) => has(m, true)) ?? messages.find((m) => has(m, false));
      window.requestAnimationFrame(() => {
        const el = match && (match.role === "user" ? userMessageRefs.current.get(match.id) : assistantResponseRefs.current.get(match.id));
        if (el && match) {
          scrollToElementStart(el);
          setFlashMessageId(match.id);
          window.setTimeout(() => setFlashMessageId((id) => (id === match.id ? null : id)), 2200);
        } else {
          chatMessagesRef.current?.scrollTo({ top: chatMessagesRef.current.scrollHeight });
        }
        scrollIntentRef.current = "none";
      });
      return;
    }

    if (intent === "session-open" && activeSession?.messages?.length) {
      const messages = activeSession.messages;
      const lastAssistant = [...messages].reverse().find((m) => m.role === "assistant");
      const lastAssistantIdx = lastAssistant ? messages.indexOf(lastAssistant) : -1;
      const precedingUser =
        lastAssistantIdx > 0 ? [...messages.slice(0, lastAssistantIdx)].reverse().find((m) => m.role === "user") : undefined;
      window.requestAnimationFrame(() => {
        const target = precedingUser ? userMessageRefs.current.get(precedingUser.id) : undefined;
        if (target || lastAssistant) {
          scrollToElementStart(target || assistantResponseRefs.current.get(lastAssistant!.id));
        }
        scrollIntentRef.current = "none";
      });
    }
  }, [activeSession?.messages, loading, pendingQuery, error]);

  useEffect(() => {
    const wasLoading = prevLoadingRef.current;
    prevLoadingRef.current = loading;

    if (wasLoading && !loading && activeSession?.messages?.length) {
      const last = activeSession.messages[activeSession.messages.length - 1];
      if (last.role === "assistant") {
        setEnteringMessageId(last.id);
        // Matches (with headroom) .chat-response-enter's 380ms animation
        // (globals.css) -- clearing the class before that animation
        // finishes would cut it off mid-flight instead of letting it land.
        const timer = window.setTimeout(() => setEnteringMessageId(null), 420);
        return () => window.clearTimeout(timer);
      }
    }
  }, [loading, activeSession?.messages]);

  useEffect(() => {
    const init = async () => {
      try {
        loadFolders();
        // Temporary chats left by a refresh or closed tab: delete them now (they stay hidden meanwhile).
        await Promise.all(readPendingTemp().map((id) => discardTemporary(id)));
        const list = await loadSessions();
        if (list.length === 0) {
          await createNewSession();
          return;
        }
        const storedId = readStoredSessionId();
        const restored = storedId ? list.find((s) => s.session_id === storedId) : undefined;
        if (restored) {
          try {
            await openSession(restored.session_id);
            return;
          } catch (err) {
            console.warn("[ui] stored session could not be opened, falling back to latest:", err);
          }
        }
        await openSession(list[0].session_id);
      } catch (err: any) {
        setError(err.message || "Failed to load sessions");
      }
    };
    init();
  }, []);

  useEffect(() => {
    // A temporary chat is never reopened after a reload, so it is not remembered.
    const temp = !!activeSession?.session_id && activeSession.session_id === tempSessionRef.current;
    if (activeSession?.session_id && !temp) storeSessionId(activeSession.session_id);
    if (!temp) recordOpened(activeSession?.session_id ?? null);  // feeds the search dialog's "Last opened"
    setShownLanguage({});  // another chat: every answer as written
  }, [activeSession?.session_id]);

  // Ctrl/Cmd+K opens the chat search (like ChatGPT).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && !e.shiftKey && !e.altKey && e.key.toLowerCase() === "k") {
        e.preventDefault();
        if (tempSessionRef.current) return; // history stays out of sight in a temporary chat
        setHistorySearchOpen(true);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  useEffect(() => {
    const id = activeSession?.session_id;
    setVotes({});
    setExportOpen(false);
    setAnswerExportOpen(null);
    setEditing(null);
    if (!id) return;
    apiClient
      .getFeedback(id)
      .then((v) => setVotes(v))
      .catch(() => undefined);  // no votes shown; answering is unaffected
  }, [activeSession?.session_id]);

  const requestTranslation = (response: AIQueryResponse, language: LanguagePreference) => {
    const traceId = response.debug?.trace_id;
    if (!traceId || !activeSession || language === "auto" || (response.answer_language ?? "en") === language) return;
    const key = `${traceId}:${language}`;
    const known = translations[key];
    if ((known && known !== "failed") || translationsInFlight.current.has(key)) return;
    translationsInFlight.current.add(key);
    setTranslations((t) => ({ ...t, [key]: "pending" }));
    apiClient
      .translate(activeSession.session_id, traceId, language)
      .then((r) => setTranslations((t) => ({ ...t, [key]: r.translation.status === "failed" ? "failed" : r })))
      .catch((err) => {
        console.warn("[ui] translation failed -- showing the original answer:", err);
        setTranslations((t) => ({ ...t, [key]: "failed" }));
      })
      .finally(() => translationsInFlight.current.delete(key));
  };

  const changeLanguage = (language: LanguagePreference) => {
    setResponseLanguage(language);
    storeLanguage(language);
    // The answer on screen follows the new language straight away (no new question, no SQL).
    const messages = activeSession?.messages ?? [];
    let latest = -1;
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === "assistant" && messages[i].response) {
        latest = i;
        break;
      }
    }
    if (latest < 0) return;
    setShownLanguage((s) => ({ ...s, [latest]: language }));
    requestTranslation(messages[latest].response!, language);
  };

  /** The answer as shown: its translation when one was chosen and is ready, else the original. */
  const displayed = (index: number, response: AIQueryResponse): { result: AIQueryResponse; note: "translating" | "failed" | null } => {
    const language = shownLanguage[index];
    if (!language || language === "auto" || (response.answer_language ?? "en") === language) return { result: response, note: null };
    const t = translations[`${response.debug?.trace_id}:${language}`];
    if (t === "pending") return { result: response, note: "translating" };
    if (t === "failed" || !t) return { result: response, note: t === "failed" ? "failed" : null };
    return {
      result: {
        ...response, summary: t.summary, insights: t.insights, warnings: t.warnings, answer_language: t.answer_language,
        kpis: t.kpis, charts: t.charts, tables: t.tables, suggested_followup_labels: t.suggested_followup_labels,
        resolved_question_label: t.resolved_question_label,
      },
      note: null,
    };
  };

  useEffect(() => {
    // Reflects whatever datasets the backend already has loaded at page load
    // -- e.g. previous uploads that are still live because the server was
    // never restarted (loaded datasets live in memory only).
    refreshDatasets();
  }, []);

  const runQuery = async (
    q: string, page = 1, pageSize = ALL_ROWS, viaSuggestion = false, canonical?: string, opts: RunOptions = {}
  ) => {
    if (!q.trim() || !activeSession || loading) return;
    const trimmed = q.trim();
    let session = activeSession;
    console.log(`%c[ui] submitting query: "${trimmed}"`, "color:#5e35b1");
    scrollIntentRef.current = "new-response";
    setPendingQuery(trimmed);
    setLoading(true);
    setError("");
    const controller = new AbortController();
    const requestId = newRequestId();
    stopRef.current = { controller, requestId };
    try {
      if (opts.truncateTo !== undefined) {
        // The old answer (and anything after an edited question) leaves the chat and its memory first.
        await apiClient.truncateSession(session.session_id, opts.truncateTo);
        session = { ...session, messages: session.messages.slice(0, opts.truncateTo) };
        setActiveSession(session);
        if (controller.signal.aborted) throw new QueryStoppedError();
      }
      const emptyDraft = (): AIQueryResponse => ({
        question: trimmed, summary: "", kpis: [], charts: [], tables: [], insights: [], warnings: [], suggested_followups: [],
      });
      const onEvent = (event: AIStreamEvent) => {
        if (event.type === "status") setStreamStatus(event.message);
        else if (event.type === "partial")
          setStreamDraft((d) => ({
            ...(d ?? emptyDraft()), kpis: event.kpis, charts: event.charts, tables: event.tables,
            debug: { resolved_question: event.resolved_question ?? null },
          }));
        else if (event.type === "summary_delta")
          setStreamDraft((d) => ({ ...(d ?? emptyDraft()), summary: (d?.summary ?? "") + event.text }));
        // The backend's number check corrected the streamed summary: show the corrected one.
        else if (event.type === "summary_replace")
          setStreamDraft((d) => ({ ...(d ?? emptyDraft()), summary: event.text }));
      };
      // Stop wins straight away, even while the non-streaming fallback request is still out.
      const stopped = new Promise<never>((_, reject) =>
        controller.signal.addEventListener("abort", () => reject(new QueryStoppedError()), { once: true })
      );
      const res = await Promise.race([
        apiClient.aiQueryStream(
          {
            question: trimmed,
            session_id: session.session_id,
            filters,
            page,
            page_size: pageSize,
            via_followup: viaSuggestion,
            response_language: responseLanguage,
            canonical_question: canonical,
            request_id: requestId,
          },
          onEvent,
          controller.signal
        ),
        stopped,
      ]);
      // Render straight from `res` -- it already carries everything
      // ResultRenderer needs (kpis/charts/tables/summary/etc.). This used to
      // discard nearly all of it and make a SECOND, blocking
      // GET /api/sessions/{id} call just to get the same answer back
      // wrapped in a SessionMessage -- a full extra network round trip (plus
      // re-serializing the whole session's message history server-side) on
      // top of the already-measured ai-query time, for data already in
      // hand. `id` here is a local-only key (React key/ref-map lookups and
      // the entering-animation match in this file) -- never sent back to the
      // backend (table paging keys off `table_id`+`question`, renaming off
      // `session_id`), so a client-generated id is safe.
      const now = new Date().toISOString();
      const userMessage: SessionMessage = {
        id: `local-${Date.now()}-u`, role: "user", content: trimmed, timestamp: now,
      };
      const assistantMessage: SessionMessage = {
        id: `local-${Date.now()}-a`, role: "assistant", content: res.summary, timestamp: now,
        response: res, via_followup: viaSuggestion,
      };
      // setActiveSession and clearing pendingQuery/loading must land in the
      // SAME render pass (no await between them) -- see the flash this
      // avoids, described in the git history for this block.
      setActiveSession({ ...session, messages: [...session.messages, userMessage, assistantMessage] });
      setPendingQuery(null);
      setStreamDraft(null);
      setStreamStatus(null);
      setLoading(false);
      console.log("%c[ui] rendered response optimistically from ai-query", "color:#5e35b1");
      chatInputRef.current?.clear();
      // Reconcile with the backend's own persisted version in the
      // background -- picks up real message ids/session name/etc. without
      // making the user wait for it.
      apiClient
        .getSession(res.session_id || session.session_id)
        .then((detail) => setActiveSession((prev) => (prev && prev.session_id === detail.session_id ? detail : prev)))
        .catch((err) => console.error("[ui] background session reconcile failed:", err));
      loadSessions().catch((err) => console.error("[ui] loadSessions failed:", err));
    } catch (err: any) {
      if (err instanceof QueryStoppedError || controller.signal.aborted) {
        // Stopped: nothing is saved; the question goes back in the box to edit or send again.
        console.log("%c[ui] stopped by the user", "color:#5e35b1");
        scrollIntentRef.current = "none";
        chatInputRef.current?.setText(trimmed);
      } else {
        console.error("[ui] query failed:", err);
        setError(err.message || "Failed to run AI query");
      }
      setPendingQuery(null);
      setStreamDraft(null);
      setStreamStatus(null);
      setLoading(false);
    } finally {
      if (stopRef.current?.controller === controller) stopRef.current = null;
      chatInputRef.current?.focus();
    }
  };

  const stopQuery = useCallback(() => {
    const running = stopRef.current;
    if (!running) return;
    running.controller.abort();
    void apiClient.cancelQuery(running.requestId);  // the backend stops too and stores nothing
  }, []);

  /** Export as shown on screen (charts, metric cards, tables) -- see utils/snapshot.ts. */
  const exportSnapshot = (format: SnapshotFormat, title: string, parts: SnapshotPart[], closeMenu: () => void) => {
    closeMenu();
    if (!parts.length) return;
    // Let the menu close first so it is not in the copy.
    window.setTimeout(() => {
      if (format === "png" || format === "pdf") {
        // Drawing the image / pages takes a second or more; the file is saved as soon as it is ready.
        setExportNotice({ kind: "working" });
        (format === "png" ? renderSnapshotPng : renderSnapshotPdf)(title, parts)
          .then(({ blob, filename }) => {
            triggerDownload(blob, filename);
            setExportNotice(null);
          })
          .catch((err: any) => {
            console.error("[ui] export failed:", err);
            setExportNotice({ kind: "error", text: err?.message || "Export failed" });
          });
        return;
      }
      downloadSnapshotHtml(title, parts).catch((err: any) => {
        console.error("[ui] export failed:", err);
        setExportNotice({ kind: "error", text: err?.message || "Export failed" });
      });
    }, 0);
  };

  /** One answer: its card, headed by the question that produced it. */
  const answerParts = (index: number): SnapshotPart[] => {
    const messages = activeSession?.messages ?? [];
    const msg = messages[index];
    const node = msg ? assistantResponseRefs.current.get(msg.id) : undefined;
    if (!node) return [];
    let u = index - 1;
    while (u >= 0 && messages[u].role !== "user") u--;
    return [{ node, heading: u >= 0 ? messages[u].content : msg.response?.question }];
  };

  /** Edit and resubmit: the edited question replaces the original and everything after it. */
  const submitEdit = () => {
    if (!editing || loading) return;
    const text = editing.text.trim();
    if (!text) return;
    const index = editing.index;
    setEditing(null);
    void runQuery(text, 1, ALL_ROWS, false, undefined, { truncateTo: index });
  };
  // Stable handles for the (memoized) answer cards, so re-rendering the
  // workspace does not force every past answer's charts and tables to redraw.
  const runQueryRef = useRef(runQuery);
  runQueryRef.current = runQuery;
  const onFollowUp = useCallback(
    (q: string, canonical?: string) => runQueryRef.current(q, 1, ALL_ROWS, true, canonical),
    []
  );
  const onSubmitQuestion = useCallback((text: string) => runQueryRef.current(text), []);

  /** Rename / pin / archive / move a chat, then refresh the sidebar (and the open chat's header). */
  const updateSession = async (sessionId: string, patch: SessionPatch) => {
    try {
      const updated = await apiClient.updateSession(sessionId, patch);
      setSessions((prev) => prev.map((s) => (s.session_id === sessionId ? { ...s, ...updated } : s)));
      setActiveSession((prev) =>
        prev && prev.session_id === sessionId
          ? {
              ...prev,
              session_name: updated.session_name,
              pinned: updated.pinned,
              archived: updated.archived,
              archived_at: updated.archived_at,
              delete_at: updated.delete_at,
              folder_id: updated.folder_id,
            }
          : prev
      );
      await Promise.all([loadSessions(), patch.folder_id !== undefined ? loadFolders() : Promise.resolve()]);
    } catch (err: any) {
      setError(err.message || "Failed to update chat");
    }
  };

  const createFolder = async (name: string): Promise<ChatFolder | null> => {
    try {
      const folder = await apiClient.createFolder(name);
      await loadFolders();
      return folder;
    } catch (err: any) {
      setError(err.message || "Failed to create folder");
      return null;
    }
  };

  /** Rename and/or pin a folder. */
  const updateFolder = async (folderId: string, patch: { name?: string; pinned?: boolean }) => {
    try {
      const updated = await apiClient.updateFolder(folderId, patch);
      setFolders((prev) => prev.map((f) => (f.folder_id === folderId ? { ...f, ...updated } : f)));
      await loadFolders();
    } catch (err: any) {
      setError(err.message || "Failed to update folder");
    }
  };

  const deleteFolder = async (folderId: string) => {
    try {
      await apiClient.deleteFolder(folderId);
      await Promise.all([loadFolders(), loadSessions()]);
      setActiveSession((prev) => (prev && prev.folder_id === folderId ? { ...prev, folder_id: null } : prev));
    } catch (err: any) {
      setError(err.message || "Failed to delete folder");
    }
  };

  const deleteSession = async (sessionId: string) => {
    setConfirmDelete({
      open: true,
      sessionId,
      sessionName: sessions.find((s) => s.session_id === sessionId)?.session_name ?? null,
    });
  };

  const confirmDeleteSession = async (sessionId: string | null = null) => {
    const id = sessionId ?? confirmDelete.sessionId;
    if (!id) return;
    try {
      await apiClient.deleteSession(id);
      const list = await loadSessions();
      if (activeSession?.session_id === id) {
        if (list.length) {
          await openSession(list[0].session_id);
        } else {
          await createNewSession();
        }
      }
    } catch (err: any) {
      setError(err.message || "Failed to delete session");
    } finally {
      setConfirmDelete({ open: false, sessionId: null, sessionName: null });
    }
  };

  const cancelDeleteSession = () => {
    setConfirmDelete({ open: false, sessionId: null, sessionName: null });
  };


  // The screen's own text follows the dropdown, or with Auto-detect the latest answer's language.
  let latestAnswerLanguage: string | undefined;
  const allMessages = activeSession?.messages ?? [];
  for (let i = allMessages.length - 1; i >= 0; i--) {
    const response = allMessages[i].response;
    if (allMessages[i].role === "assistant" && response) {
      latestAnswerLanguage = displayed(i, response).result.answer_language;
      break;
    }
  }
  const uiLang = screenLanguage(responseLanguage, latestAnswerLanguage);
  const t = (key: Parameters<typeof uiText>[1], vars?: Record<string, string | number>) => uiText(uiLang, key, vars);

  const isTemporary = !!activeSession && (activeSession.session_id === tempId || !!activeSession.temporary);
  const activeSessionName = useMemo(
    () => (isTemporary ? t("temporaryChat") : activeSession ? sessionLabel(uiLang, activeSession.session_name) : "Project Workspace"),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [activeSession, isTemporary, uiLang]
  );
  // The page's language follows the screen language (font shaping, screen-reader voice).
  useEffect(() => {
    document.documentElement.lang = langTag(uiLang);
  }, [uiLang]);
  // Turning temporary chat on remembers the chat it was opened from; turning it off goes back
  // there (or to the most recent chat) instead of creating a new empty session. A new one is
  // made only when there is no saved chat at all.
  const beforeTempRef = useRef<string | null>(null);
  const [historyShown, setHistoryShown] = useState(true);
  const toggleTemporary = async (e?: React.MouseEvent) => {
    if (loading) return;
    const origin = originOf(e);
    if (!isTemporary) {
      beforeTempRef.current = activeSession?.session_id ?? null;
      void createNewSession(true, modeTransition(origin, "in"));
      return;
    }
    const out = modeTransition(origin, "out");
    const back =
      sessions.find((s) => s.session_id === beforeTempRef.current) ?? sessions.find((s) => !s.archived);
    beforeTempRef.current = null;
    // The history list can be thousands of rows; drawing it inside the switch would hold the
    // fold-back animation for seconds, so it comes back just after the animation.
    setHistoryShown(false);
    try {
      if (back) {
        await openSession(back.session_id, true, undefined, out);
        return;
      }
    } catch {
      // The chat could not be opened (deleted elsewhere): fall back to a new one below.
    } finally {
      window.setTimeout(() => setHistoryShown(true), 700);
    }
    await createNewSession(false, out);
  };


  // "Continue previous analysis": an older (or archived) chat opens with a note that follow-ups
  // still use its context, and a one-click re-run of its last question on today's data.
  const continueBanner = (() => {
    const msgs = activeSession?.messages ?? [];
    if (!activeSession || !msgs.length || loading || pendingQuery || dismissedBanners.has(activeSession.session_id)) return null;
    const lastAt = msgs[msgs.length - 1].timestamp;
    // Disabled for now (17 Nov): the "Continuing from …" / re-run banner is hidden; only the
    // archived-chat notice still shows. Flip SHOW_CONTINUE_BANNER to bring it back.
    const old = SHOW_CONTINUE_BANNER && lastAt ? Date.now() - new Date(lastAt).getTime() > CONTINUE_AFTER_MS : false;
    if (!old && !activeSession.archived) return null;
    const lastQuestion = [...msgs].reverse().find((m) => m.role === "user")?.content;
    return { lastAt, lastQuestion, archived: !!activeSession.archived, deleteAt: activeSession.delete_at, old };
  })();
  const dismissBanner = () =>
    activeSession && setDismissedBanners((prev) => new Set(prev).add(activeSession.session_id));
  const continueBannerEl = continueBanner ? (
    <div className="chat-continue-banner" role="status">
      {continueBanner.archived ? <ArchiveRestore size={15} /> : <History size={15} />}
      <div className="chat-continue-text">
        <strong>
          {continueBanner.archived ? t("archivedBanner") : t("continueTitle", { when: longDate(continueBanner.lastAt) })}
        </strong>
        {continueBanner.archived && continueBanner.deleteAt ? (
          <span>{t("archivedBannerExpiry", { date: longDate(continueBanner.deleteAt) })}</span>
        ) : null}
        {continueBanner.old ? <span>{t("continueBody")}</span> : null}
      </div>
      <div className="chat-continue-actions">
        {continueBanner.archived ? (
          <button type="button" className="chat-continue-btn" onClick={() => updateSession(activeSession!.session_id, { archived: false })}>
            {t("unarchive")}
          </button>
        ) : null}
        {continueBanner.old && continueBanner.lastQuestion ? (
          <button
            type="button"
            className="chat-continue-btn primary"
            onClick={() => runQuery(continueBanner.lastQuestion!)}
            title={continueBanner.lastQuestion}
          >
            <RefreshCw size={13} /> {t("rerunLatest")}
          </button>
        ) : null}
        <button type="button" className="chat-icon-btn" onClick={dismissBanner} title={t("dismiss")} aria-label={t("dismiss")}>
          <X size={13} />
        </button>
      </div>
    </div>
  ) : null;

  // On a fresh chat the question box sits inside the welcome card; once the
  // conversation starts it moves back to the fixed bar at the bottom.
  const isEmptyChat = !loading && !pendingQuery && (!activeSession?.messages || activeSession.messages.length === 0);
  const composer = (inline: boolean) => (
    <ChatInput
      ref={chatInputRef}
      inline={inline}
      loading={loading}
      onSubmit={onSubmitQuestion}
      onStop={stopQuery}
      stopLabel={t("stopGenerating")}
      speechUnsupportedLabel={t("speechUnsupported")}
      speechFailedLabel={t("speechStartFailed")}
      placeholder={t(isTemporary ? "tempPlaceholder" : "askPlaceholder")}
      micOnLabel={t("micOn")}
      micOffLabel={t("micOff")}
      speechLocale={speechLocale(responseLanguage)}
      above={inline ? undefined : continueBannerEl}
    />
  );

  // Chat history (pinned, folders, recents + row actions): in the expanded sidebar, or in the
  // collapsed rail's flyout (afterOpen closes the flyout once a chat is opened).
  const historyEl = (afterOpen?: () => void) => (
    <ChatHistory
      sessions={sessions}
      folders={folders}
      activeId={activeSession?.session_id ?? null}
      t={t}
      onOpen={(id) => {
        openSession(id, true);
        afterOpen?.();
      }}
      onUpdate={updateSession}
      onDelete={deleteSession}
      onCreateFolder={createFolder}
      onUpdateFolder={updateFolder}
      onDeleteFolder={deleteFolder}
    />
  );

  return (
    <div
      ref={layoutRef}
      className={`chat-app-layout${sidebarCollapsed ? " sidebar-collapsed" : ""}${isTemporary ? " temp-mode" : ""}`}
      lang={langTag(uiLang)}
      style={
        sidebarCollapsed
          ? undefined
          : ({
              "--chat-sidebar-width": `${sidebarWidth}px`,
              "--chat-content-max": `min(1480px, calc(100vw - ${sidebarWidth}px - 48px))`,
            } as React.CSSProperties)
      }
    >
      <aside className="chat-sidebar">
        {sidebarCollapsed ? null : (
          <div
            className="chat-sidebar-resizer"
            role="separator"
            aria-orientation="vertical"
            aria-label={t("resizeSidebar")}
            title={t("resizeSidebar")}
            aria-valuemin={SIDEBAR_WIDTH_MIN}
            aria-valuemax={SIDEBAR_WIDTH_MAX}
            aria-valuenow={sidebarWidth}
            tabIndex={0}
            onPointerDown={startSidebarResize}
            onDoubleClick={() => setAndStoreSidebarWidth(SIDEBAR_WIDTH_DEFAULT)}
            onKeyDown={onSidebarResizeKey}
          />
        )}
        {sidebarCollapsed ? (
          <nav className="chat-sidebar-rail" aria-label={t("showSidebar")}>
            <img className="chat-rail-logo-img" src={gmrLogo} alt="GMR" />
            <button
              className="chat-sidebar-icon chat-sidebar-rail-btn"
              type="button"
              onClick={toggleSidebar}
              aria-expanded={false}
              aria-label={t("showSidebar")}
              data-tip={t("showSidebar")}
            >
              <PanelLeft size={18} />
            </button>
            <button className="chat-sidebar-icon chat-sidebar-rail-btn" onClick={() => createNewSession()} type="button"
                    data-tip={t("newChat")} aria-label={t("newChat")}>
              <SquarePen size={18} />
            </button>
            <button
              className={`chat-sidebar-icon chat-sidebar-rail-btn chat-temp-toggle${isTemporary ? " is-on" : ""}`}
              type="button"
              onClick={toggleTemporary}
              disabled={loading}
              aria-pressed={isTemporary}
              aria-label={t(isTemporary ? "temporaryChatOff" : "temporaryChatOn")}
              data-tip={t(isTemporary ? "temporaryChatOff" : "temporaryChatOn")}
            >
              <MessageCircleDashed size={18} />
            </button>
            {/* A temporary chat keeps the history (search + recents) out of sight. */}
            {isTemporary ? null : (
              <>
                <button
                  className="chat-sidebar-icon chat-sidebar-rail-btn"
                  type="button"
                  onClick={() => setHistorySearchOpen(true)}
                  aria-haspopup="dialog"
                  aria-label={t("searchHistory")}
                  data-tip={`${t("searchHistory")} (Ctrl+K)`}
                >
                  <Search size={18} />
                </button>
                <RailRecents t={t} renderHistory={(close) => historyEl(close)} />
              </>
            )}
          </nav>
        ) : (
        <div className="chat-sidebar-top">
          <img className="chat-brand-logo" src={gmrLogo} alt="GMR" />
          <button
            className="chat-sidebar-icon chat-sidebar-search-btn"
            type="button"
            onClick={() => setHistorySearchOpen(true)}
            disabled={isTemporary}
            aria-haspopup="dialog"
            aria-label={t("searchHistory")}
            data-tip-pos="below" data-tip={`${t("searchHistory")} (Ctrl+K)`}
          >
            <Search size={18} />
          </button>
          <button
            className="chat-sidebar-icon chat-sidebar-toggle"
            type="button"
            onClick={toggleSidebar}
            aria-expanded={!sidebarCollapsed}
            aria-label={t(sidebarCollapsed ? "showSidebar" : "hideSidebar")}
            data-tip-pos="below" data-tip={t(sidebarCollapsed ? "showSidebar" : "hideSidebar")}
          >
            <PanelLeft size={18} />
          </button>
        </div>
        )}

        {sidebarCollapsed ? null : (
          <div className="chat-new-row">
            <button className="chat-new-btn" onClick={() => createNewSession()} type="button">
              <SquarePen size={17} /> {t("newChat")}
            </button>
            <button
              className={`chat-sidebar-icon chat-temp-toggle${isTemporary ? " is-on" : ""}`}
              type="button"
              onClick={toggleTemporary}
              disabled={loading}
              aria-pressed={isTemporary}
              aria-label={t(isTemporary ? "temporaryChatOff" : "temporaryChatOn")}
              data-tip-pos="below"
              data-tip={t(isTemporary ? "temporaryChatOff" : "temporaryChatOn")}
            >
              <MessageCircleDashed size={18} />
            </button>
          </div>
        )}

        {sidebarCollapsed || isTemporary || !historyShown ? null : historyEl()}
        {!sidebarCollapsed && (isTemporary || !historyShown) ? <div className="chat-temp-sidebar-fill" aria-hidden="true" /> : null}

        <ChatSearchDialog
          open={historySearchOpen}
          sessions={sessions}
          folders={folders}
          activeId={activeSession?.session_id ?? null}
          t={t}
          onOpen={(id, query) => openSession(id, true, query)}
          onClose={() => setHistorySearchOpen(false)}
        />

        <ProfileMenu
          archivedCount={sessions.filter((s) => s.archived).length}
          t={t}
          onOpenArchived={() => setShowArchivedDialog(true)}
        />
      </aside>

      <main className="chat-main">
        <header className="chat-header">
          <div>
            {/* An empty temporary chat already says so in its welcome; the header title returns once it starts. */}
            {isTemporary && isEmptyChat ? null : (
              <h2 className={isTemporary ? "chat-temp-title" : undefined}>
                {isTemporary ? <MessageCircleDashed size={16} aria-hidden="true" /> : null}
                {activeSessionName}
              </h2>
            )}
            {datasets.length <= 1 && (
              <p>
                {datasets.length === 1
                  ? t("connectedTo", { name: datasets[0].display_name })
                  : t("noData")}
              </p>
            )}
          </div>
          <div className="chat-header-controls">
            {activeSession?.messages?.length ? (
              <div className="export-menu">
                <button
                  type="button"
                  className="summary-audio-btn"
                  onClick={() => { void prepareSnapshotFonts(); setExportOpen((o) => !o); }}
                  aria-expanded={exportOpen}
                  title={t("exportTitle")}
                >
                  <Download size={14} />
                  <span>{t("exportChat")}</span>
                </button>
                {exportOpen ? (
                  <div className="export-menu-list" role="menu" onMouseLeave={() => setExportOpen(false)}>
                    {SNAPSHOT_FORMATS.map((format) => (
                      <button key={format} type="button" role="menuitem"
                              onClick={() => exportSnapshot(format, activeSessionName,
                                chatThreadRef.current ? [{ node: chatThreadRef.current }] : [], () => setExportOpen(false))}>
                        {t(SNAPSHOT_LABEL_KEYS[format])}
                      </button>
                    ))}
                  </div>
                ) : null}
              </div>
            ) : null}
            <LanguageSelect value={responseLanguage} onChange={changeLanguage} label={t("answerLanguage")} />
          </div>
        </header>

        <div className={`chat-messages${isEmptyChat ? " is-empty" : ""}${continueBannerEl && !isEmptyChat ? " has-banner" : ""}`} ref={chatMessagesRef}>
          <div className="chat-thread chat-thread-enter" key={activeSession?.session_id || "none"} ref={chatThreadRef}>
            {(() => {
              const assistantMessages = (activeSession?.messages || []).filter((m) => m.role === "assistant");
              const lastAssistantId = assistantMessages.length ? assistantMessages[assistantMessages.length - 1].id : undefined;
              return (activeSession?.messages || []).map((msg: SessionMessage, index: number) => {
                const shown = msg.response ? displayed(index, msg.response) : null;
                const answerLanguage = shown?.result.answer_language;
                return (
                <div key={msg.id} className={`chat-msg-row ${msg.role}${flashMessageId === msg.id ? " search-flash" : ""}`}>
                  {msg.role === "user" ? (
                    <div
                      className="chat-msg-col user"
                      ref={(el) => {
                        if (el) userMessageRefs.current.set(msg.id, el);
                        else userMessageRefs.current.delete(msg.id);
                      }}
                    >
                      <div className="chat-msg-meta">
                        <span className="chat-msg-avatar user">
                          <User size={12} />
                        </span>
                        <span className="chat-msg-meta-name">{t("you")}</span>
                        <span className="chat-msg-meta-time">{formatMessageTime(msg.timestamp)}</span>
                      </div>
                      {editing?.index === index ? (
                        <div className="chat-edit-box">
                          <textarea
                            className="chat-edit-textarea"
                            value={editing.text}
                            rows={3}
                            autoFocus
                            aria-label={t("editQuestion")}
                            onChange={(e) => setEditing({ index, text: e.target.value })}
                            onKeyDown={(e) => {
                              if (e.key === "Enter" && !e.shiftKey) {
                                e.preventDefault();
                                submitEdit();
                              } else if (e.key === "Escape") {
                                setEditing(null);
                              }
                            }}
                          />
                          {index + 2 < (activeSession?.messages.length ?? 0) ? (
                            <div className="chat-edit-hint">{t("editHint")}</div>
                          ) : null}
                          <div className="chat-edit-actions">
                            <button type="button" className="summary-audio-btn" onClick={() => setEditing(null)}>
                              {t("cancel")}
                            </button>
                            <button
                              type="button"
                              className="chat-edit-send"
                              onClick={submitEdit}
                              disabled={loading || !editing.text.trim()}
                            >
                              {t("editSend")}
                            </button>
                          </div>
                        </div>
                      ) : (
                        <>
                          <div className="chat-user-bubble chat-bubble-enter">{msg.content}</div>
                          <div className="chat-user-actions">
                            <CopyButton
                              getPayload={() => ({ text: msg.content, html: "" })}
                              title={t("copyMessage")}
                              copiedLabel={t("copied")}
                              compact
                              className="answer-action-btn icon-only"
                            />
                            <button
                              type="button"
                              className="answer-action-btn icon-only"
                              onClick={() => setEditing({ index, text: msg.content })}
                              disabled={loading}
                              title={t("editMessage")}
                              aria-label={t("editMessage")}
                            >
                              <Pencil size={13} />
                            </button>
                          </div>
                        </>
                      )}
                    </div>
                  ) : (
                    <div className="chat-msg-col assistant">
                      <div className="chat-msg-meta">
                        <span className="chat-msg-avatar assistant">
                          <img src={gmrLogo} alt="" />
                        </span>
                        <span className="chat-msg-meta-name">{t("assistant")}</span>
                        <span className="chat-msg-meta-time">{formatMessageTime(msg.timestamp)}</span>
                      </div>
                      <div
                        ref={(el) => {
                          if (el) assistantResponseRefs.current.set(msg.id, el);
                          else assistantResponseRefs.current.delete(msg.id);
                        }}
                        className={`chat-assistant-card${
                          enteringMessageId === msg.id ? " chat-response-enter" : ""
                        }`}
                        lang={langTag(answerLanguage)}
                      >
                        {msg.response?.debug?.resolved_question &&
                        msg.response.debug.resolved_question.trim().toLowerCase() !==
                          (msg.response.question || "").trim().toLowerCase() ? (
                          <div className="chat-interpreted-as">
                            {uiText(answerLanguage, "interpretedAs")}{" "}
                            {shown?.result.resolved_question_label ? (
                              <span>{shown.result.resolved_question_label}</span>
                            ) : (
                              <span lang="en">{msg.response.debug.resolved_question}</span>
                            )}
                          </div>
                        ) : null}
                        {shown?.note ? (
                          <div className={`chat-translation-note ${shown.note}`} role="status" aria-live="polite">
                            {shown.note === "translating"
                              ? uiText(shownLanguage[index], "translating")
                              : uiText(shownLanguage[index], "translationFailed")}
                          </div>
                        ) : null}
                        {shown ? (
                          <ResultRenderer
                            result={shown.result}
                            onFollowUp={onFollowUp}
                            showFollowups={msg.id === lastAssistantId}
                            feedbackSessionId={activeSession?.session_id}
                            feedbackRating={votes[msg.response?.debug?.trace_id ?? ""]}
                            onFeedbackChange={recordVote}
                          />
                        ) : (
                          <div className="chat-summary-text">{msg.content}</div>
                        )}
                      </div>
                      {msg.response ? (
                        <div className="chat-answer-actions">
                          {activeSession ? (
                            <div className="export-menu">
                              <button
                                type="button"
                                className="answer-action-btn"
                                onClick={() => { void prepareSnapshotFonts(); setAnswerExportOpen((o) => (o === msg.id ? null : msg.id)); }}
                                aria-expanded={answerExportOpen === msg.id}
                                title={t("exportAnswerTitle")}
                              >
                                <Download size={13} />
                                <span>{t("exportAnswer")}</span>
                              </button>
                              {answerExportOpen === msg.id ? (
                                <div className="export-menu-list left" role="menu" onMouseLeave={() => setAnswerExportOpen(null)}>
                                  {SNAPSHOT_FORMATS.map((format) => (
                                    <button key={format} type="button" role="menuitem"
                                            onClick={() => {
                                              const parts = answerParts(index);
                                              exportSnapshot(format, parts[0]?.heading || activeSessionName, parts,
                                                             () => setAnswerExportOpen(null));
                                            }}>
                                      {t(SNAPSHOT_LABEL_KEYS[format])}
                                    </button>
                                  ))}
                                </div>
                              ) : null}
                            </div>
                          ) : null}
                          <SourcesButton
                            citations={shown?.result.citations ?? msg.response.citations ?? []}
                            label={t("sources")}
                            title={t("viewSourcesTitle")}
                          />
                        </div>
                      ) : null}
                    </div>
                  )}
                </div>
                );
              });
            })()}

            {pendingQuery ? (
              <div className="chat-msg-row user">
                <div className="chat-msg-col user" ref={pendingQueryRef}>
                  <div className="chat-msg-meta">
                    <span className="chat-msg-avatar user">
                      <User size={12} />
                    </span>
                    <span className="chat-msg-meta-name">{t("you")}</span>
                    <span className="chat-msg-meta-time">{formatMessageTime(new Date().toISOString())}</span>
                  </div>
                  <div className="chat-user-bubble chat-bubble-enter">{pendingQuery}</div>
                </div>
              </div>
            ) : null}

            {loading && streamDraft ? (
              <div className="chat-msg-row assistant">
                <div className="chat-msg-col assistant">
                  <div className="chat-msg-meta">
                    <span className="chat-msg-avatar assistant">
                      <img src={gmrLogo} alt="" />
                    </span>
                    <span className="chat-msg-meta-name">{t("assistant")}</span>
                  </div>
                  <div className="chat-assistant-card chat-bubble-enter" ref={loadingPlaceholderRef}>
                    {streamStatus ? <LoadingStatus message={statusText(uiLang, streamStatus)} /> : null}
                    <ResultRenderer result={streamDraft} showFollowups={false} />
                  </div>
                </div>
              </div>
            ) : null}

            {loading && !streamDraft && (streamStatus || loadingStatus) ? (
              <div className="chat-msg-row assistant">
                <div className="chat-msg-col assistant">
                  <div className="chat-msg-meta">
                    <span className="chat-msg-avatar assistant">
                      <img src={gmrLogo} alt="" />
                    </span>
                    <span className="chat-msg-meta-name">{t("assistant")}</span>
                  </div>
                  <div className="chat-loading-placeholder chat-thinking-card chat-bubble-enter" ref={loadingPlaceholderRef}>
                    <TypingIndicator />
                    <LoadingStatus message={statusText(uiLang, streamStatus || loadingStatus || "")} />
                  </div>
                </div>
              </div>
            ) : null}

            {isEmptyChat ? (
              <div className="chat-msg-row empty">
                <EmptyState
                  title={isTemporary ? t("temporaryChat") : "GMR AI Intelligence"}
                  temporary={isTemporary}
                  temporaryHeadline={t("tempHeadline")}
                  lang={uiLang}
                  onSuggestedQuestion={(q, canonical) => runQuery(q, 1, ALL_ROWS, true, canonical)}
                  composer={composer(true)}
                />
              </div>
            ) : null}

            {error ? (
              <div className="chat-msg-row assistant">
                <div className="chat-error" ref={errorMessageRef}><strong>{t("queryError")}</strong> {error}</div>
              </div>
            ) : null}
          </div>
        </div>

        {isEmptyChat ? null : composer(false)}
        {exportNotice ? (
          <div className={`export-notice ${exportNotice.kind}`} role="status" aria-live="polite">
            {exportNotice.kind === "working" ? (
              <span>{t("exportPreparing")}</span>
            ) : (
              <span><strong>{t("exportFailed")}</strong> {exportNotice.text}</span>
            )}
            {exportNotice.kind !== "working" ? (
              <button type="button" className="export-notice-close" aria-label={t("exportDismiss")} title={t("exportDismiss")}
                      onClick={() => setExportNotice(null)}>×</button>
            ) : null}
          </div>
        ) : null}
        <ArchivedChatsDialog
          open={showArchivedDialog}
          sessions={sessions}
          t={t}
          blocked={confirmDelete.open}
          onClose={() => setShowArchivedDialog(false)}
          onOpenChat={(id) => openSession(id, true)}
          onUnarchive={(id) => updateSession(id, { archived: false })}
          onDelete={deleteSession}
        />
        <ConfirmDialog
          open={confirmDelete.open}
          title={t("confirmDelete")}
          message={t("deleteMessage", { name: confirmDelete.sessionName ? sessionLabel(uiLang, confirmDelete.sessionName) : t("thisChat") })}
          confirmLabel={t("delChat")}
          cancelLabel={t("cancel")}
          closeLabel={t("closeDialog")}
          onCancel={cancelDeleteSession}
          onConfirm={() => confirmDeleteSession(confirmDelete.sessionId)}
        />
      </main>
    </div>
  );
};
