import React, { useEffect, useMemo, useRef, useState } from "react";
import { Menu, Plus, MessageSquare, Pencil, Trash2, Send, Mic, Search, X, User, Bot, ChevronDown, FileSpreadsheet } from "lucide-react";
import { apiClient } from "../api/client";
import type { AIQueryResponse, AIStreamEvent, DatasetSummary, FilterRequest, SessionSummary, SessionDetail, SessionMessage } from "../api/client";
import SpeechRecognition, { useSpeechRecognition } from "react-speech-recognition";
import { ResultRenderer } from "./ResultRenderer";
import { LoadingStatus } from "./LoadingStatus";
import { TypingIndicator } from "./TypingIndicator";
import { ConfirmDialog } from "./ConfirmDialog";
import { EmptyState } from "./EmptyState";
import gmrLogo from "../assets/gmr_logo.png";

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

type ScrollIntent = "none" | "new-response" | "session-open";

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

function focusWithoutScroll(element: HTMLTextAreaElement | null | undefined) {
  if (!element) return;
  try {
    element.focus({ preventScroll: true });
  } catch {
    element.focus();
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
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [activeSession, setActiveSession] = useState<SessionDetail | null>(null);
  const [error, setError] = useState("");
  const [pendingQuery, setPendingQuery] = useState<string | null>(null);
  const [loadingStatus, setLoadingStatus] = useState<string | null>(null);
  // Streaming answer in progress: the server's live status line, and the
  // answer so far (numbers first, then the summary growing word by word).
  const [streamStatus, setStreamStatus] = useState<string | null>(null);
  const [streamDraft, setStreamDraft] = useState<AIQueryResponse | null>(null);
  const [renamingSessionId, setRenamingSessionId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const [confirmDelete, setConfirmDelete] = useState<{ open: boolean; sessionId: string | null; sessionName: string | null }>({
    open: false,
    sessionId: null,
    sessionName: null,
  });
  const [speechError, setSpeechError] = useState<string | null>(null);
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [filesExpanded, setFilesExpanded] = useState(false);
  const {
    transcript,
    listening: speechActive,
    resetTranscript,
    browserSupportsSpeechRecognition,
  } = useSpeechRecognition();
  const [sessionSearch, setSessionSearch] = useState("");
  const chatMessagesRef = useRef<HTMLDivElement>(null);
  const assistantResponseRefs = useRef<Map<string, HTMLDivElement>>(new Map());
  const userMessageRefs = useRef<Map<string, HTMLDivElement>>(new Map());
  const pendingQueryRef = useRef<HTMLDivElement>(null);
  const loadingPlaceholderRef = useRef<HTMLDivElement>(null);
  const errorMessageRef = useRef<HTMLDivElement>(null);
  const scrollIntentRef = useRef<ScrollIntent>("none");
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const prevLoadingRef = useRef(false);
  const [enteringMessageId, setEnteringMessageId] = useState<string | null>(null);

  const loadSessions = async () => {
    const list = await apiClient.listSessions();
    setSessions(list);
    return list;
  };

  const openSession = async (sessionId: string, scrollToLatest = true) => {
    if (scrollToLatest) {
      scrollIntentRef.current = "session-open";
    }
    const detail = await apiClient.getSession(sessionId);
    setActiveSession(detail);
  };

  const createNewSession = async () => {
    const created = await apiClient.createSession("New Session");
    await loadSessions();
    setActiveSession(created);
    setQuestion("");
    scrollIntentRef.current = "none";
    window.setTimeout(() => focusWithoutScroll(inputRef.current), 0);
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

  // react-speech-recognition keeps the live transcript in its own internal
  // state; mirror it into the question box as it updates so typing/voice
  // input share the same field the same way the previous implementation did.
  useEffect(() => {
    if (speechActive) {
      setQuestion(transcript);
    }
  }, [transcript, speechActive]);

  useEffect(() => {
    return () => {
      SpeechRecognition.stopListening();
    };
  }, []);

  useEffect(() => {
    const init = async () => {
      try {
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
    if (activeSession?.session_id) storeSessionId(activeSession.session_id);
  }, [activeSession?.session_id]);

  useEffect(() => {
    // Reflects whatever datasets the backend already has loaded at page load
    // -- e.g. previous uploads that are still live because the server was
    // never restarted (loaded datasets live in memory only).
    refreshDatasets();
  }, []);

  const runQuery = async (q: string, page = 1, pageSize = ALL_ROWS, viaSuggestion = false) => {
    if (!q.trim() || !activeSession || loading) return;
    const trimmed = q.trim();
    console.log(`%c[ui] submitting query: "${trimmed}"`, "color:#5e35b1");
    scrollIntentRef.current = "new-response";
    setPendingQuery(trimmed);
    setLoading(true);
    setError("");
    try {
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
      const res = await apiClient.aiQueryStream(
        {
          question: trimmed,
          session_id: activeSession.session_id,
          filters,
          page,
          page_size: pageSize,
          via_followup: viaSuggestion,
        },
        onEvent
      );
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
      setActiveSession({ ...activeSession, messages: [...activeSession.messages, userMessage, assistantMessage] });
      setPendingQuery(null);
      setStreamDraft(null);
      setStreamStatus(null);
      setLoading(false);
      console.log("%c[ui] rendered response optimistically from ai-query", "color:#5e35b1");
      setQuestion("");
      // Reconcile with the backend's own persisted version in the
      // background -- picks up real message ids/session name/etc. without
      // making the user wait for it.
      apiClient
        .getSession(res.session_id || activeSession.session_id)
        .then((detail) => setActiveSession((prev) => (prev && prev.session_id === detail.session_id ? detail : prev)))
        .catch((err) => console.error("[ui] background session reconcile failed:", err));
      loadSessions().catch((err) => console.error("[ui] loadSessions failed:", err));
    } catch (err: any) {
      console.error("[ui] query failed:", err);
      setError(err.message || "Failed to run AI query");
      setPendingQuery(null);
      setStreamDraft(null);
      setStreamStatus(null);
      setLoading(false);
    } finally {
      focusWithoutScroll(inputRef.current);
    }
  };

  const beginRename = (s: SessionSummary) => {
    setRenamingSessionId(s.session_id);
    setRenameValue(s.session_name);
  };

  const commitRename = async () => {
    if (!renamingSessionId) return;
    try {
      await apiClient.renameSession(renamingSessionId, renameValue.trim() || "New Session");
      setRenamingSessionId(null);
      await loadSessions();
      if (activeSession?.session_id === renamingSessionId) {
        await openSession(renamingSessionId, false);
      }
    } catch (err: any) {
      setError(err.message || "Failed to rename session");
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


  const activeSessionName = useMemo(() => activeSession?.session_name || "Project Workspace", [activeSession]);

  const filteredSessions = useMemo(() => {
    const q = sessionSearch.trim().toLowerCase();
    if (!q) return sessions;
    return sessions.filter((s) => {
      const name = (s.session_name || "New Session").toLowerCase();
      const lastQuestion = (s.last_question || "").toLowerCase();
      return name.includes(q) || lastQuestion.includes(q);
    });
  }, [sessions, sessionSearch]);

  // Voice input runs on react-speech-recognition, a wrapper around the
  // browser's built-in Web Speech API (SpeechRecognition).
  const toggleSpeech = () => {
    if (!browserSupportsSpeechRecognition) {
      setSpeechError("Speech recognition is not supported in this browser.");
      return;
    }

    if (speechActive) {
      SpeechRecognition.stopListening();
      return;
    }

    setSpeechError(null);
    resetTranscript();
    setQuestion("");
    SpeechRecognition.startListening({ continuous: true, language: "en-US" }).catch(() => {
      setSpeechError("Unable to start speech recognition.");
    });
  };

  return (
    <div className="chat-app-layout">
      <aside className="chat-sidebar">
        <div className="chat-sidebar-top">
          <img className="chat-brand-logo" src={gmrLogo} alt="GMR" />
          <button className="chat-icon-btn" type="button">
            <Menu size={14} />
          </button>
        </div>

        <button className="chat-new-btn" onClick={createNewSession} type="button">
          <Plus size={14} /> New Chat
        </button>

        <div className="chat-menu">
          <button className="chat-menu-item active" type="button">
            <MessageSquare size={14} /> Chat
          </button>
        </div>

        <div className="chat-files-section">
          <button
            className="chat-menu-item chat-files-toggle"
            type="button"
            onClick={() => setFilesExpanded((v) => !v)}
            aria-expanded={filesExpanded}
          >
            <FileSpreadsheet size={14} />
            Data Sources
            <span className="chat-files-count">{datasets.length}</span>
            <ChevronDown size={14} className={`chat-files-chevron ${filesExpanded ? "expanded" : ""}`} />
          </button>
          {filesExpanded ? (
            <div className="chat-files-list">
              {datasets.length === 0 ? (
                <div className="chat-files-empty">No files loaded yet.</div>
              ) : (
                datasets.map((ds) => (
                  <div key={ds.dataset_id} className="chat-files-item">
                    <FileSpreadsheet size={13} className="chat-files-item-icon" />
                    <div className="chat-files-item-text">
                      <div className="chat-files-item-name" title={ds.display_name}>
                        {ds.display_name}
                      </div>
                      <div className="chat-files-item-meta">
                        {ds.row_count.toLocaleString()} rows
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>
          ) : null}
        </div>

        <div className="chat-session-search">
          <Search size={13} />
          <input
            type="text"
            value={sessionSearch}
            onChange={(e) => setSessionSearch(e.target.value)}
            placeholder="Search chat history..."
            aria-label="Search chat history"
          />
          {sessionSearch ? (
            <button
              type="button"
              className="chat-session-search-clear"
              onClick={() => setSessionSearch("")}
              aria-label="Clear search"
              title="Clear search"
            >
              <X size={12} />
            </button>
          ) : null}
        </div>
        <div className="chat-sessions-title">RECENT SESSIONS</div>
        <div className="chat-sessions-list">
          {filteredSessions.length === 0 ? (
            <div className="chat-sessions-empty">
              {sessionSearch ? `No chats match "${sessionSearch}"` : "No sessions yet"}
            </div>
          ) : (
            filteredSessions.map((s) => (
            <div
              key={s.session_id}
              className={`chat-session-item ${activeSession?.session_id === s.session_id ? "active" : ""}`}
              title={s.session_name}
            >
              {renamingSessionId === s.session_id ? (
                <input
                  className="chat-session-rename"
                  value={renameValue}
                  onChange={(e) => setRenameValue(e.target.value)}
                  onBlur={commitRename}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") commitRename();
                  }}
                  autoFocus
                />
              ) : (
                <button className="chat-session-name" type="button" onClick={() => openSession(s.session_id)}>
                  {s.session_name || "New Session"}
                </button>
              )}
              <div className="chat-session-actions">
                <button className="chat-icon-btn" type="button" onClick={() => beginRename(s)}>
                  <Pencil size={12} />
                </button>
                <button className="chat-icon-btn danger" type="button" onClick={() => deleteSession(s.session_id)}>
                  <Trash2 size={12} />
                </button>
              </div>
            </div>
            ))
          )}
        </div>

        <div className="chat-sidebar-footer">
          <div className="role">{datasets.length} file{datasets.length === 1 ? "" : "s"} loaded</div>
          <div className="subrole">Refreshed automatically</div>
        </div>
      </aside>

      <main className="chat-main">
        <header className="chat-header">
          <div>
            <h2>{activeSessionName}</h2>
            <p>
              {datasets.length > 1
                ? `${datasets.length} files loaded -- each question is auto-routed to the right one`
                : datasets.length === 1
                ? `Connected to: ${datasets[0].display_name}`
                : "No data loaded yet"}
            </p>
          </div>
          <div className="chat-header-controls">
            <span className="chat-active-badge">Active Session</span>
          </div>
        </header>

        <div className="chat-messages" ref={chatMessagesRef}>
          <div className="chat-thread chat-thread-enter" key={activeSession?.session_id || "none"}>
            {(() => {
              const assistantMessages = (activeSession?.messages || []).filter((m) => m.role === "assistant");
              const lastAssistantId = assistantMessages.length ? assistantMessages[assistantMessages.length - 1].id : undefined;
              return (activeSession?.messages || []).map((msg: SessionMessage) => (
                <div key={msg.id} className={`chat-msg-row ${msg.role}`}>
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
                        <span className="chat-msg-meta-name">You</span>
                        <span className="chat-msg-meta-time">{formatMessageTime(msg.timestamp)}</span>
                      </div>
                      <div className="chat-user-bubble chat-bubble-enter">{msg.content}</div>
                    </div>
                  ) : (
                    <div className="chat-msg-col assistant">
                      <div className="chat-msg-meta">
                        <span className="chat-msg-avatar assistant">
                          <Bot size={12} />
                        </span>
                        <span className="chat-msg-meta-name">Assistant</span>
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
                      >
                        {msg.response?.debug?.resolved_question &&
                        msg.response.debug.resolved_question.trim().toLowerCase() !==
                          (msg.response.question || "").trim().toLowerCase() ? (
                          <div className="chat-interpreted-as">
                            Interpreted as: <span>{msg.response.debug.resolved_question}</span>
                          </div>
                        ) : null}
                        {msg.response ? (
                          <ResultRenderer
                            result={msg.response}
                            onFollowUp={(q) => runQuery(q, 1, ALL_ROWS, true)}
                            showFollowups={msg.id === lastAssistantId}
                          />
                        ) : (
                          <div className="chat-summary-text">{msg.content}</div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              ));
            })()}

            {pendingQuery ? (
              <div className="chat-msg-row user">
                <div className="chat-msg-col user" ref={pendingQueryRef}>
                  <div className="chat-msg-meta">
                    <span className="chat-msg-avatar user">
                      <User size={12} />
                    </span>
                    <span className="chat-msg-meta-name">You</span>
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
                      <Bot size={12} />
                    </span>
                    <span className="chat-msg-meta-name">Assistant</span>
                  </div>
                  <div className="chat-assistant-card chat-bubble-enter" ref={loadingPlaceholderRef}>
                    {streamStatus ? <LoadingStatus message={streamStatus} /> : null}
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
                      <Bot size={12} />
                    </span>
                    <span className="chat-msg-meta-name">Assistant</span>
                  </div>
                  <div className="chat-loading-placeholder chat-thinking-card chat-bubble-enter" ref={loadingPlaceholderRef}>
                    <TypingIndicator />
                    <LoadingStatus message={streamStatus || loadingStatus || ""} />
                  </div>
                </div>
              </div>
            ) : null}

            {!loading && !pendingQuery && (!activeSession?.messages || activeSession.messages.length === 0) ? (
              <div className="chat-msg-row empty">
                <EmptyState
                  title="GMR AI Intelligence"
                  message="Start the conversation by asking a question. Your AI insights, summaries, and charts will appear here."
                  onSuggestedQuestion={(q) => runQuery(q, 1, ALL_ROWS, true)}
                />
              </div>
            ) : null}

            {error ? (
              <div className="chat-msg-row assistant">
                <div className="chat-error" ref={errorMessageRef}><strong>Query Error:</strong> {error}</div>
              </div>
            ) : null}
          </div>
        </div>

        <div className="chat-input-bar">
          <div className="chat-input-inner">
            <div className="chat-input-shell">
              <textarea
                ref={inputRef}
                className="chat-textarea"
                value={question}
                disabled={loading}
                placeholder="Ask anything about your data... (e.g., 'Which areas have the highest theft risk?')"
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    runQuery(question);
                  }
                }}
              />
              <button
                className={`chat-icon-btn mic-btn ${speechActive ? "active" : ""}`}
                type="button"
                onClick={toggleSpeech}
                title={speechActive ? "Turn mic off" : "Turn mic on"}
              >
                <Mic size={18} />
              </button>
              <button className="chat-send-btn" type="button" onClick={() => runQuery(question)} disabled={loading || speechActive}>
                {loading ? <TypingIndicator /> : <Send size={14} />}
              </button>
            </div>
          </div>
          {speechError ? <div className="chat-speech-error">{speechError}</div> : null}
        </div>
        <ConfirmDialog
          open={confirmDelete.open}
          title="Confirm Delete"
          message={`Delete the session "${confirmDelete.sessionName ?? "this chat"}"? This action cannot be undone.`}
          confirmLabel="Delete"
          cancelLabel="Cancel"
          onCancel={cancelDeleteSession}
          onConfirm={() => confirmDeleteSession(confirmDelete.sessionId)}
        />
      </main>
    </div>
  );
};
