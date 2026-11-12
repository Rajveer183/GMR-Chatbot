import type { LanguagePreference } from "../i18n/languages";

export interface FilterRequest {
  business_group: string;
  business_sector_name: string;
  unit_name: string;
  customer_name: string;
  project_name: string;
  cpc: string;
  ccn: string;
}

export interface AIQueryRequest {
  question: string;
  filters: FilterRequest;
  page: number;
  page_size: number;
  session_id?: string;
  /** True when this question was submitted via a suggested follow-up/chart
   * drilldown click rather than typed directly -- persisted on the stored
   * message so the "show full detail" view survives a page refresh. */
  via_followup?: boolean;
  /** The language dropdown: "auto" answers in the language of the question. */
  response_language?: LanguagePreference;
  /** A clicked follow-up shown translated: the verified English question behind the
   * translated text in `question` -- that is what runs. */
  canonical_question?: string;
}

/** An earlier answer's text in another language (POST /api/translate). Only the
 * written text -- KPI cards, charts and tables are data and never change. */
export interface TranslatedAnswer {
  trace_id: string;
  target_language: LanguagePreference;
  answer_language: string;
  summary: string;
  insights: string[];
  warnings: string[];
  /** Same values as the stored answer; display_* fields / column_labels carry the labels. */
  kpis: AIKPI[];
  charts: AIChartSpec[];
  tables: AITableSpec[];
  suggested_followups: string[];
  suggested_followup_labels?: string[] | null;
  /** "failed" = the translation could not be made; the text is the original. */
  translation: { target: string; status: "ok" | "partial" | "failed" | "skipped" };
}

export interface AIKPI {
  label: string;
  value: number;
  formatted_value: string;
  description?: string;
  unit?: string;
  /** The label in the answer's language (the backend's translation layer); `label` stays English. */
  display_label?: string;
  display_description?: string;
}

export interface AIChartSpec {
  id: string;
  title: string;
  type:
    | "bar" | "grouped_bar" | "stacked_bar" | "percent_stacked_bar" | "horizontal_bar" | "stacked_horizontal_bar"
    | "line" | "area" | "combo" | "pie" | "donut" | "funnel" | "radial" | "heatmap";
  x: string[];
  y: number[];
  /** Multi-series charts; in a "combo" chart, series with type "line" use the second axis. */
  series?: { name: string; data: number[]; type?: "bar" | "line" }[];
  unit?: string;
  /** Unit of the second (line) axis in a combo chart. */
  unit2?: string;
  /** The WFM dashboard KPI card this chart reproduces (backend dashboard_charts.py):
   * drawn in that card's style and series colours. */
  dashboard?: string;
  /** Dashboard value labels: inside each segment, only the stack total on top, or none. */
  labels?: "segments" | "total" | "none";
  /** Rates (Productivity, Utilization Rate, % SAT vs MI) shown above each bar, not drawn as bars. */
  rates?: { name: string; data: number[] }[];
  /** Technical (HES) boards: the board's colour key for each category (slice / bar) ... */
  point_keys?: string[];
  /** ... or for each series (Poor / Moderate, Dips / Surges, last / this month). */
  series_keys?: string[];
  description?: string;
  /** Title / description in the answer's language; `title` stays English. */
  display_title?: string;
  display_description?: string;
}

export interface AITableSpec {
  id: string;
  title: string;
  /** Set when one message held several questions: the answer run this table pages from
   * (/api/table-page), and its position in that run. Absent = the answer's own trace id. */
  trace_id?: string;
  table_index?: number;
  /** The template's own heading for this table, shown instead of "Result Preview". */
  heading?: string | null;
  columns: string[];
  rows: Record<string, unknown>[];
  page: number;
  page_size: number;
  total_rows: number;
  has_next: boolean;
  has_previous?: boolean;
  /** Title / heading in the answer's language. */
  display_title?: string;
  display_heading?: string;
  /** Column -> its header in the answer's language. Row keys stay the original columns. */
  column_labels?: Record<string, string>;
}

/** One server-side page of an answer's table (POST /api/table-page). */
export interface TablePage {
  columns: string[];
  rows: unknown[][];
  pagination: {
    page: number;
    page_size: number;
    total_count: number;
    total_pages: number;
    has_next: boolean;
    has_previous: boolean;
  };
}

export interface InvestigationHypothesis {
  h: string;
  status: "SUPPORTED" | "NOT SUPPORTED" | "PARTLY" | "CHECK" | "NOT TESTABLE" | string;
  evidence: string;
}

export interface InvestigationDimension {
  by: string;
  measure: string;
  periods: [string, string];
  rate: boolean;
  groups: number;
  total_prev?: number;
  total_curr?: number;
  change?: number;
  pct?: number | null;
  hypotheses: InvestigationHypothesis[];
}

export interface Investigation {
  measure: string | null;
  periods: [string, string] | null;
  dimensions: InvestigationDimension[];
  contradictions: string[];
  confidence: { level: "high" | "medium" | "low"; reasons: string[] };
  lead: { by: string; name: string; prev?: number; curr?: number; chg?: number; share?: number } | null;
  premise?: string | null;
  coverage_gap?: boolean;
}

export interface RelatedInvestigation {
  id: string;
  at: string;
  question: string;
  similarity?: number;
  lead?: Investigation["lead"];
  confidence?: string | null;
  action?: string;
}

/** Insights panel (GET /api/metrics, /api/alerts, /api/insights/quality, /api/data/freshness,
 * /api/investigations, /api/llm/status) -- all computed from the logs, no LLM call. */
export interface OpsTarget { target: number; value: number; met: boolean }
export interface OpsMetrics {
  window: { hours: number; since: string; questions: number; llm_calls_logged: number };
  latency_ms: { p50?: number; p95?: number; ttft_p50?: number; ttft_p95?: number };
  tokens_per_question: { input_p50?: number; output_p50?: number; total_mean?: number; total_p95?: number };
  work_per_question: { llm_calls_mean?: number; failed_llm_attempts?: number; db_queries_mean?: number };
  quality: { refusal_rate?: number; busy_rate?: number; error_rate?: number; sql_repair_rate?: number;
    number_correction_rate?: number; terminations?: Record<string, number> };
  cache: { answer_cache_hit_rate?: number };
  cost_usd_per_question?: number;
  cost_per_success?: { successes?: number; success_rate?: number; tokens?: number; wasted_token_share?: number };
  llm_failures_by_model?: Record<string, number>;
  attribution?: { by_pattern?: { key: string; questions: number; token_share: number; success_rate: number; p50_ms: number }[] };
  targets?: Record<string, OpsTarget>;
}
export interface OpsAlert {
  kind: string;
  severity: "high" | "medium" | "low" | string;
  title: string;
  detail?: string;
  evidence?: { examples?: string[]; [k: string]: unknown };
}
export interface OpsQuality {
  window: { hours: number; since: string };
  questions: number;
  good_answer_rate: number;
  root_causes: { cause: string; count: number; share: number }[];
  unanswered: { question: string; count: number; causes: Record<string, number>; last_seen: string; topics?: string[] }[];
  recurring_failures: { signature: string; count: number; recovered: number; last_seen: string; examples?: string[] }[];
  feedback: { votes: number; up: number; down: number; satisfaction: number | null; reasons: Record<string, number> };
  models: { model: string; attempts: number; success_rate: number; p50_ms: number; p95_ms: number; failures?: Record<string, number> }[];
}
export interface FreshnessRow { table: string; column: string; latest: string; monthly: boolean; stale: boolean; behind: string; label: string }
export interface InvestigationRecord {
  id: string;
  at: string;
  question: string;
  measure?: string | null;
  periods?: string[] | null;
  by?: string | null;
  total_prev?: number | null;
  total_curr?: number | null;
  lead?: Investigation["lead"];
  confidence?: { level?: string; reasons?: string[] } | string | null;
  contradictions?: string[];
  action?: string;
}
export interface LlmModelStatus { provider: string; model: string; enabled: boolean; cooling_down_seconds?: number | null }

export interface InvestigationRecheck {
  ok: boolean;
  summary?: string;
  error?: string;
  same_window?: boolean;
}

export interface AIQueryResponse {
  question: string;
  interpreted_query?: string;
  interpreted_intent?: string;
  intents?: string[];
  entities?: Record<string, string | null>;
  summary: string;
  kpis: AIKPI[];
  charts: AIChartSpec[];
  tables: AITableSpec[];
  insights: string[];
  warnings: string[];
  suggested_followups: string[];
  /** The follow-ups as shown, in the answer's language (same order). A click sends the shown
   * text as the question and the English one as canonical_question. */
  suggested_followup_labels?: string[] | null;
  /** Where the answer comes from, built by the backend's grounding guard (never by the
   * LLM): the live query + verified pattern/rules for data answers, the cited rule /
   * glossary chunks for explanations. */
  citations?: { id: string; source: string; location: string }[];
  /** Why-questions: each explanation tested by code on the rows (backend presentation/investigate.py). */
  investigation?: Investigation | null;
  /** Earlier investigations of a similar question (backend ops/investigations.py). */
  related_investigations?: RelatedInvestigation[];
  text_summary?: string;
  display_mode?: "summary_first" | "full";
  spelling_corrections?: { original: string; corrected: string }[];
  session_id?: string;
  /** True when the backend declined to answer (out of scope / unsafe). */
  refused?: boolean;
  /** The dropdown value the answer was asked with. */
  response_language?: LanguagePreference;
  /** The language the question was written in. */
  detected_language?: string | null;
  /** The language the summary / insights are written in. */
  answer_language?: string;
  /** Server-side counters for this answer (supervisor/run.py). */
  execution?: {
    latency_ms?: number;
    db_queries?: number;
    llm_calls?: number;
    input_tokens?: number;
    output_tokens?: number;
    [key: string]: unknown;
  };
  debug?: {
    planner_used?: string;
    /** "athena" (MDM) | "postgres" (WFM) | null */
    data_engine?: string | null;
    /** The SQL that produced the answer's data, when one was run. */
    sql?: string | null;
    /** Supporting queries run alongside the main one (cross-domain, investigation splits). */
    extra_sql?: string[];
    /** The model's short explanation of how it answered. */
    reasoning?: string | null;
    topics?: string[];
    schema_tables?: string[];
    answer_model?: string | null;
    /** Wall-clock time for the whole request, server side. */
    total_ms?: number;
    /** Every LLM call made for this answer, in order. */
    llm_calls?: LLMCallSummary[];
    /** Short ID shared by every backend log line for this request -- grep the
     * server log for it to see the full server-side trace. */
    trace_id?: string;
    /** The question actually answered, when it differs from what was typed
     * (Hinglish -> English, or a follow-up merged with the previous turn). */
    resolved_question?: string | null;
    /** How a follow-up was resolved against the previous turn. */
    followup?: "regex" | "llm" | null;
    /** One entry per pipeline stage (route/plan/execute/verify/shape/narrate),
     * each with its own timing in ms and a few key facts about what that
     * stage decided. Printed as a table in the browser console. */
    trace?: Array<{ step?: string; ms?: number; [key: string]: unknown }>;
  };
}

export interface LLMCallSummary {
  task: string;
  model: string;
  status: string;
  ms: number;
  prompt_tokens?: number | null;
  completion_tokens?: number | null;
}



export interface SessionMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  response?: AIQueryResponse;
  /** Persisted flag (assistant messages only): was this answer triggered by
   * a suggested follow-up / chart drilldown click? Used to decide whether
   * to render in full detail, in a way that survives a page refresh. */
  via_followup?: boolean;
}

/** One entry from GET /api/datasets -- every table ingested from the
 * server's data/ folder at startup that the AI-query auto-router can
 * currently pick between. There is no upload flow any more; this list is
 * fixed by whatever the backend has loaded. */
export interface DatasetSummary {
  dataset_id: string;
  display_name: string;
  domain: string;
  row_count: number;
  loaded_at: string;
  is_default: boolean;
}

export interface SessionSummary {
  session_id: string;
  session_name: string;
  created_at: string;
  updated_at: string;
  message_count: number;
  last_question: string;
}

export interface SessionDetail {
  session_id: string;
  session_name: string;
  created_at: string;
  updated_at: string;
  messages: SessionMessage[];
  last_question?: string;
}

export const FilterRequest = {};

export type FeedbackRating = "up" | "down" | "clear";

export const FEEDBACK_REASONS = ["wrong_number", "wrong_period", "wrong_filter", "not_what_i_asked", "incomplete", "slow", "other"] as const;
export type FeedbackReason = (typeof FEEDBACK_REASONS)[number];

export interface FeedbackPayload {
  trace_id: string;
  rating: FeedbackRating;
  session_id?: string;
  reason?: FeedbackReason;
  comment?: string;
}


const API_BASE = import.meta.env.VITE_API_BASE ?? "";

/** Logs every request/response with timing. Uses console.groupCollapsed so the
 * normal console stays readable, but the full request/response body is one
 * click away when debugging -- this is the "each step" logging on the
 * frontend side, mirroring the backend's per-request trace. */
// Request/SQL/trace logging is a developer aid: production builds never print it.
const DEBUG_LOG = import.meta.env.DEV;

function logRequest(method: string, endpoint: string, body?: unknown) {
  if (!DEBUG_LOG) return;
  console.log(`%c[api] -> ${method} ${endpoint}`, "color:#7a7a7a", body ?? "");
}

function logResponse(method: string, endpoint: string, ms: number, ok: boolean, data: unknown) {
  if (!DEBUG_LOG) return;
  const style = ok ? "color:#2e7d32" : "color:#c62828";
  console.groupCollapsed(`%c[api] <- ${method} ${endpoint} (${ms.toFixed(0)}ms)`, style);
  console.log(data);
  console.groupEnd();
}

/** For /api/ai-query specifically: the backend attaches debug.trace, one entry
 * per pipeline stage (route/plan/execute/verify/shape/narrate) with its own
 * timing. Printed as a table so the whole journey from question to response
 * is visible in the browser console without touching the server logs. */
function logQueryTrace(question: string, data: AIQueryResponse) {
  if (!DEBUG_LOG) return;
  const trace = data.debug?.trace;
  console.groupCollapsed(
    `%c[query] "${question}" -> ${data.debug?.answer_model ?? data.debug?.planner_used ?? "?"}${
      data.debug?.total_ms != null ? ` (${(data.debug.total_ms / 1000).toFixed(1)}s)` : ""
    }`,
    "color:#1565c0; font-weight:bold"
  );
  if (trace && trace.length) {
    console.table(trace);
  }
  if (data.debug?.sql) console.log("sql:", data.debug.sql);
  console.log("summary:", data.summary);
  console.log(
    `blocks: kpi=${data.kpis?.length ?? 0} chart=${data.charts?.length ?? 0} table=${data.tables?.length ?? 0}`
  );
  if (data.warnings?.length) console.warn("warnings:", data.warnings);
  console.groupEnd();
}

/** ai-query runs several LLM calls (plan, SQL, repair, narrate) so it gets
 * a generous budget; everything else is a plain lookup. */
const AI_QUERY_TIMEOUT_MS = 120_000;
const DEFAULT_TIMEOUT_MS = 20_000;

/** Pulls a human-readable message out of an error response. FastAPI sends
 * `{"detail": "..."}` (or a list of validation errors) -- show that instead
 * of the bare status text. */
async function errorMessage(response: Response): Promise<string> {
  const fallback = `API error: ${response.status} ${response.statusText}`.trim();
  try {
    const text = await response.text();
    if (!text) return fallback;
    try {
      const body = JSON.parse(text);
      const detail = body?.detail;
      if (typeof detail === "string" && detail.trim()) return detail;
      if (Array.isArray(detail) && detail.length) {
        return detail
          .map((d: any) => (typeof d === "string" ? d : d?.msg || JSON.stringify(d)))
          .join("; ");
      }
      if (detail) return JSON.stringify(detail);
    } catch {
      // Not JSON -- fall through to the raw body text.
    }
    return text.length <= 300 ? text : fallback;
  } catch {
    return fallback;
  }
}

/** One fetch wrapper for every JSON call: request/response logging, a hard
 * timeout via AbortController, and the backend's `detail` text on errors. */
async function requestJSON(method: string, endpoint: string, body?: unknown, timeoutMs = DEFAULT_TIMEOUT_MS) {
  logRequest(method, endpoint, body);
  const t0 = performance.now();
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  const seconds = Math.round(timeoutMs / 1000);
  try {
    let response: Response;
    try {
      response = await fetch(`${API_BASE}${endpoint}`, {
        method,
        headers: body !== undefined ? { "Content-Type": "application/json" } : undefined,
        body: body !== undefined ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });
    } catch (err: any) {
      logResponse(method, endpoint, performance.now() - t0, false, err);
      if (err?.name === "AbortError") throw err;
      throw new Error("Could not reach the server. Check that the backend is running and try again.");
    }
    if (!response.ok) {
      const message = await errorMessage(response);
      logResponse(method, endpoint, performance.now() - t0, false, message);
      throw new Error(message);
    }
    const data = await response.json();
    logResponse(method, endpoint, performance.now() - t0, true, data);
    return data;
  } catch (err: any) {
    if (err?.name === "AbortError") {
      throw new Error(
        `The server did not respond within ${seconds} seconds. It may be busy or rate-limited -- please try again.`
      );
    }
    throw err;
  } finally {
    window.clearTimeout(timer);
  }
}

const postJSON = (endpoint: string, body: unknown, timeoutMs?: number) => requestJSON("POST", endpoint, body, timeoutMs);
const patchJSON = (endpoint: string, body: unknown) => requestJSON("PATCH", endpoint, body);
const deleteJSON = (endpoint: string) => requestJSON("DELETE", endpoint);
const getJSON = (endpoint: string) => requestJSON("GET", endpoint);

/** One event from POST /api/ai-query/stream (newline-delimited JSON). */
export type AIStreamEvent =
  | { type: "status"; message: string }
  | { type: "partial"; kpis: AIKPI[]; charts: AIChartSpec[]; tables: AITableSpec[]; resolved_question?: string | null }
  | { type: "summary_delta"; text: string }
  | { type: "summary_replace"; text: string }
  | { type: "done"; response: AIQueryResponse }
  | { type: "error"; message: string };

/** Streams an answer: progress, then the numbers, then the written answer word
 * by word. Resolves with the final response (same shape as /api/ai-query). */
async function streamQuery(payload: AIQueryRequest, onEvent: (event: AIStreamEvent) => void): Promise<AIQueryResponse> {
  logRequest("POST", "/api/ai-query/stream", payload);
  const t0 = performance.now();
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), AI_QUERY_TIMEOUT_MS);
  try {
    const response = await fetch(`${API_BASE}/api/ai-query/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    if (!response.ok || !response.body) throw new Error(await errorMessage(response));
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let final: AIQueryResponse | null = null;
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let newline = buffer.indexOf("\n");
      while (newline >= 0) {
        const line = buffer.slice(0, newline).trim();
        buffer = buffer.slice(newline + 1);
        newline = buffer.indexOf("\n");
        if (!line) continue;
        const event = JSON.parse(line) as AIStreamEvent;
        if (event.type === "error") throw new Error(event.message);
        if (event.type === "done") final = event.response;
        onEvent(event);
      }
    }
    if (!final) throw new Error("The answer stream ended early -- please try again.");
    logResponse("POST", "/api/ai-query/stream", performance.now() - t0, true, final);
    return final;
  } catch (err: any) {
    if (err?.name === "AbortError") {
      throw new Error(`The server did not respond within ${Math.round(AI_QUERY_TIMEOUT_MS / 1000)} seconds -- please try again.`);
    }
    throw err;
  } finally {
    window.clearTimeout(timer);
  }
}

export const apiClient = {
  aiQuery: async (payload: AIQueryRequest): Promise<AIQueryResponse> => {
    const data = (await postJSON("/api/ai-query", payload, AI_QUERY_TIMEOUT_MS)) as AIQueryResponse;
    logQueryTrace(payload.question, data);
    return data;
  },

  /** Streaming variant; falls back to the plain endpoint if the stream fails
   * before anything was shown. */
  aiQueryStream: async (payload: AIQueryRequest, onEvent: (event: AIStreamEvent) => void): Promise<AIQueryResponse> => {
    let shown = false;
    try {
      const data = await streamQuery(payload, (event) => {
        if (event.type !== "status") shown = true;
        onEvent(event);
      });
      logQueryTrace(payload.question, data);
      return data;
    } catch (err) {
      if (shown) throw err;
      console.warn("[api] stream failed before any answer was shown -- using /api/ai-query:", err);
      return apiClient.aiQuery(payload);
    }
  },

  /** A page of an earlier answer's table: the backend re-runs that answer's validated
   * SQL with OFFSET / LIMIT (no LLM call, no new SQL). */
  tablePage: async (trace_id: string, page: number, page_size: number, table_index = 0, offset?: number): Promise<TablePage> => {
    return postJSON("/api/table-page", { trace_id, page, page_size, table_index, offset });
  },

  /** The full table behind an answer as CSV / Excel: the backend re-runs the answer's validated
   * SQL read-only without its display limit (no LLM call). Returns the file and its row count. */
  tableExport: async (
    trace_id: string, table_index: number, format: "csv" | "xlsx", labels: string[], title: string
  ): Promise<{ blob: Blob; filename: string; rows: number; truncated: boolean }> => {
    const controller = new AbortController();
    const timer = window.setTimeout(() => controller.abort(), AI_QUERY_TIMEOUT_MS);
    try {
      const response = await fetch(`${API_BASE}/api/table-export`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ trace_id, table_index, format, labels, title }),
        signal: controller.signal,
      });
      if (!response.ok) throw new Error(await errorMessage(response));
      const disp = response.headers.get("Content-Disposition") ?? "";
      const filename = /filename="([^"]+)"/.exec(disp)?.[1] ?? `table.${format}`;
      return {
        blob: await response.blob(),
        filename,
        rows: Number(response.headers.get("X-Row-Count") ?? 0),
        truncated: response.headers.get("X-Truncated") === "1",
      };
    } finally {
      window.clearTimeout(timer);
    }
  },

  // Insights panel: read-only reports built from the logs (no LLM call, no question re-run).
  getMetrics: async (hours = 24): Promise<OpsMetrics> => getJSON(`/api/metrics?hours=${hours}`),
  getAlerts: async (hours = 24): Promise<{ at: string; alerts: OpsAlert[] }> => getJSON(`/api/alerts?hours=${hours}`),
  getQuality: async (hours = 168): Promise<OpsQuality> => getJSON(`/api/insights/quality?hours=${hours}`),
  getFreshness: async (): Promise<{ tables: FreshnessRow[]; stale: number }> => getJSON("/api/data/freshness"),
  getSchemaChanges: async (): Promise<{ changes: Record<string, unknown>[] }> => getJSON("/api/data/schema-changes"),
  getInvestigations: async (limit = 50): Promise<{ items: InvestigationRecord[] }> =>
    getJSON(`/api/investigations?limit=${limit}`),
  getLlmStatus: async (): Promise<{ models: LlmModelStatus[] }> => getJSON("/api/llm/status"),
  digestUrl: (hours = 24): string => `${API_BASE}/api/insights/digest?hours=${hours}&format=md`,

  /** An earlier answer re-presented in another language: the backend translates the stored
   * answer's text (no SQL re-run, no new chat message, numbers checked unchanged). */
  translate: async (session_id: string, trace_id: string, target_language: LanguagePreference): Promise<TranslatedAnswer> => {
    return postJSON("/api/translate", { session_id, trace_id, target_language }, AI_QUERY_TIMEOUT_MS);
  },

  listSessions: async (): Promise<SessionSummary[]> => {
    return getJSON("/api/sessions");
  },

  createSession: async (session_name: string = "New Session"): Promise<SessionDetail> => {
    return postJSON("/api/sessions", { session_name });
  },

  getSession: async (session_id: string): Promise<SessionDetail> => {
    return getJSON(`/api/sessions/${encodeURIComponent(session_id)}`);
  },

  renameSession: async (session_id: string, session_name: string): Promise<SessionDetail> => {
    return patchJSON(`/api/sessions/${encodeURIComponent(session_id)}`, { session_name });
  },

  deleteSession: async (session_id: string): Promise<{ deleted: boolean; session_id: string }> => {
    return deleteJSON(`/api/sessions/${encodeURIComponent(session_id)}`);
  },

  listDatasets: async (): Promise<{ datasets: DatasetSummary[] }> => {
    return getJSON("/api/datasets");
  },

  /** Thumbs up / down on an answer (rating "clear" withdraws the vote). Logged for the
   * learning reports only -- it never changes the answer. */
  sendFeedback: async (payload: FeedbackPayload): Promise<{ ok: boolean; trace_id: string; rating: FeedbackRating }> => {
    return postJSON("/api/feedback", payload);
  },

  /** trace_id -> rating for one session (the thumbs state after a reload). */
  getFeedback: async (session_id: string): Promise<Record<string, FeedbackRating>> => {
    const data = (await getJSON(`/api/feedback?session_id=${encodeURIComponent(session_id)}`)) as { votes: Record<string, FeedbackRating> };
    return data.votes ?? {};
  },

  /** Decision-impact tracking: re-run an earlier investigation's query (read-only, no LLM call). */
  recheckInvestigation: async (id: string): Promise<InvestigationRecheck> => {
    return postJSON(`/api/investigations/${encodeURIComponent(id)}/recheck`, {});
  },

  /** Download link for the whole conversation (no query re-run, no LLM call). */
  exportSessionUrl: (session_id: string, format: "md" | "html" = "html"): string => {
    return `${API_BASE}/api/sessions/${encodeURIComponent(session_id)}/export?format=${format}`;
  },

  // Summary "Play" button: sends the summary text, gets back a playable
  // audio clip (Groq PlayAI TTS) as a blob.
  textToSpeech: async (text: string): Promise<Blob> => {
    const controller = new AbortController();
    const timer = window.setTimeout(() => controller.abort(), DEFAULT_TIMEOUT_MS * 2);
    try {
      const response = await fetch(`${API_BASE}/api/text-to-speech`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
        signal: controller.signal,
      });
      if (!response.ok) {
        throw new Error(await errorMessage(response));
      }
      return await response.blob();
    } catch (err: any) {
      if (err?.name === "AbortError") throw new Error("Audio generation timed out -- please try again.");
      throw err;
    } finally {
      window.clearTimeout(timer);
    }
  },
};
