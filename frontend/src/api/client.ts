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
}

export interface AIKPI {
  label: string;
  value: number;
  formatted_value: string;
  description?: string;
  unit?: string;
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
  description?: string;
}

export interface AITableSpec {
  id: string;
  title: string;
  /** The template's own heading for this table, shown instead of "Result Preview". */
  heading?: string | null;
  columns: string[];
  rows: Record<string, unknown>[];
  page: number;
  page_size: number;
  total_rows: number;
  has_next: boolean;
  has_previous?: boolean;
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
  /** Where the answer comes from, built by the backend's grounding guard (never by the
   * LLM): the live query + verified pattern/rules for data answers, the cited rule /
   * glossary chunks for explanations. */
  citations?: { id: string; source: string; location: string }[];
  text_summary?: string;
  display_mode?: "summary_first" | "full";
  spelling_corrections?: { original: string; corrected: string }[];
  session_id?: string;
  /** True when the backend declined to answer (out of scope / unsafe). */
  refused?: boolean;
  debug?: {
    planner_used?: string;
    /** The SQL that produced the answer's data, when one was run. */
    sql?: string | null;
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


const API_BASE = import.meta.env.VITE_API_BASE ?? "";

/** Logs every request/response with timing. Uses console.groupCollapsed so the
 * normal console stays readable, but the full request/response body is one
 * click away when debugging -- this is the "each step" logging on the
 * frontend side, mirroring the backend's per-request trace. */
function logRequest(method: string, endpoint: string, body?: unknown) {
  console.log(`%c[api] -> ${method} ${endpoint}`, "color:#7a7a7a", body ?? "");
}

function logResponse(method: string, endpoint: string, ms: number, ok: boolean, data: unknown) {
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
