import React, { memo, useMemo, useState } from "react";
import type { AIChartSpec, AIQueryResponse } from "../api/client";
import { SummaryCard } from "./SummaryCard";
import { MetricCard } from "./MetricCard";
import { ChartCard } from "./ChartCard";
import { ChartVisualization } from "./ChartVisualization";
import { DASHBOARD_CARDS } from "./charts/dashboardPalette";
import { ExpandableChartModal } from "./ExpandableChartModal";
import { ResultTable } from "./ResultTable";
import { InsightList } from "./InsightList";
import { FeederComparePanel } from "./FeederComparePanel";
import { WarningBox } from "./WarningBox";
import { SourcesNote } from "./SourcesNote";
import { SuggestedQuestions } from "./SuggestedQuestions";
import { FeedbackButtons } from "./FeedbackButtons";
import type { FeedbackRating } from "../api/client";
import { apiClient } from "../api/client";
import { resultToPayload } from "../utils/clipboard";
import { resultToSpeech } from "../utils/speech";
import { uiText } from "../i18n/languages";

const DRILLDOWN_CHART_TYPES = new Set(["bar", "horizontal_bar", "stacked_bar"]);

interface ResultRendererProps {
  result: AIQueryResponse;
  /** question = the text shown (and sent); canonical = the verified English follow-up behind
   * a translated chip, when there is one. */
  onFollowUp?: (question: string, canonical?: string) => void;
  showFollowups?: boolean;
  /** Thumbs up / down under the summary; shown when onFeedbackChange is given (omitted for drafts
   * still streaming). Flat props rather than one object so React.memo can skip unchanged answers. */
  feedbackSessionId?: string;
  feedbackRating?: FeedbackRating;
  onFeedbackChange?: (traceId: string, rating: FeedbackRating) => void;
}

// The backend now decides what to show per-answer (backend/ai/response_shape.py):
// a scalar answer's kpis/charts/tables arrays are simply empty, a ranking gets a
// chart+table, a full-list request gets everything. There is no separate
// "compact" rendering mode any more -- an answer with empty arrays already
// renders identically to the old summary-first view, just from real data
// instead of a UI-side toggle.
// Memoized: past answers (charts + full tables) must not re-render when unrelated workspace
// state changes, e.g. the streaming status line of a new question.
export const ResultRenderer = memo(function ResultRenderer({
  result, onFollowUp, showFollowups, feedbackSessionId, feedbackRating, onFeedbackChange,
}: ResultRendererProps) {
  const speechText = useMemo(() => resultToSpeech(result), [result]);
  const [expandedChart, setExpandedChart] = useState<AIChartSpec | null>(null);
  const followups = result.suggested_followups?.length && showFollowups ? result.suggested_followups : [];
  // Shown in the answer's language when the backend translated them; a click still runs the
  // verified English question behind each one.
  const followupLabels = followups.map((q, i) => result.suggested_followup_labels?.[i] || q);
  // Rendered exactly as the backend sent them. There used to be a
  // `filterDisplayKpis` pass here that silently dropped any KPI whose label
  // contained "total reconciliation difference" unless the *question* happened
  // to mention reconciliation -- the frontend overriding which figures the
  // backend had decided the answer should contain. backend/agent/shape.py owns
  // that decision.
  // A consumer's feeder comparison is drawn by its own panel: no KPI cards / plain chart, tables folded below it.
  const feederCompare = result.feeder_compare ?? null;
  const displayKpis = feederCompare ? [] : result.kpis ?? [];
  const charts = feederCompare ? [] : result.charts ?? [];
  const tables = result.tables ?? [];
  // Section headings follow the language the answer is written in.
  const lang = result.answer_language;

  // Only bar-family charts label their x-axis with an actual entity name
  // (project/customer/unit/etc) -- pie/donut slices are usually categorical
  // buckets (ageing bands, dues vs non-dues) where "tell me about <slice>"
  // wouldn't resolve to anything, so those stay hover-only.
  const drilldownHandler = (chart: AIChartSpec) =>
    onFollowUp && DRILLDOWN_CHART_TYPES.has(chart.type)
      ? (label: string) => onFollowUp(uiText(lang, "drillDown", { x: label }))
      : undefined;

  return (
    <div className="chat-result-panel">
      <SummaryCard
        summary={result.summary}
        interpretedIntent={result.interpreted_intent}
        interpretedQuery={result.interpreted_query}
        getFullCopyPayload={() => resultToPayload(result)}
        speechText={speechText}
        lang={lang}
      />

      {displayKpis.length > 0 ? (
        <section
          className="chat-result-section result-stagger"
          style={{ "--stagger": 0 } as React.CSSProperties}
        >
          <h4 className="chat-section-heading">{uiText(lang, "keyMetrics")}</h4>
          <div className="kpi-grid chat-kpi-grid">
            {displayKpis.map((kpi) => (
              <MetricCard
                key={kpi.label}
                label={kpi.display_label || kpi.label}
                formattedValue={kpi.formatted_value}
                subtitle={kpi.display_description || kpi.description}
              />
            ))}
          </div>
        </section>
      ) : null}

      {charts.length > 0 ? (
        <section
          className="chat-result-section result-stagger"
          style={{ "--stagger": 1 } as React.CSSProperties}
        >
          <div className="charts-row chat-charts-row">
            {charts.map((chart) => (
              <ChartCard
                key={chart.id}
                title={chart.display_title || chart.title}
                subtitle={chart.display_description || chart.description || (chart.dashboard && DASHBOARD_CARDS[chart.dashboard]
                  ? uiText(lang, "dashboardView", { card: DASHBOARD_CARDS[chart.dashboard] }) : undefined)}
                onExpand={() => setExpandedChart(chart)}
              >
                <ChartVisualization chart={chart} height={340} onPointClick={drilldownHandler(chart)} />
              </ChartCard>
            ))}
          </div>
        </section>
      ) : null}

      {feederCompare ? (
        <section className="chat-result-section result-stagger" style={{ "--stagger": 1 } as React.CSSProperties}>
          <FeederComparePanel data={feederCompare} onAsk={onFollowUp ? (q) => onFollowUp(q) : undefined} />
        </section>
      ) : null}

      {feederCompare && tables.length ? (
        <details className="chat-result-section fcp-data">
          <summary>Data tables &amp; downloads ({tables.length})</summary>
          {tables.map((table, index) => (
        <section
          key={table.id}
          className="chat-result-section result-stagger"
          style={{ "--stagger": 2 + index } as React.CSSProperties}
        >
          <h4 className="chat-section-heading">
            {table.display_heading || table.heading || (index === 0 ? uiText(lang, "resultPreview") : table.display_title || table.title)}
          </h4>
          <ResultTable table={table} traceId={table.trace_id ?? result.debug?.trace_id}
            tableIndex={table.table_index ?? index} lang={lang} />
        </section>
      ))}

        </details>
      ) : null}

      {!feederCompare ? (
        <>
      {tables.map((table, index) => (
        <section
          key={table.id}
          className="chat-result-section result-stagger"
          style={{ "--stagger": 2 + index } as React.CSSProperties}
        >
          <h4 className="chat-section-heading">
            {table.display_heading || table.heading || (index === 0 ? uiText(lang, "resultPreview") : table.display_title || table.title)}
          </h4>
          <ResultTable table={table} traceId={table.trace_id ?? result.debug?.trace_id}
            tableIndex={table.table_index ?? index} lang={lang} />
        </section>
      ))}

        </>
      ) : null}

      {!feederCompare ? <InsightList title={uiText(lang, "insights")} items={result.insights ?? []} /> : null}

      {result.report?.trace_id ? (
        <section className="chat-result-section">
          <div className="table-download">
            <a className="table-download-btn" href={apiClient.reportExportUrl(result.report.trace_id, "xlsx")} download>
              Download report (Excel)
            </a>
            <a className="table-download-btn" href={apiClient.reportExportUrl(result.report.trace_id, "html")} download>
              Download report (HTML)
            </a>
          </div>
        </section>
      ) : null}

      {/* Internal checks (hypotheses tested, confidence, earlier investigations) stay in the backend: the
          answer itself carries the verdict. */}

      {followups.length && onFollowUp ? (
        <section
          className="chat-result-section result-stagger no-export"
          style={{ "--stagger": 2 + tables.length } as React.CSSProperties}
        >
          <h4 className="chat-section-heading">{uiText(lang, "followups")}</h4>
          <SuggestedQuestions
            questions={followupLabels}
            onPick={(label, i) => onFollowUp(label, label !== followups[i] ? followups[i] : undefined)}
          />
        </section>
      ) : null}

      <WarningBox warnings={result.warnings ?? []} title={uiText(lang, "warnings")} />

      {/* Below the follow-ups, right above Sources. */}
      {onFeedbackChange && result.debug?.trace_id ? (
        <FeedbackButtons
          traceId={result.debug.trace_id}
          sessionId={feedbackSessionId}
          initial={feedbackRating}
          lang={lang}
          onChange={onFeedbackChange}
        />
      ) : null}

      <SourcesNote citations={result.citations ?? []} title={uiText(lang, "sources")} />

      {/* Drafts still streaming have no trace yet; the panel renders nothing until the final answer. */}

      <ExpandableChartModal
        open={expandedChart !== null}
        title={expandedChart?.display_title || expandedChart?.title || "Chart"}
        subtitle={expandedChart?.display_description || expandedChart?.description}
        onClose={() => setExpandedChart(null)}
      >
        {(available: number) =>
          expandedChart ? (
            <ChartVisualization chart={expandedChart} height={available} onPointClick={drilldownHandler(expandedChart)} />
          ) : null
        }
      </ExpandableChartModal>
    </div>
  );
});
