import React, { useState } from "react";
import type { AIChartSpec, AIQueryResponse } from "../api/client";
import { SummaryCard } from "./SummaryCard";
import { MetricCard } from "./MetricCard";
import { ChartCard } from "./ChartCard";
import { ChartVisualization } from "./ChartVisualization";
import { ExpandableChartModal } from "./ExpandableChartModal";
import { ResultTable } from "./ResultTable";
import { InsightList } from "./InsightList";
import { WarningBox } from "./WarningBox";
import { SuggestedQuestions } from "./SuggestedQuestions";
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
}

// The backend now decides what to show per-answer (backend/ai/response_shape.py):
// a scalar answer's kpis/charts/tables arrays are simply empty, a ranking gets a
// chart+table, a full-list request gets everything. There is no separate
// "compact" rendering mode any more -- an answer with empty arrays already
// renders identically to the old summary-first view, just from real data
// instead of a UI-side toggle.
export const ResultRenderer: React.FC<ResultRendererProps> = ({ result, onFollowUp, showFollowups }) => {
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
  const displayKpis = result.kpis ?? [];
  const charts = result.charts ?? [];
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
        speechText={resultToSpeech(result)}
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
                subtitle={chart.display_description || chart.description}
                onExpand={() => setExpandedChart(chart)}
              >
                <ChartVisualization chart={chart} height={340} onPointClick={drilldownHandler(chart)} />
              </ChartCard>
            ))}
          </div>
        </section>
      ) : null}

      {tables.map((table, index) => (
        <section
          key={table.id}
          className="chat-result-section result-stagger"
          style={{ "--stagger": 2 + index } as React.CSSProperties}
        >
          <h4 className="chat-section-heading">
            {table.display_heading || table.heading || (index === 0 ? uiText(lang, "resultPreview") : table.display_title || table.title)}
          </h4>
          <ResultTable table={table} traceId={result.debug?.trace_id} tableIndex={index} lang={lang} />
        </section>
      ))}

      <InsightList title={uiText(lang, "insights")}items={result.insights ?? []} />

      {followups.length && onFollowUp ? (
        <section
          className="chat-result-section result-stagger"
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

      <ExpandableChartModal
        open={expandedChart !== null}
        title={expandedChart?.display_title || expandedChart?.title || "Chart"}
        subtitle={expandedChart?.display_description || expandedChart?.description}
        onClose={() => setExpandedChart(null)}
      >
        {expandedChart ? (
          <ChartVisualization chart={expandedChart} height={680} onPointClick={drilldownHandler(expandedChart)} />
        ) : null}
      </ExpandableChartModal>
    </div>
  );
};
