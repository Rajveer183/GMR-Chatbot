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

const DRILLDOWN_CHART_TYPES = new Set(["bar", "horizontal_bar", "stacked_bar"]);

interface ResultRendererProps {
  result: AIQueryResponse;
  onFollowUp?: (question: string) => void;
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
  // Rendered exactly as the backend sent them. There used to be a
  // `filterDisplayKpis` pass here that silently dropped any KPI whose label
  // contained "total reconciliation difference" unless the *question* happened
  // to mention reconciliation -- the frontend overriding which figures the
  // backend had decided the answer should contain. backend/agent/shape.py owns
  // that decision.
  const displayKpis = result.kpis ?? [];
  const charts = result.charts ?? [];
  const tables = result.tables ?? [];

  // Only bar-family charts label their x-axis with an actual entity name
  // (project/customer/unit/etc) -- pie/donut slices are usually categorical
  // buckets (ageing bands, dues vs non-dues) where "tell me about <slice>"
  // wouldn't resolve to anything, so those stay hover-only.
  const drilldownHandler = (chart: AIChartSpec) =>
    onFollowUp && DRILLDOWN_CHART_TYPES.has(chart.type)
      ? (label: string) => onFollowUp(`Tell me about ${label}`)
      : undefined;

  return (
    <div className="chat-result-panel">
      <SummaryCard
        summary={result.summary}
        interpretedIntent={result.interpreted_intent}
        interpretedQuery={result.interpreted_query}
        getFullCopyPayload={() => resultToPayload(result)}
        speechText={resultToSpeech(result)}
      />

      {displayKpis.length > 0 ? (
        <section
          className="chat-result-section result-stagger"
          style={{ "--stagger": 0 } as React.CSSProperties}
        >
          <h4 className="chat-section-heading">Key Metrics</h4>
          <div className="kpi-grid chat-kpi-grid">
            {displayKpis.map((kpi) => (
              <MetricCard
                key={kpi.label}
                label={kpi.label}
                formattedValue={kpi.formatted_value}
                subtitle={kpi.description}
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
                title={chart.title}
                subtitle={chart.description}
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
          <h4 className="chat-section-heading">{table.heading || (index === 0 ? "Result Preview" : table.title)}</h4>
          <ResultTable table={table} traceId={result.debug?.trace_id} tableIndex={index} />
        </section>
      ))}

      <InsightList title="Insights" items={result.insights ?? []} />

      {followups.length && onFollowUp ? (
        <section
          className="chat-result-section result-stagger"
          style={{ "--stagger": 2 + tables.length } as React.CSSProperties}
        >
          <h4 className="chat-section-heading">Suggested Follow-ups</h4>
          <SuggestedQuestions questions={followups} onPick={onFollowUp} />
        </section>
      ) : null}

      <WarningBox warnings={result.warnings ?? []} />

      <ExpandableChartModal
        open={expandedChart !== null}
        title={expandedChart?.title || "Chart"}
        subtitle={expandedChart?.description}
        onClose={() => setExpandedChart(null)}
      >
        {expandedChart ? (
          <ChartVisualization chart={expandedChart} height={680} onPointClick={drilldownHandler(expandedChart)} />
        ) : null}
      </ExpandableChartModal>
    </div>
  );
};
