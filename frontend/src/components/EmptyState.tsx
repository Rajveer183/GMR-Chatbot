import React from "react";
import { BarChart3, Languages, MousePointerClick, Sparkles } from "lucide-react";
import { SuggestedQuestions } from "./SuggestedQuestions";
import gmrLogo from "../assets/gmr_logo.png";

interface EmptyStateProps {
  title?: string;
  message?: string;
  /** When provided, the empty state renders clickable starter questions
   * that run immediately -- turning the first screen into a usable
   * jumping-off point instead of a static placeholder card. */
  onSuggestedQuestion?: (question: string) => void;
}

/* Starters verified against the live backend (WFM KPI tables + MDM feeder
 * auto-indexing): each one returns a correct answer with KPI tiles or a chart,
 * and together they cover headline stats, project ranking (a drillable bar
 * chart), monthly trend, the supply-to-collection funnel and auto-indexing.
 * Re-check with backend/eval/question_bank/run_bank.py before changing them. */
const STARTER_QUESTIONS = [
  "Give me installed so far, SAT percentage and remaining stock in one go.",
  "Which project installed the most meters in the last 30 days?",
  "Describe the installation trend over the last 12 months.",
  "Build the full funnel from supply to collection with conversion at each stage.",
  "What share of auto-indexing is complete?",
];

// Each card gets its own accent (drawn from the same brand chart palette as
// ChartVisualization -- frontend/src/components/chartOptions.ts) so this
// first screen reads as three distinct capabilities, not one flat block, and
// still stays on-brand rather than reaching for arbitrary new colours.
const CAPABILITY_CARDS = [
  {
    icon: Languages,
    title: "Ask in English or Hinglish",
    text: "Type it the way you'd ask a colleague -- \"Kal kitne meter lage?\" works too, and follow-ups like \"only in Kashi\" keep the context.",
    accent: "#0b3c6d",
    accentBg: "rgba(11, 60, 109, 0.1)",
  },
  {
    icon: MousePointerClick,
    title: "Click to drill down",
    text: "Click a project, OEM or division bar in a chart to get its full breakdown.",
    accent: "#1baf7a",
    accentBg: "rgba(27, 175, 122, 0.1)",
  },
  {
    icon: BarChart3,
    title: "KPIs, charts & data",
    text: "Answers come with a plain-language summary, KPI tiles or a dashboard-style chart, and the underlying table to verify it.",
    accent: "#4a3aa7",
    accentBg: "rgba(74, 58, 167, 0.1)",
  },
];

export const EmptyState: React.FC<EmptyStateProps> = ({
  title = "GMR AI Intelligence",
  message = "Ask a question to start the conversation. Your insights and summaries will appear here.",
  onSuggestedQuestion,
}) => {
  return (
    <div className="empty-state">
      <div className="empty-state-glow" aria-hidden="true" />
      <div className="empty-state-icon">
        <img src={gmrLogo} alt="GMR" />
      </div>
      <h3 className="empty-state-title">{title}</h3>
      <p className="empty-state-message">{message}</p>

      {onSuggestedQuestion ? (
        <div className="empty-state-starters">
          <div className="empty-state-starters-label">
            <Sparkles size={11} /> Try asking
          </div>
          <SuggestedQuestions questions={STARTER_QUESTIONS} onPick={onSuggestedQuestion} />
        </div>
      ) : null}

      <div className="empty-state-capabilities">
        {CAPABILITY_CARDS.map(({ icon: Icon, title: cardTitle, text, accent, accentBg }) => (
          <div className="empty-state-capability-card" key={cardTitle}>
            <div className="empty-state-capability-icon" style={{ color: accent, background: accentBg }}>
              <Icon size={14} />
            </div>
            <div>
              <div className="empty-state-capability-title">{cardTitle}</div>
              <div className="empty-state-capability-text">{text}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
