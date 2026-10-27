import React from "react";
import { BarChart3, Languages, MousePointerClick, Sparkles } from "lucide-react";
import { SuggestedQuestions } from "./SuggestedQuestions";
import gmrLogo from "../assets/gmr_logo.png";
import { uiText } from "../i18n/languages";

interface EmptyStateProps {
  title?: string;
  message?: string;
  /** When provided, the empty state renders clickable starter questions
   * that run immediately -- turning the first screen into a usable
   * jumping-off point instead of a static placeholder card. */
  onSuggestedQuestion?: (question: string, canonical?: string) => void;
  /** Screen language: the cards, the intro and the starter questions follow it. */
  lang?: string;
}

/* Starters verified against the live backend (WFM KPI tables + MDM feeder
 * auto-indexing): each one returns a correct answer with KPI tiles or a chart,
 * and together they cover headline stats, project ranking (a drillable bar
 * chart), monthly trend, the supply-to-collection funnel and auto-indexing.
 * Re-check with backend/eval/question_bank/run_bank.py before changing them. */
const STARTER_KEYS = ["starter1", "starter2", "starter3", "starter4", "starter5"] as const;

// Each card gets its own accent (drawn from the same brand chart palette as
// ChartVisualization -- frontend/src/components/chartOptions.ts) so this
// first screen reads as three distinct capabilities, not one flat block, and
// still stays on-brand rather than reaching for arbitrary new colours.
const CAPABILITY_CARDS = [
  {
    icon: Languages,
    title: "es_card1Title",
    text: "es_card1Text",
    accent: "#0b3c6d",
    accentBg: "rgba(11, 60, 109, 0.1)",
  },
  {
    icon: MousePointerClick,
    title: "es_card2Title",
    text: "es_card2Text",
    accent: "#1baf7a",
    accentBg: "rgba(27, 175, 122, 0.1)",
  },
  {
    icon: BarChart3,
    title: "es_card3Title",
    text: "es_card3Text",
    accent: "#4a3aa7",
    accentBg: "rgba(74, 58, 167, 0.1)",
  },
] as const;

export const EmptyState: React.FC<EmptyStateProps> = ({
  title = "GMR AI Intelligence",
  message,
  onSuggestedQuestion,
  lang = "en",
}) => {
  // Starters verified in English: shown in the screen language, a click runs the English one.
  const starters = STARTER_KEYS.map((key) => ({ shown: uiText(lang, key), verified: uiText("en", key) }));
  return (
    <div className="empty-state">
      <div className="empty-state-glow" aria-hidden="true" />
      <div className="empty-state-icon">
        <img src={gmrLogo} alt="GMR" />
      </div>
      <h3 className="empty-state-title">{title}</h3>
      <p className="empty-state-message">{message ?? uiText(lang, "es_message")}</p>

      {onSuggestedQuestion ? (
        <div className="empty-state-starters">
          <div className="empty-state-starters-label">
            <Sparkles size={11} /> {uiText(lang, "tryAsking")}
          </div>
          <SuggestedQuestions
            questions={starters.map((s) => s.shown)}
            onPick={(q, i) => onSuggestedQuestion(q, q !== starters[i].verified ? starters[i].verified : undefined)}
          />
        </div>
      ) : null}

      <div className="empty-state-capabilities">
        {CAPABILITY_CARDS.map(({ icon: Icon, title: cardTitle, text, accent, accentBg }) => (
          <div className="empty-state-capability-card" key={cardTitle}>
            <div className="empty-state-capability-icon" style={{ color: accent, background: accentBg }}>
              <Icon size={14} />
            </div>
            <div>
              <div className="empty-state-capability-title">{uiText(lang, cardTitle)}</div>
              <div className="empty-state-capability-text">{uiText(lang, text)}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
