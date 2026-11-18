import React from "react";
import { ArrowUpRight, FileSpreadsheet, Gauge, Languages, MessageCircleDashed, MessagesSquare, Sparkles, TrendingUp, Trophy } from "lucide-react";
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
  /** The question box, shown inside the card under the intro line. */
  composer?: React.ReactNode;
  /** Temporary chat: GMR logo, a small "Temporary chat" label (title), a two-tone headline and
   * the question box -- nothing else. */
  temporary?: boolean;
  temporaryHeadline?: string;
  temporarySubline?: string;
}

/* Starters verified against the live backend (WFM KPI tables + MDM feeder
 * auto-indexing): each one returns a correct answer with KPI tiles or a chart,
 * and together they cover headline stats, project ranking (a drillable bar
 * chart) and the monthly trend.
 * Re-check with backend/eval/question_bank/run_bank.py before changing them. */
const STARTERS = [
  { key: "starter1", icon: Gauge },
  { key: "starter2", icon: Trophy },
  { key: "starter3", icon: TrendingUp },
] as const;

// Each card gets its own accent (drawn from the same brand chart palette as
// ChartVisualization -- frontend/src/components/chartOptions.ts) so this
// first screen reads as three distinct capabilities, not one flat block, and
// still stays on-brand rather than reaching for arbitrary new colours.
// Every card describes something the chat UI really does: mic input + read-aloud
// (ChatInput / play button), context-carrying follow-ups + "Interpreted as",
// and CSV/Excel table download + whole-chat export (ResultTable / export menu).
const CAPABILITY_CARDS = [
  {
    icon: Languages,
    title: "es_card1Title",
    accent: "#0b3c6d",
    accentBg: "rgba(11, 60, 109, 0.1)",
  },
  {
    icon: MessagesSquare,
    title: "es_card2Title",
    accent: "#c27c0e",
    accentBg: "rgba(194, 124, 14, 0.12)",
  },
  {
    icon: FileSpreadsheet,
    title: "es_card3Title",
    accent: "#4a3aa7",
    accentBg: "rgba(74, 58, 167, 0.1)",
  },
] as const;

export const EmptyState: React.FC<EmptyStateProps> = ({
  title = "GMR AI Intelligence",
  message,
  onSuggestedQuestion,
  lang = "en",
  composer,
  temporary = false,
  temporaryHeadline,
  temporarySubline,
}) => {
  // Starters verified in English: shown in the screen language, a click runs the English one.
  const starters = STARTERS.map(({ key, icon }) => ({ icon, shown: uiText(lang, key), verified: uiText("en", key) }));
  return (
    <div className="empty-state">
      <div className="empty-state-glow" aria-hidden="true" />
      {temporary ? (
        // A temporary chat shows only the GMR logo, the title and the question box.
        <div className="tc-hero">
          <img className="tc-logo" src={gmrLogo} alt="GMR" />
          <div className="tc-eyebrow">
            <MessageCircleDashed size={13} aria-hidden="true" />
            {title}
          </div>
          <h3 className="tc-title">
            {temporaryHeadline}
            <span className="tc-subline">{temporarySubline}</span>
          </h3>
        </div>
      ) : (
        <>
          <div className="empty-state-icon">
            <img src={gmrLogo} alt="GMR" />
          </div>
          <h3 className="empty-state-title">{title}</h3>
        </>
      )}
      {temporary ? null : <p className="empty-state-message">{message ?? uiText(lang, "es_message")}</p>}

      {onSuggestedQuestion && !temporary ? (
        <div className="empty-state-starters">
          <div className="empty-state-starters-label">
            <Sparkles size={11} /> {uiText(lang, "tryAsking")}
          </div>
          <div className="empty-state-starter-grid">
            {starters.map(({ icon: Icon, shown, verified }) => (
              <button
                key={verified}
                type="button"
                className="empty-state-starter"
                onClick={() => onSuggestedQuestion(shown, shown !== verified ? verified : undefined)}
              >
                <span className="empty-state-starter-icon">
                  <Icon size={16} />
                </span>
                <span className="empty-state-starter-text">{shown}</span>
                <ArrowUpRight size={16} className="empty-state-starter-arrow" />
              </button>
            ))}
          </div>
        </div>
      ) : null}

      {temporary ? null : <div className="empty-state-capabilities">
        {CAPABILITY_CARDS.map(({ icon: Icon, title: cardTitle, accent, accentBg }) => (
          <div
            className="empty-state-capability-card"
            key={cardTitle}
            style={{ "--cap-accent": accent } as React.CSSProperties}
          >
            <div className="empty-state-capability-icon" style={{ color: accent, background: accentBg }}>
              <Icon size={19} />
            </div>
            <div className="empty-state-capability-title">{uiText(lang, cardTitle)}</div>
          </div>
        ))}
      </div>}

      {composer ? <div className="empty-state-composer">{composer}</div> : null}
    </div>
  );
};
