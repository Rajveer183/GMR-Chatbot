import React, { useEffect, useState } from "react";
import { ThumbsUp, ThumbsDown } from "lucide-react";
import { apiClient, FEEDBACK_REASONS, type FeedbackReason, type FeedbackRating } from "../api/client";
import { uiText } from "../i18n/languages";
import type { UIKey } from "../i18n/uiStrings";

interface FeedbackButtonsProps {
  traceId: string;
  sessionId?: string;
  /** The vote already stored for this answer (after a reload). */
  initial?: FeedbackRating;
  lang?: string;
  onChange?: (traceId: string, rating: FeedbackRating) => void;
}

/** Thumbs up / down on an answer. A downvote asks (optionally) what was wrong. Votes feed the
 * learning reports (/api/insights/quality) -- they never change the answer itself. Clicking the
 * active thumb again withdraws the vote. */
export const FeedbackButtons: React.FC<FeedbackButtonsProps> = ({ traceId, sessionId, initial, lang, onChange }) => {
  const [rating, setRating] = useState<FeedbackRating | undefined>(initial);
  const [askReason, setAskReason] = useState(false);
  const [reason, setReason] = useState<FeedbackReason | undefined>();
  const [comment, setComment] = useState("");
  const [status, setStatus] = useState<"idle" | "sent" | "error">("idle");

  useEffect(() => setRating(initial), [initial]);

  const send = async (next: FeedbackRating, why?: FeedbackReason, note?: string) => {
    const previous = rating;
    setRating(next === "clear" ? undefined : next);
    try {
      await apiClient.sendFeedback({ trace_id: traceId, rating: next, session_id: sessionId, reason: why, comment: note || undefined });
      setStatus(next === "clear" ? "idle" : "sent");
      onChange?.(traceId, next);
    } catch {
      setRating(previous);
      setStatus("error");
    }
  };

  const clickUp = () => {
    setAskReason(false);
    void send(rating === "up" ? "clear" : "up");
  };

  const clickDown = () => {
    if (rating === "down") {
      setAskReason(false);
      void send("clear");
      return;
    }
    void send("down");
    setAskReason(true);
  };

  const submitReason = () => {
    setAskReason(false);
    void send("down", reason, comment.trim());
  };

  return (
    <div className="feedback-wrap">
      <div className="feedback-buttons" role="group" aria-label={uiText(lang, "helpful")}>
        <button
          type="button"
          className={`feedback-btn ${rating === "up" ? "active up" : ""}`}
          onClick={clickUp}
          aria-pressed={rating === "up"}
          title={uiText(lang, "helpful")}
          aria-label={uiText(lang, "helpful")}
        >
          <ThumbsUp size={14} />
        </button>
        <button
          type="button"
          className={`feedback-btn ${rating === "down" ? "active down" : ""}`}
          onClick={clickDown}
          aria-pressed={rating === "down"}
          title={uiText(lang, "notHelpful")}
          aria-label={uiText(lang, "notHelpful")}
        >
          <ThumbsDown size={14} />
        </button>
        {status === "sent" && !askReason ? <span className="feedback-note">{uiText(lang, "feedbackThanks")}</span> : null}
        {status === "error" ? <span className="feedback-note error">{uiText(lang, "feedbackFailed")}</span> : null}
      </div>
      {askReason ? (
        <div className="feedback-reason">
          <span className="feedback-reason-label">{uiText(lang, "feedbackWhat")}</span>
          <div className="feedback-chips">
            {FEEDBACK_REASONS.map((r) => (
              <button
                key={r}
                type="button"
                className={`feedback-chip ${reason === r ? "selected" : ""}`}
                onClick={() => setReason(reason === r ? undefined : r)}
              >
                {uiText(lang, `fb_${r}` as UIKey)}
              </button>
            ))}
          </div>
          <div className="feedback-comment-row">
            <input
              type="text"
              className="feedback-comment"
              maxLength={500}
              value={comment}
              placeholder={uiText(lang, "feedbackComment")}
              onChange={(e) => setComment(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") submitReason();
              }}
            />
            <button type="button" className="summary-audio-btn" onClick={submitReason}>
              {uiText(lang, "feedbackSend")}
            </button>
          </div>
        </div>
      ) : null}
    </div>
  );
};
