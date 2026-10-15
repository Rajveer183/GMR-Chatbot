import React, { useEffect, useMemo, useRef, useState } from "react";
import { Play, Pause, Loader2 } from "lucide-react";
import { apiClient } from "../api/client";
import { CopyButton } from "./CopyButton";
import { summaryToPayload, type CopyPayload } from "../utils/clipboard";

interface SummaryCardProps {
  summary: string;
  interpretedIntent?: string;
  interpretedQuery?: string;
  /** Builds the whole answer (summary + metrics + tables) for the Copy button. */
  getFullCopyPayload?: () => CopyPayload;
  /** What Play reads aloud (summary + metrics + table preview); defaults to the summary. */
  speechText?: string;
}

function renderInlineMarkdown(text: string): React.ReactNode[] {
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  return parts.map((part, idx) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={idx}>{part.slice(2, -2)}</strong>;
    }
    return <React.Fragment key={idx}>{part}</React.Fragment>;
  });
}

function stripMarkdown(text: string) {
  return text.replace(/\*\*(.*?)\*\*/g, "$1").replace(/\[(.*?)\]\((.*?)\)/g, "$1");
}

export const SummaryCard: React.FC<SummaryCardProps> = ({ summary, interpretedIntent, interpretedQuery, getFullCopyPayload, speechText }) => {
  const [playing, setPlaying] = useState(false);
  const [loadingAudio, setLoadingAudio] = useState(false);
  const [audioError, setAudioError] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  // Caches the fetched clip against the exact text it was generated for, so
  // pause/play on the same summary never re-hits the API, but a genuinely
  // different summary (a different message) always fetches fresh audio.
  const cachedAudioRef = useRef<{ text: string; url: string } | null>(null);

  const plainTextSummary = useMemo(() => speechText || stripMarkdown(summary), [speechText, summary]);

  useEffect(() => {
    return () => {
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
      }
      if (cachedAudioRef.current) {
        URL.revokeObjectURL(cachedAudioRef.current.url);
        cachedAudioRef.current = null;
      }
    };
  }, []);

  // Text-to-speech now renders through the backend (Groq PlayAI TTS) instead
  // of the browser's built-in speechSynthesis -- that API's voice quality and
  // availability varies a lot browser-to-browser, whereas this always sounds
  // the same and works anywhere `<audio>` does.
  const toggleSpeech = async () => {
    setAudioError(null);

    if (playing && audioRef.current) {
      audioRef.current.pause();
      setPlaying(false);
      return;
    }

    if (audioRef.current && cachedAudioRef.current?.text === plainTextSummary) {
      audioRef.current.play();
      setPlaying(true);
      return;
    }

    setLoadingAudio(true);
    try {
      if (cachedAudioRef.current) {
        URL.revokeObjectURL(cachedAudioRef.current.url);
        cachedAudioRef.current = null;
      }
      const blob = await apiClient.textToSpeech(plainTextSummary);
      const url = URL.createObjectURL(blob);
      cachedAudioRef.current = { text: plainTextSummary, url };

      const audio = new Audio(url);
      audio.onended = () => setPlaying(false);
      audio.onerror = () => {
        setPlaying(false);
        setAudioError("Could not play the generated audio.");
      };
      audioRef.current = audio;
      await audio.play();
      setPlaying(true);
    } catch (err: any) {
      setAudioError(err?.message || "Could not generate audio for this summary.");
    } finally {
      setLoadingAudio(false);
    }
  };

  // With a full-answer payload, "Copy" takes the metrics and tables too; without
  // one it falls back to the summary text alone.
  const getCopyPayload = () => (getFullCopyPayload ? getFullCopyPayload() : summaryToPayload(summary));

  return (
    <section className="chat-result-section chat-summary-section">
      <div className="summary-header-row">
        <h4 className="chat-section-heading">Summary</h4>
        <div className="summary-header-actions">
          <CopyButton
            getPayload={getCopyPayload}
            label={getFullCopyPayload ? "Copy all" : "Copy"}
            title={getFullCopyPayload ? "Copy summary, metrics and tables" : "Copy summary"}
          />
          <button
            type="button"
            className={`summary-audio-btn ${playing ? "playing" : ""}`}
            onClick={toggleSpeech}
            disabled={loadingAudio}
            aria-label={playing ? "Pause summary audio" : "Play summary audio"}
            title={loadingAudio ? "Loading audio..." : playing ? "Pause audio" : "Play audio"}
          >
            {loadingAudio ? (
              <Loader2 size={16} className="spin" />
            ) : playing ? (
              <Pause size={16} />
            ) : (
              <Play size={16} />
            )}
            <span>{loadingAudio ? "Loading..." : playing ? "Pause" : "Play"}</span>
          </button>
        </div>
      </div>
      {interpretedQuery ? <p className="chat-meta-line">Interpreted query: {interpretedQuery}</p> : null}
      {!interpretedQuery && interpretedIntent ? (
        <p className="chat-meta-line">Interpreted intent: {interpretedIntent}</p>
      ) : null}
      <div className="chat-summary-text">{renderInlineMarkdown(summary)}</div>
      {audioError ? <p className="chat-speech-error">{audioError}</p> : null}
    </section>
  );
};
