import React, { useEffect, useMemo, useRef, useState } from "react";
import { uiText } from "../i18n/languages";
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
  /** The answer's language: headings and buttons follow it. */
  lang?: string;
}

// A figure worth picking out in a summary: grouped (4,214,678 / 42,14,678), decimal (3.36),
// a share or Indian unit (97.5% / 16.45L / 1.2Cr / 4.30 million), or a bare count of five or
// more digits. Dates and day counts ('27 Sep 2026', 'last 30 days') stay plain.
const FIGURE_RE = /(\b\d{1,3}(?:,\d{2,3})+(?:\.\d+)?\b(?:\s?(?:%|L\b|Cr\b|lakh|crore|million))?|\b\d+\.\d+\b(?:\s?(?:%|L\b|Cr\b|lakh|crore|million))?|\b\d+\s?%|\b\d{5,}\b)/g;

function highlightFigures(text: string, keyPrefix: string): React.ReactNode[] {
  return text.split(FIGURE_RE).map((part, idx) =>
    idx % 2 === 1 ? <strong key={`${keyPrefix}-${idx}`} className="summary-figure">{part}</strong>
      : <React.Fragment key={`${keyPrefix}-${idx}`}>{part}</React.Fragment>
  );
}

function renderInlineMarkdown(text: string): React.ReactNode[] {
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  return parts.flatMap((part, idx): React.ReactNode[] => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return [<strong key={idx}>{part.slice(2, -2)}</strong>];
    }
    // The answer's key figures stand out the way the dashboard's tiles do.
    return highlightFigures(part, String(idx));
  });
}

function stripMarkdown(text: string) {
  return text.replace(/\*\*(.*?)\*\*/g, "$1").replace(/\[(.*?)\]\((.*?)\)/g, "$1");
}

// The Groq voice (Orpheus) reads English only: a Hindi / Tamil / ... answer goes
// to the browser's own voice for that language instead of being read as noise.
const GROQ_VOICE_LANGS = new Set(["", "en", "hinglish"]);

const browserVoiceAvailable = () => typeof window !== "undefined" && "speechSynthesis" in window;

/**
 * Reads `text` with the browser's built-in voice. Used for non-English answers and
 * whenever the backend voice fails (its daily character quota is small). It never
 * skips text, but sounds like whatever voices the OS has. Returns false when the
 * browser has no speech support.
 */
function speakInBrowser(text: string, lang: string | undefined, onDone: () => void): boolean {
  if (!browserVoiceAvailable()) return false;
  const synth = window.speechSynthesis;
  synth.cancel();
  const locale = !lang || lang === "en" || lang === "hinglish" ? "en-IN" : `${lang}-IN`;
  const voices = synth.getVoices();
  const voice = voices.find((v) => v.lang.replace("_", "-") === locale)
    ?? voices.find((v) => v.lang.toLowerCase().startsWith(locale.slice(0, 2)));
  const clean = text.replace(/_/g, " ").replace(/\s+\/\s+/g, ", ").replace(/\s+/g, " ").trim();
  // Chrome cuts a single long utterance off after ~15s: queue sentence-sized pieces.
  const pieces: string[] = [];
  for (const s of clean.split(/(?<=[.!?।])\s+/)) {
    const last = pieces.length - 1;
    if (last >= 0 && pieces[last].length + s.length < 220) pieces[last] += ` ${s}`;
    else pieces.push(s);
  }
  pieces.forEach((piece, i) => {
    const u = new SpeechSynthesisUtterance(piece);
    u.lang = voice?.lang ?? locale;
    if (voice) u.voice = voice;
    u.rate = 1.05;
    if (i === pieces.length - 1) u.onend = onDone;
    u.onerror = (e) => {
      if (e.error !== "interrupted" && e.error !== "canceled") onDone();
    };
    synth.speak(u);
  });
  return true;
}

export const SummaryCard: React.FC<SummaryCardProps> = ({ summary, interpretedIntent, interpretedQuery, getFullCopyPayload, speechText, lang }) => {
  const [playing, setPlaying] = useState(false);
  const [loadingAudio, setLoadingAudio] = useState(false);
  const [audioError, setAudioError] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  // Caches the fetched clip against the exact text it was generated for, so
  // pause/play on the same summary never re-hits the API, but a genuinely
  // different summary (a different message) always fetches fresh audio.
  const cachedAudioRef = useRef<{ text: string; url: string } | null>(null);
  // Text the browser voice is reading (or paused on); null when it isn't in use.
  const browserTextRef = useRef<string | null>(null);

  const plainTextSummary = useMemo(() => speechText || stripMarkdown(summary), [speechText, summary]);

  useEffect(() => {
    return () => {
      if (browserTextRef.current !== null && browserVoiceAvailable()) {
        window.speechSynthesis.cancel();
        browserTextRef.current = null;
      }
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

    // Browser voice in use: pause / resume it.
    if (browserTextRef.current !== null && browserVoiceAvailable()) {
      if (playing) {
        window.speechSynthesis.pause();
        setPlaying(false);
        return;
      }
      if (browserTextRef.current === plainTextSummary && window.speechSynthesis.paused) {
        window.speechSynthesis.resume();
        setPlaying(true);
        return;
      }
    }

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

    const readInBrowser = () => {
      const started = speakInBrowser(plainTextSummary, lang, () => {
        browserTextRef.current = null;
        setPlaying(false);
      });
      if (started) {
        browserTextRef.current = plainTextSummary;
        setPlaying(true);
      }
      return started;
    };

    if (!GROQ_VOICE_LANGS.has(lang ?? "") && readInBrowser()) return;

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
        setAudioError(uiText(lang, "audioPlayFailed"));
      };
      audioRef.current = audio;
      await audio.play();
      setPlaying(true);
    } catch (err: any) {
      // Backend voice unavailable (quota, network): read it with the browser's voice instead.
      if (!readInBrowser()) setAudioError(err?.message || uiText(lang, "audioFailed"));
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
        <h4 className="chat-section-heading">{uiText(lang, "summary")}</h4>
        <div className="summary-header-actions">
          <CopyButton
            getPayload={getCopyPayload}
            label={uiText(lang, getFullCopyPayload ? "copyAll" : "copy")}
            title={uiText(lang, getFullCopyPayload ? "copyAllTitle" : "copySummaryTitle")}
            copiedLabel={uiText(lang, "copied")}
          />
          <button
            type="button"
            className={`summary-audio-btn ${playing ? "playing" : ""}`}
            onClick={toggleSpeech}
            disabled={loadingAudio}
            aria-label={uiText(lang, playing ? "pauseTitle" : "playTitle")}
            title={loadingAudio ? uiText(lang, "loadingAudio") : uiText(lang, playing ? "pauseTitle" : "playTitle")}
          >
            {loadingAudio ? (
              <Loader2 size={16} className="spin" />
            ) : playing ? (
              <Pause size={16} />
            ) : (
              <Play size={16} />
            )}
            <span>{loadingAudio ? uiText(lang, "loadingAudio") : uiText(lang, playing ? "pause" : "play")}</span>
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
