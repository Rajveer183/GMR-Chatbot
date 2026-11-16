import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import { Mic, Send } from "lucide-react";
import SpeechRecognition, { useSpeechRecognition } from "react-speech-recognition";
import { TypingIndicator } from "./TypingIndicator";

// The question box lives in its own component so a keystroke (or a live speech
// transcript update) re-renders only this input bar. When the text state sat in
// AIWorkspace, every character re-rendered the whole chat -- every past answer's
// charts and full result tables -- which made typing visibly lag in long sessions.

export interface ChatInputHandle {
  clear: () => void;
  focus: () => void;
}

interface ChatInputProps {
  loading: boolean;
  onSubmit: (text: string) => void;
  placeholder: string;
  micOnLabel: string;
  micOffLabel: string;
  /** BCP-47 locale for speech recognition. */
  speechLocale: string;
  /** Render in the page flow (inside the welcome card) instead of the fixed bottom bar. */
  inline?: boolean;
}

export const ChatInput = forwardRef<ChatInputHandle, ChatInputProps>(
  ({ loading, onSubmit, placeholder, micOnLabel, micOffLabel, speechLocale, inline = false }, ref) => {
    const [question, setQuestion] = useState("");
    const [speechError, setSpeechError] = useState<string | null>(null);
    const inputRef = useRef<HTMLTextAreaElement>(null);
    const { transcript, listening: speechActive, resetTranscript, browserSupportsSpeechRecognition } = useSpeechRecognition();

    // Start at one line and grow with the text (up to the CSS max-height).
    useEffect(() => {
      const el = inputRef.current;
      if (!el) return;
      el.style.height = "auto";
      el.style.height = `${el.scrollHeight}px`;
    }, [question]);

    useImperativeHandle(ref, () => ({
      clear: () => setQuestion(""),
      focus: () => {
        const el = inputRef.current;
        if (!el) return;
        try {
          el.focus({ preventScroll: true });
        } catch {
          el.focus();
        }
      },
    }));

    // react-speech-recognition keeps the live transcript in its own internal
    // state; mirror it into the question box as it updates so typing/voice
    // input share the same field.
    useEffect(() => {
      if (speechActive) setQuestion(transcript);
    }, [transcript, speechActive]);

    useEffect(() => {
      return () => {
        SpeechRecognition.stopListening();
      };
    }, []);

    // Voice input runs on react-speech-recognition, a wrapper around the
    // browser's built-in Web Speech API (SpeechRecognition).
    const toggleSpeech = () => {
      if (!browserSupportsSpeechRecognition) {
        setSpeechError("Speech recognition is not supported in this browser.");
        return;
      }
      if (speechActive) {
        SpeechRecognition.stopListening();
        return;
      }
      setSpeechError(null);
      resetTranscript();
      setQuestion("");
      SpeechRecognition.startListening({ continuous: true, language: speechLocale }).catch(() => {
        setSpeechError("Unable to start speech recognition.");
      });
    };

    return (
      <div className={`chat-input-bar${inline ? " inline" : ""}`}>
        <div className="chat-input-inner">
          <div className="chat-input-shell">
            <textarea
              ref={inputRef}
              className="chat-textarea"
              rows={1}
              value={question}
              disabled={loading}
              placeholder={placeholder}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  onSubmit(question);
                }
              }}
            />
            <button
              className={`chat-icon-btn mic-btn ${speechActive ? "active" : ""}`}
              type="button"
              onClick={toggleSpeech}
              title={speechActive ? micOffLabel : micOnLabel}
            >
              <Mic size={18} />
            </button>
            <button className="chat-send-btn" type="button" onClick={() => onSubmit(question)} disabled={loading || speechActive}>
              {loading ? <TypingIndicator /> : <Send size={14} />}
            </button>
          </div>
        </div>
        {speechError ? <div className="chat-speech-error">{speechError}</div> : null}
      </div>
    );
  }
);
ChatInput.displayName = "ChatInput";
