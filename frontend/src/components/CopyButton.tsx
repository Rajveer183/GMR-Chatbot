import React, { useEffect, useRef, useState } from "react";
import { Copy, Check } from "lucide-react";
import { copyToClipboard, type CopyPayload } from "../utils/clipboard";

interface CopyButtonProps {
  /** Built lazily on click so large tables aren't serialised on every render. */
  getPayload: () => CopyPayload;
  label?: string;
  title?: string;
  /** "Copied" in the screen language. */
  copiedLabel?: string;
  /** Icon only, for tight spots like a metric card header. */
  compact?: boolean;
  className?: string;
}

export const CopyButton: React.FC<CopyButtonProps> = ({
  getPayload,
  label = "Copy",
  title = "Copy",
  copiedLabel = "Copied",
  compact = false,
  className = "summary-audio-btn",
}) => {
  const [copied, setCopied] = useState(false);
  const timerRef = useRef<number | null>(null);

  useEffect(() => () => {
    if (timerRef.current) window.clearTimeout(timerRef.current);
  }, []);

  const onClick = async (event: React.MouseEvent) => {
    event.stopPropagation();
    await copyToClipboard(getPayload());
    setCopied(true);
    if (timerRef.current) window.clearTimeout(timerRef.current);
    timerRef.current = window.setTimeout(() => setCopied(false), 1800);
  };

  return (
    <button
      type="button"
      className={`${className} ${copied ? "copied" : ""}`}
      onClick={onClick}
      aria-label={copied ? copiedLabel : title}
      title={copied ? copiedLabel : title}
    >
      {copied ? <Check size={compact ? 13 : 16} /> : <Copy size={compact ? 13 : 16} />}
      {compact ? null : <span>{copied ? copiedLabel : label}</span>}
    </button>
  );
};
