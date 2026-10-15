import React, { useEffect, useRef } from "react";
import { createPortal } from "react-dom";
import { X } from "lucide-react";

interface ExpandableChartModalProps {
  open: boolean;
  title: string;
  subtitle?: string;
  onClose: () => void;
  children: React.ReactNode;
}

export const ExpandableChartModal: React.FC<ExpandableChartModalProps> = ({
  open,
  title,
  subtitle,
  onClose,
  children,
}) => {
  const savedScrollTopRef = useRef<number | null>(null);

  useEffect(() => {
    if (!open) return;
    const scrollContainer = document.querySelector(".chat-messages") as HTMLElement | null;
    if (scrollContainer) {
      savedScrollTopRef.current = scrollContainer.scrollTop;
    }
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      const container = document.querySelector(".chat-messages") as HTMLElement | null;
      if (container && savedScrollTopRef.current !== null) {
        container.scrollTop = savedScrollTopRef.current;
      }
      savedScrollTopRef.current = null;
    };
  }, [open, onClose]);

  if (!open) return null;

  // Ported directly to document.body -- a fixed-position overlay mounted
  // inline in the message tree is at the mercy of every ancestor between it
  // and the page root: any one of them gaining a transform/filter/perspective
  // (an entrance animation, e.g.) silently re-anchors "fixed" to that
  // ancestor instead of the viewport, breaking the overlay. A portal makes
  // that impossible regardless of what animations exist above it, now or later.
  return createPortal(
    <div className="chart-modal-overlay" onClick={onClose} role="presentation">
      <div className="chart-modal" onClick={(e) => e.stopPropagation()} role="dialog" aria-modal="true" aria-label={title}>
        <div className="chart-modal-header">
          <div>
            <h3 className="chart-modal-title">{title}</h3>
            {subtitle ? <p className="chart-modal-subtitle">{subtitle}</p> : null}
          </div>
          <button type="button" className="chart-modal-close" onClick={onClose} aria-label="Close chart">
            <X size={18} />
          </button>
        </div>
        <div className="chart-modal-body">{children}</div>
      </div>
    </div>,
    document.body
  );
};
