import React, { useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { X } from "lucide-react";

interface ExpandableChartModalProps {
  open: boolean;
  title: string;
  subtitle?: string;
  onClose: () => void;
  /** A render function receives the pixel height actually available inside the
   * modal body, so the chart is drawn to fit instead of at a fixed height that
   * the modal (capped to the viewport) would clip. */
  children: React.ReactNode | ((availableHeight: number) => React.ReactNode);
}

export const ExpandableChartModal: React.FC<ExpandableChartModalProps> = ({
  open,
  title,
  subtitle,
  onClose,
  children,
}) => {
  const savedScrollTopRef = useRef<number | null>(null);
  const bodyRef = useRef<HTMLDivElement | null>(null);
  const [bodyHeight, setBodyHeight] = useState(0);

  // Track the body's inner height (minus padding) so the chart always fits the
  // modal, whatever the window size, and re-fits when the window is resized.
  useLayoutEffect(() => {
    if (!open) return;
    const el = bodyRef.current;
    if (!el) return;
    const measure = () => {
      const cs = window.getComputedStyle(el);
      const inner = el.clientHeight - parseFloat(cs.paddingTop) - parseFloat(cs.paddingBottom);
      setBodyHeight(Math.max(240, Math.floor(inner)));
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [open]);

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
        <div className="chart-modal-body" ref={bodyRef}>
          {typeof children === "function" ? (bodyHeight ? children(bodyHeight) : null) : children}
        </div>
      </div>
    </div>,
    document.body
  );
};
