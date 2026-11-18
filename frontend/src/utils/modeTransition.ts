// Switching temporary chat on/off: the new look spreads out in a circle from the toggle that was
// clicked (on), or the temporary look folds back into it (off). Uses the View Transitions API --
// the browser snapshots the page, React applies the change synchronously, then the two snapshots
// are animated. Browsers without it, and people who prefer reduced motion, just switch instantly.
import { flushSync } from "react-dom";

export type Wrap = (update: () => void) => void;

/** Apply the update directly (no animation). */
export const runNow: Wrap = (update) => update();

type ViewTransition = { ready: Promise<void>; finished: Promise<void> };
type DocWithVT = Document & { startViewTransition?: (cb: () => void) => ViewTransition };

export function modeTransition(origin: { x: number; y: number } | null, direction: "in" | "out"): Wrap {
  return (update) => {
    const doc = document as DocWithVT;
    const reduced = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    if (!doc.startViewTransition || reduced || !origin) {
      update();
      return;
    }
    const root = document.documentElement;
    const cls = direction === "in" ? "tc-vt-in" : "tc-vt-out";
    root.classList.add(cls);
    let vt: ViewTransition;
    try {
      vt = doc.startViewTransition(() => flushSync(update));
    } catch {
      root.classList.remove(cls);
      update();
      return;
    }
    const { x, y } = origin;
    const r = Math.hypot(Math.max(x, window.innerWidth - x), Math.max(y, window.innerHeight - y));
    const small = `circle(0px at ${x}px ${y}px)`;
    const big = `circle(${r}px at ${x}px ${y}px)`;
    vt.ready
      .then(() => {
        root.animate(
          { clipPath: direction === "in" ? [small, big] : [big, small] },
          {
            duration: direction === "in" ? 700 : 560,
            easing: direction === "in" ? "cubic-bezier(0.65, 0, 0.35, 1)" : "cubic-bezier(0.55, 0, 0.7, 0.2)",
            pseudoElement: direction === "in" ? "::view-transition-new(root)" : "::view-transition-old(root)",
          }
        );
      })
      .catch(() => {});
    vt.finished.finally(() => root.classList.remove(cls));
  };
}

/** Centre of the clicked control, the circle's origin. */
export function originOf(e?: { currentTarget: EventTarget | null }): { x: number; y: number } | null {
  const el = e?.currentTarget as HTMLElement | null;
  if (!el?.getBoundingClientRect) return null;
  const r = el.getBoundingClientRect();
  return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
}
