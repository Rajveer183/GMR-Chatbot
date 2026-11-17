// App-wide instant tooltips. The browser's native `title` tooltip waits ~1s before showing;
// this shows the same text immediately on hover/keyboard focus for every element in the UI.
//
// How: on hover, an element's `title` is moved to `data-tip-title` (so the native tooltip never
// fires) and its text is shown in one shared, fixed-position bubble. React only re-writes
// `title` when its value changes, and the next hover simply moves it again. Elements can also
// use `data-tip` directly; `data-tip-pos` = "right" | "below" | "above" picks the side.
// Snapshot exports put `data-tip-title` back as `title` (utils/snapshot.ts).

const STASH = "data-tip-title";
const GAP = 8;
const MARGIN = 8;

type Pos = "right" | "below" | "above";

let bubble: HTMLDivElement | null = null;
let current: Element | null = null;

function ensureBubble(): HTMLDivElement {
  if (bubble && document.body.contains(bubble)) return bubble;
  bubble = document.createElement("div");
  bubble.className = "instant-tip";
  bubble.setAttribute("role", "tooltip");
  bubble.setAttribute("aria-hidden", "true");
  document.body.appendChild(bubble);
  return bubble;
}

/** The nearest element (self or ancestor) carrying tooltip text; moves `title` aside first. */
function tipTarget(start: EventTarget | null): Element | null {
  if (!(start instanceof Element)) return null;
  const el = start.closest(`[title], [data-tip], [${STASH}]`);
  if (!el || el.closest(".instant-tip")) return null;
  const title = el.getAttribute("title");
  if (title !== null) {
    if (title.trim()) {
      el.setAttribute(STASH, title);
      // Keep an accessible name for icon-only controls that relied on the title.
      if (!el.hasAttribute("aria-label") && !el.textContent?.trim()) el.setAttribute("aria-label", title);
    } else {
      el.removeAttribute(STASH);
    }
    el.removeAttribute("title");
  }
  return el;
}

function tipText(el: Element): string {
  return (el.getAttribute("data-tip") ?? el.getAttribute(STASH) ?? "").trim();
}

function preferredPos(el: Element): Pos {
  const p = el.getAttribute("data-tip-pos");
  if (p === "right" || p === "below" || p === "above") return p;
  if (el.closest(".chat-sidebar-rail")) return "right";
  return "below";
}

function hide() {
  current = null;
  if (bubble) bubble.classList.remove("show");
}

function show(el: Element) {
  const text = tipText(el);
  // No tip while the control's own menu/popover is open.
  if (!text || (el.hasAttribute("aria-haspopup") && el.getAttribute("aria-expanded") === "true")) {
    hide();
    return;
  }
  current = el;
  const tip = ensureBubble();
  tip.textContent = text;
  tip.style.left = "0px";
  tip.style.top = "0px";
  tip.classList.add("show");

  const r = el.getBoundingClientRect();
  const tw = tip.offsetWidth;
  const th = tip.offsetHeight;
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  let pos = preferredPos(el);
  if (pos === "right" && r.right + GAP + tw > vw - MARGIN) pos = "below";
  if (pos === "below" && r.bottom + GAP + th > vh - MARGIN) pos = "above";
  if (pos === "above" && r.top - GAP - th < MARGIN) pos = "below";

  let left: number;
  let top: number;
  if (pos === "right") {
    left = r.right + GAP;
    top = r.top + r.height / 2 - th / 2;
  } else {
    left = r.left + r.width / 2 - tw / 2;
    top = pos === "below" ? r.bottom + GAP : r.top - GAP - th;
  }
  left = Math.max(MARGIN, Math.min(left, vw - tw - MARGIN));
  top = Math.max(MARGIN, Math.min(top, vh - th - MARGIN));
  tip.style.left = `${Math.round(left)}px`;
  tip.style.top = `${Math.round(top)}px`;
}

let installed = false;

export function installInstantTooltips() {
  if (installed || typeof document === "undefined") return;
  installed = true;

  document.addEventListener(
    "mouseover",
    (e) => {
      const el = tipTarget(e.target);
      if (el === current) return;
      if (el) show(el);
      else hide();
    },
    true,
  );
  document.addEventListener(
    "mouseout",
    (e) => {
      if (!current) return;
      const to = e.relatedTarget as Node | null;
      if (!to || !current.contains(to)) {
        // Moving onto another tip target is handled by the mouseover that follows.
        if (!(to instanceof Element) || !tipTarget(to)) hide();
      }
    },
    true,
  );
  document.addEventListener(
    "focusin",
    (e) => {
      const el = e.target instanceof Element && e.target.matches(":focus-visible") ? tipTarget(e.target) : null;
      if (el) show(el);
    },
    true,
  );
  document.addEventListener("focusout", hide, true);
  // A click usually opens/changes something; the tip would sit on top of it.
  document.addEventListener("mousedown", hide, true);
  document.addEventListener("keydown", hide, true);
  document.addEventListener("scroll", hide, true);
  window.addEventListener("resize", hide);
  window.addEventListener("blur", hide);
}
