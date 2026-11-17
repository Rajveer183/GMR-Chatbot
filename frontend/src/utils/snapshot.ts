/** "Export as shown": an answer (or the whole chat) saved the way it looks on screen -- summary,
 * metric cards, charts, tables -- as a PNG image, a PDF (browser print -> Save as PDF) or a
 * self-contained HTML page. Built from the rendered DOM (ApexCharts draws SVG, which copies
 * as-is), so it needs no backend call and no query re-run.
 *
 * Tables export the page currently shown; the full table is the CSV / Excel download. */
import { toPng } from "html-to-image";

/** Controls that make no sense in a saved copy. Other buttons that carry text (e.g. the table's
 * sortable column headers) are kept as plain text -- see stripControls. */
const SKIP_SELECTOR = [
  "input", "select", "textarea",
  ".no-export", ".chat-answer-actions", ".chat-user-actions", ".feedback-wrap",
  ".summary-header-actions", ".table-pager", ".table-download", ".table-card-actions",
  ".apexcharts-toolbar", ".apexcharts-tooltip", ".apexcharts-xaxistooltip", ".apexcharts-yaxistooltip",
].join(", ");

export interface SnapshotPart {
  /** The rendered element to copy (an answer card, or the chat thread). */
  node: HTMLElement;
  /** Shown above it (the question that was asked), when given. */
  heading?: string;
}

function safeName(text: string): string {
  return (text || "answer").replace(/[^\p{L}\p{N}\-_ ]+/gu, "").trim().replace(/\s+/g, "_").slice(0, 60) || "answer";
}

export function snapshotFilename(title: string, ext: string): string {
  const d = new Date();
  const stamp = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
  return `${safeName(title)}_${stamp}.${ext}`;
}

function stripControls(clone: HTMLElement) {
  clone.querySelectorAll(SKIP_SELECTOR).forEach((el) => el.remove());
  clone.querySelectorAll("button").forEach((button) => {
    if (!button.textContent?.trim()) {
      button.remove();  // icon-only control (expand chart, close ...)
      return;
    }
    const span = document.createElement("span");
    span.className = button.className;
    span.append(...Array.from(button.childNodes));
    button.replaceWith(span);
  });
}

/** A detached copy: header + cloned content, controls removed, at the on-screen width. Chart
 * hover data is kept only for the HTML page (`withTips`); a picture or a print cannot hover. */
function buildClone(title: string, parts: SnapshotPart[], withTips = false): HTMLElement {
  const width = Math.max(...parts.map((p) => p.node.getBoundingClientRect().width), 640);
  const root = document.createElement("div");
  root.className = "snapshot-root";
  root.style.width = `${Math.ceil(width) + 48}px`;

  const header = document.createElement("div");
  header.className = "snapshot-header";
  const h = document.createElement("div");
  h.className = "snapshot-title";
  h.textContent = title;
  const meta = document.createElement("div");
  meta.className = "snapshot-meta";
  meta.textContent = `GMR AI Intelligence · exported ${new Date().toLocaleString()}`;
  header.append(h, meta);
  root.appendChild(header);

  for (const part of parts) {
    if (part.heading) {
      const q = document.createElement("div");
      q.className = "snapshot-question";
      q.textContent = part.heading;
      root.appendChild(q);
    }
    const clone = part.node.cloneNode(true) as HTMLElement;
    stripControls(clone);
    if (!withTips) clone.querySelectorAll("[data-chart-tip]").forEach((el) => el.removeAttribute("data-chart-tip"));
    root.appendChild(clone);
  }
  return root;
}

/** Every stylesheet on the page as text (cross-origin sheets, e.g. Google Fonts, as @import). */
function collectCss(): string {
  const imports: string[] = [];
  const rules: string[] = [];
  for (const sheet of Array.from(document.styleSheets)) {
    try {
      for (const rule of Array.from(sheet.cssRules)) {
        if (rule instanceof CSSImportRule) imports.push(rule.cssText);
        else rules.push(rule.cssText);
      }
    } catch {
      if (sheet.href) imports.push(`@import url("${sheet.href}");`);
    }
  }
  return [...imports, ...rules].join("\n");
}

/** The saved page's chart tooltips. The live ones are drawn by ApexCharts on mouse move, and the
 * saved page has no chart library, so this shows the figures the chart wrapper carries as JSON
 * (`data-chart-tip`, built by components/charts/exportTip.ts). The hovered point is the mark under
 * the mouse (bar, slice, heatmap cell, which ApexCharts tags with its data index `j`), or, on
 * line / area / column charts, the category nearest the mouse across the plot. The plot's size is
 * read from the chart's own clip rectangle (`gridRectMask`): its width is the grid width + 4 +
 * stroke, offset by -(stroke / 2 + 2). */
const TOOLTIP_SCRIPT = `(function () {
  var tip = document.createElement("div");
  tip.className = "snapshot-tip";
  tip.style.display = "none";
  document.body.appendChild(tip);
  function esc(s) { return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;"); }
  function hide() { tip.style.display = "none"; }
  function attr(el, name) { return el.getAttribute(name) != null ? el.getAttribute(name) : el.getAttribute(name.toLowerCase()); }
  function seriesIndexOf(mark) {
    var group = mark.closest(".apexcharts-series");
    var v = group ? attr(group, "data:realIndex") : null;
    if (v == null) v = mark.getAttribute("i");
    return v == null ? 0 : parseInt(v, 10) || 0;
  }
  function plotIndex(host, data, ev) {
    var inner = host.querySelector(".apexcharts-inner");
    var clip = host.querySelector('clipPath[id^="gridRectMask"] rect');
    if (!inner || !clip || !inner.getScreenCTM()) return -1;
    var x = parseFloat(clip.getAttribute("x")) || 0, y = parseFloat(clip.getAttribute("y")) || 0;
    var w = (parseFloat(clip.getAttribute("width")) || 0) + 2 * x;
    var h = (parseFloat(clip.getAttribute("height")) || 0) + 2 * y;
    var svg = inner.ownerSVGElement, pt = svg.createSVGPoint();
    pt.x = ev.clientX; pt.y = ev.clientY;
    var p = pt.matrixTransform(inner.getScreenCTM().inverse());
    var n = data.cats.length;
    var pad = 10;  // the end points' markers sit on the plot edge and spill past it
    if (w <= 0 || n === 0 || p.x < -pad || p.x > w + pad || p.y < -pad || p.y > h + pad) return -1;
    var px = Math.max(0, Math.min(w, p.x));
    if (data.mode === "on") return n === 1 ? 0 : Math.round(px / (w / (n - 1)));
    return Math.min(n - 1, Math.floor(px / (w / n)));
  }
  function row(s, j) {
    var dot = s.color ? '<span class="snapshot-tip-dot" style="background:' + esc(s.color) + '"></span>' : "";
    var name = s.name ? esc(s.name) + ": " : "";
    return '<div class="snapshot-tip-row">' + dot + name + "<b>" + esc(s.values[j]) + "</b></div>";
  }
  function show(ev) {
    var host = ev.target.closest && ev.target.closest("[data-chart-tip]");
    if (!host) return hide();
    var data;
    try { data = JSON.parse(host.getAttribute("data-chart-tip")); } catch (e) { return hide(); }
    var html = "";
    if (data.mode === "on" || data.mode === "between") {
      var j = plotIndex(host, data, ev);
      if (j < 0) return hide();
      html = data.series.filter(function (s) { return s.values[j] != null; }).map(function (s) { return row(s, j); }).join("");
      html = '<div class="snapshot-tip-title">' + esc(data.cats[j] || "") + "</div>" + html;
    } else {
      var mark = ev.target.closest("[j]");
      if (!mark || !host.contains(mark)) return hide();
      var k = parseInt(mark.getAttribute("j"), 10);
      var s = data.series[data.mode === "slice" ? 0 : seriesIndexOf(mark)];
      if (!s || s.values[k] == null) return hide();
      html = '<div class="snapshot-tip-title">' + esc(data.cats[k] || "") + "</div>" + row(s, k);
    }
    tip.innerHTML = html;
    tip.style.display = "block";
    var left = ev.clientX + 14, top = ev.clientY + 14;
    if (left + tip.offsetWidth > window.innerWidth - 8) left = ev.clientX - tip.offsetWidth - 14;
    if (top + tip.offsetHeight > window.innerHeight - 8) top = ev.clientY - tip.offsetHeight - 14;
    tip.style.left = Math.max(8, left) + "px";
    tip.style.top = Math.max(8, top) + "px";
  }
  document.addEventListener("mousemove", show);
  document.addEventListener("mouseleave", hide);
  window.addEventListener("scroll", hide, { passive: true });
})();`;

const TOOLTIP_CSS = `
.snapshot-tip { position: fixed; z-index: 1000; pointer-events: none; max-width: 320px;
  background: var(--card-bg, #fff); color: var(--text-main, #0f172a); border: 1px solid var(--border, #e2e8f0);
  border-radius: 8px; box-shadow: 0 6px 20px rgba(15, 23, 42, .14); padding: 8px 10px; font-size: 12px; line-height: 1.5; }
.snapshot-tip-title { font-weight: 600; margin-bottom: 4px; }
.snapshot-tip-row { display: flex; align-items: center; gap: 6px; white-space: nowrap; }
.snapshot-tip-dot { width: 8px; height: 8px; border-radius: 50%; flex: none; }
@media print { .snapshot-tip { display: none !important; } }`;

function buildHtml(title: string, parts: SnapshotPart[], withTips = false): string {
  const clone = buildClone(title, parts, withTips);
  const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;");
  return `<!doctype html>
<html lang="${document.documentElement.lang || "en"}">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>${esc(title)}</title>
<style>
${collectCss()}
html, body { background: var(--page-bg, #f8fafc); margin: 0; }
body { display: flex; justify-content: center; padding: 24px 0; }
.snapshot-root { max-width: 100%; }
@media print {
  @page { margin: 12mm; }
  body { padding: 0; display: block; }
  .snapshot-root { width: auto !important; box-shadow: none; }
}${withTips ? TOOLTIP_CSS : ""}
</style>
</head>
<body>
${clone.outerHTML}${withTips ? `
<script>
${TOOLTIP_SCRIPT}
</script>` : ""}
</body>
</html>`;
}

function triggerDownload(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.style.display = "none";
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

/** A self-contained web page that looks like the screen (opens offline in any browser). */
export function downloadSnapshotHtml(title: string, parts: SnapshotPart[]) {
  const html = buildHtml(title, parts, true);
  triggerDownload(new Blob([html], { type: "text/html;charset=utf-8" }), snapshotFilename(title, "html"));
}

/** Opens the browser's print dialog on the copy -- choose "Save as PDF". */
export function printSnapshot(title: string, parts: SnapshotPart[]): Promise<void> {
  const html = buildHtml(title, parts);
  return new Promise((resolve, reject) => {
    const frame = document.createElement("iframe");
    frame.setAttribute("aria-hidden", "true");
    Object.assign(frame.style, { position: "fixed", right: "0", bottom: "0", width: "0", height: "0", border: "0" });
    let done = false;
    const cleanup = () => window.setTimeout(() => frame.remove(), 1000);
    frame.onload = () => {
      if (done) return;
      done = true;
      const win = frame.contentWindow;
      if (!win) {
        frame.remove();
        reject(new Error("Could not prepare the PDF."));
        return;
      }
      // Give the web font a moment so the PDF does not use a fallback font.
      const fonts = (frame.contentDocument as Document & { fonts?: FontFaceSet }).fonts;
      const ready = fonts ? Promise.race([fonts.ready, new Promise((r) => window.setTimeout(r, 1500))]) : Promise.resolve();
      void ready.then(() => {
        win.addEventListener("afterprint", cleanup, { once: true });
        win.focus();
        win.print();
        window.setTimeout(() => frame.remove(), 120_000);
        resolve();
      });
    };
    frame.srcdoc = html;
    document.body.appendChild(frame);
  });
}

/** A PNG picture of the copy (2x for sharp text). */
export async function downloadSnapshotPng(title: string, parts: SnapshotPart[]): Promise<void> {
  const clone = buildClone(title, parts);
  const holder = document.createElement("div");
  Object.assign(holder.style, { position: "fixed", left: "-100000px", top: "0", pointerEvents: "none" });
  holder.appendChild(clone);
  document.body.appendChild(holder);
  try {
    const background = getComputedStyle(document.body).getPropertyValue("--page-bg").trim() || "#f8fafc";
    let dataUrl: string;
    try {
      dataUrl = await toPng(clone, { pixelRatio: 2, backgroundColor: background, cacheBust: true });
    } catch {
      // Web-font embedding can fail (offline / blocked); the picture is still right without it.
      dataUrl = await toPng(clone, { pixelRatio: 2, backgroundColor: background, skipFonts: true });
    }
    const blob = await (await fetch(dataUrl)).blob();
    triggerDownload(blob, snapshotFilename(title, "png"));
  } finally {
    holder.remove();
  }
}
