/** "Export as shown": an answer (or the whole chat) saved the way it looks on screen -- summary,
 * metric cards, charts, tables -- as a PNG image, an A4 PDF or a self-contained HTML page, all
 * downloaded directly, headed by the GMR logo, and with no site address anywhere in them. Built from the rendered DOM (ApexCharts draws SVG, which copies
 * as-is), so it needs no backend call and no query re-run.
 *
 * Tables export the page currently shown; the full table is the CSV / Excel download. */
import { getFontEmbedCSS, toBlob, toCanvas, toSvg } from "html-to-image";
import gmrLogo from "../assets/gmr_logo.png";

/** Controls that make no sense in a saved copy. Other buttons that carry text (e.g. the table's
 * sortable column headers) are kept as plain text -- see stripControls. */
const SKIP_SELECTOR = [
  "input", "select", "textarea",
  ".no-export", ".chat-answer-actions", ".chat-user-actions", ".feedback-wrap",
  ".summary-header-actions", ".table-pager", ".table-download", ".table-card-actions",
  ".apexcharts-toolbar", ".apexcharts-tooltip", ".apexcharts-xaxistooltip", ".apexcharts-yaxistooltip",
].join(", ");

let logoData: Promise<string> | null = null;

/** The GMR logo as a data URI, so the saved HTML shows it offline and the PDF / PNG copies do not
 * depend on a network fetch. Loaded once per page load; falls back to the asset URL on failure. */
export function prepareSnapshotLogo(): Promise<string> {
  if (!logoData) {
    logoData = fetch(gmrLogo)
      .then((r) => (r.ok ? r.blob() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((blob) => new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result));
        reader.onerror = () => reject(reader.error);
        reader.readAsDataURL(blob);
      }))
      .catch((err) => {
        console.warn("[ui] export: logo not embedded:", err);
        logoData = null;
        return gmrLogo;
      });
  }
  return logoData;
}

export interface SnapshotPart {
  /** The rendered element to copy (an answer card, or the chat thread). */
  node: HTMLElement;
  /** Shown above it (the question that was asked), when given. */
  heading?: string;
}

function safeName(text: string): string {
  // \p{M}: vowel signs / viramas (Hindi, Tamil ...) are part of the word, not punctuation.
  return (text || "answer").replace(/[^\p{L}\p{M}\p{N}\-_ ]+/gu, "").trim().replace(/\s+/g, "_").slice(0, 60) || "answer";
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
function buildClone(title: string, parts: SnapshotPart[], logo: string, withTips = false): HTMLElement {
  const width = Math.max(...parts.map((p) => p.node.getBoundingClientRect().width), 640);
  const logoUrl = new URL(gmrLogo, window.location.href).href;
  const root = document.createElement("div");
  root.className = "snapshot-root";
  root.style.width = `${Math.ceil(width) + 48}px`;

  const header = document.createElement("div");
  header.className = "snapshot-header";
  const img = document.createElement("img");
  img.className = "snapshot-logo";
  img.src = logo;
  img.alt = "GMR";
  const text = document.createElement("div");
  text.className = "snapshot-header-text";
  const h = document.createElement("div");
  h.className = "snapshot-title";
  h.textContent = title;
  const meta = document.createElement("div");
  meta.className = "snapshot-meta";
  meta.textContent = `GMR AI Intelligence · exported ${new Date().toLocaleString()}`;
  text.append(h, meta);
  header.append(img, text);
  root.appendChild(header);

  for (const part of parts) {
    if (part.heading) {
      const q = document.createElement("div");
      q.className = "snapshot-question";
      q.textContent = part.heading;
      root.appendChild(q);
    }
    const clone = part.node.cloneNode(true) as HTMLElement;
    // The answer's own GMR avatar points at the dev / app server ("/src/assets/..."), which a saved
    // copy cannot reach; use the embedded logo.
    clone.querySelectorAll("img").forEach((img) => {
      if (img.src === logoUrl) img.setAttribute("src", logo);
    });
    // Instant tooltips (utils/instantTooltips.ts) moved hovered titles aside; put them back.
    clone.querySelectorAll("[data-tip-title]").forEach((el) => {
      el.setAttribute("title", el.getAttribute("data-tip-title") ?? "");
      el.removeAttribute("data-tip-title");
    });
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
      if (data.cards && data.cards[j]) return place(data.cards[j], ev);
      html = data.series.filter(function (s) { return s.values[j] != null; }).map(function (s) { return row(s, j); }).join("");
      html = '<div class="snapshot-tip-title">' + esc(data.cats[j] || "") + "</div>" + html;
    } else {
      var mark = ev.target.closest("[j]");
      if (!mark || !host.contains(mark)) return hide();
      var k = parseInt(mark.getAttribute("j"), 10);
      if (data.cards && data.cards[k]) return place(data.cards[k], ev);
      var s = data.series[data.mode === "slice" ? 0 : seriesIndexOf(mark)];
      if (!s || s.values[k] == null) return hide();
      html = '<div class="snapshot-tip-title">' + esc(data.cats[k] || "") + "</div>" + row(s, k);
    }
    place(html, ev);
  }
  function place(html, ev) {
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

function buildHtml(title: string, parts: SnapshotPart[], logo: string, withTips = false): string {
  const clone = buildClone(title, parts, logo, withTips);
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

/** Saves a file through a link click. Browsers trust a download only while the click that asked
 * for it is still recent (user activation, about 5 s in Chrome); a later one counts as automatic.
 * A slow PNG is still saved directly (the user asked for no extra click): if Chrome asks once
 * "Download multiple files?", Allow is remembered for the site. */
export function triggerDownload(blob: Blob, filename: string) {
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
export async function downloadSnapshotHtml(title: string, parts: SnapshotPart[]) {
  const html = buildHtml(title, parts, await prepareSnapshotLogo(), true);
  triggerDownload(new Blob([html], { type: "text/html;charset=utf-8" }), snapshotFilename(title, "html"));
}

function withTimeout<T>(work: Promise<T>, ms: number, message: string): Promise<T> {
  let timer = 0;
  const limit = new Promise<never>((_, reject) => { timer = window.setTimeout(() => reject(new Error(message)), ms); });
  return Promise.race([work, limit]).finally(() => window.clearTimeout(timer));
}

/** html-to-image rejects with the bare `error` Event of the <img> it loads its SVG copy into when
 * the browser refuses that copy -- no message at all. Name what failed instead. */
function describe(err: unknown): string {
  if (typeof Event !== "undefined" && err instanceof Event) return `the browser refused the page copy (image ${err.type})`;
  if (err instanceof Error) return err.message || err.name;
  return String(err || "unknown error");
}

const XML_NAME = /^[A-Za-z_][\w.-]*$/;
const XML_KEEP = new Set(["xlink:href", "xml:space", "xml:lang", "xmlns:xlink"]);
const FOREIGN_TAGS = new Set(["IFRAME", "VIDEO", "AUDIO", "OBJECT", "EMBED", "CANVAS"]);

/** Makes the copy safe to put in an SVG image: drops what does not belong to the answer (elements a
 * browser extension adds, custom / shadow-DOM elements, embedded media) and attributes that are not
 * plain XML names (ApexCharts' `data:realIndex`, extension markers), any of which can make the
 * browser reject the whole image. Only the picture uses this; the screen is untouched. */
function sanitiseForImage(root: HTMLElement) {
  root.querySelectorAll("*").forEach((el) => {
    if (FOREIGN_TAGS.has(el.tagName.toUpperCase()) || el.tagName.includes("-")) {
      el.remove();
      return;
    }
    for (const attr of Array.from(el.attributes)) {
      if (!XML_NAME.test(attr.name) && !XML_KEEP.has(attr.name)) el.removeAttribute(attr.name);
    }
  });
}

/** Why the copy cannot be drawn, for the error message and the console: whether the SVG
 * html-to-image builds is well-formed, and how big it is. */
async function diagnose(node: HTMLElement): Promise<string> {
  try {
    const url = await toSvg(node, { skipFonts: true });
    const svg = decodeURIComponent(url.slice(url.indexOf(",") + 1));
    const parseError = new DOMParser().parseFromString(svg, "image/svg+xml").querySelector("parsererror");
    const size = `${(svg.length / 1_048_576).toFixed(1)} MB`;
    console.error("[ui] PNG export diagnosis:", { size, parseError: parseError?.textContent ?? null, svgStart: svg.slice(0, 400) });
    return parseError ? `invalid markup in the page copy (${parseError.textContent?.trim().split("\n")[0].slice(0, 160)})` : `page copy ${size}`;
  } catch (err) {
    return `copy could not be built: ${describe(err)}`;
  }
}

let fontCss: Promise<string> | null = null;

/** The page's web fonts as embeddable CSS (each font file fetched and inlined). The same for every
 * export, so it is built once per page load -- the export menu starts it when it opens, and a
 * PNG export then skips the slowest step. Empty when the fonts cannot be fetched (the picture is
 * then drawn with the fallback font); a failure is retried on the next export. */
export function prepareSnapshotFonts(): Promise<string> {
  void prepareSnapshotLogo();
  if (!fontCss) {
    fontCss = withTimeout(getFontEmbedCSS(document.body), 15_000, "timed out").catch((err) => {
      console.warn("[ui] PNG export: web font not embedded:", describe(err));
      fontCss = null;
      return "";
    });
  }
  return fontCss;
}

type PngOptions = Parameters<typeof toBlob>[1];

/** Draws the copy off screen and hands it to `draw` (2x for sharp text; html-to-image scales a very
 * long chat down to the browser's canvas limit). `draw` returns null when the result came out
 * empty (too large for the browser).
 *
 * The copy is cleaned first (sanitiseForImage): a browser extension's attribute in the page made
 * every uncleaned attempt fail, and cleaning changes nothing visible. Tried in turn, stopping at
 * the first that works: with the web font; without it; at 1x. If none works the error says which
 * step failed and why (see diagnose). */
async function drawSnapshot<T>(
  title: string,
  parts: SnapshotPart[],
  what: string,
  draw: (clone: HTMLElement, options: PngOptions, background: string) => Promise<T | null>,
): Promise<T> {
  const [fonts, logo] = await Promise.all([prepareSnapshotFonts(), prepareSnapshotLogo()]);
  const holder = document.createElement("div");
  Object.assign(holder.style, { position: "fixed", left: "-100000px", top: "0", pointerEvents: "none" });
  document.body.appendChild(holder);
  const background = getComputedStyle(document.body).getPropertyValue("--page-bg").trim() || "#f8fafc";
  const clone = buildClone(title, parts, logo);
  sanitiseForImage(clone);
  holder.appendChild(clone);
  const attempts: [string, PngOptions][] = [
    ...(fonts ? [["with font", { pixelRatio: 2, backgroundColor: background, fontEmbedCSS: fonts }] as [string, PngOptions]] : []),
    ["without font", { pixelRatio: 2, backgroundColor: background, skipFonts: true }],
    ["at 1x", { pixelRatio: 1, backgroundColor: background, skipFonts: true }],
  ];
  const failures: string[] = [];
  try {
    for (const [label, options] of attempts) {
      try {
        const result = await withTimeout(draw(clone, options, background), 45_000, "timed out");
        if (result) {
          if (failures.length) console.warn(`[ui] ${what} export recovered:`, label, "after", failures);
          return result;
        }
        failures.push(`${label}: empty image (too large for the browser)`);
      } catch (err) {
        console.error(`[ui] ${what} export attempt failed (${label}):`, err);
        failures.push(`${label}: ${describe(err)}`);
      }
    }
    const why = await diagnose(clone);
    throw new Error(`Could not draw the ${what} -- ${why}. Tried: ${failures.join("; ")}.`);
  } finally {
    holder.remove();
  }
}

/** A PNG picture of the copy. Returns the file; the caller saves it (triggerDownload). */
export async function renderSnapshotPng(title: string, parts: SnapshotPart[]): Promise<{ blob: Blob; filename: string }> {
  const blob = await drawSnapshot(title, parts, "PNG", async (clone, options) => {
    const b = await toBlob(clone, options);
    return b && b.size ? b : null;
  });
  return { blob, filename: snapshotFilename(title, "png") };
}

/* ---------------------------------------------------------------------------------------------
 * PDF: built here and downloaded directly, not through the browser's print dialog. The dialog
 * cannot control the destination: Chrome stamps its own header / footer (date, page title, the
 * site address) on every page, and Windows' "Microsoft Print to PDF" printer keeps writing the
 * file after the dialog closes, so opening it straight away shows "Failed to load PDF document".
 * A download is complete before the browser lists it.
 *
 * The pages are the same picture as the PNG, cut into A4 pages at gaps between blocks (rows,
 * cards, charts, paragraphs) so nothing is sliced through when it can be avoided.
 * ------------------------------------------------------------------------------------------- */

const A4 = { w: 595.28, h: 841.89, margin: 28 };  // points (1/72 in)

/** Where to end each page, in CSS px from the top of the copy. A page ends at the lowest point
 * that does not cut through a block, trying whole blocks first (a card that fits on a page moves
 * to the next page rather than splitting), then smaller ones; only a block taller than that is cut. */
function pageBreaks(clone: HTMLElement, total: number, pageH: number): number[] {
  const top = clone.getBoundingClientRect().top;
  const boxes: { a: number; b: number; h: number }[] = [];
  clone.querySelectorAll("*").forEach((el) => {
    if (el.parentElement?.closest("svg")) return;  // a chart is one block
    const r = el.getBoundingClientRect();
    if (r.height <= 1 || r.height > pageH * 0.9) return;
    boxes.push({ a: r.top - top, b: r.bottom - top, h: r.height });
  });
  const tiers = [0.9, 0.35, 0.1].map((f) => f * pageH);
  const ends: number[] = [];
  let y0 = 0;
  while (total - y0 > pageH) {
    const limit = y0 + pageH;
    const floor = y0 + pageH * 0.4;  // a page is never left less than 40% full
    const near = boxes.filter((x) => x.b > floor && x.a < limit);
    const candidates = new Set<number>([limit]);
    for (const x of near) {
      if (x.a > floor && x.a <= limit) candidates.add(x.a);
      if (x.b > floor && x.b <= limit) candidates.add(x.b);
    }
    const sorted = Array.from(candidates).sort((p, q) => q - p);
    let end = limit;
    for (const maxH of tiers) {
      const found = sorted.find((y) => !near.some((x) => x.h <= maxH && x.a < y - 0.5 && x.b > y + 0.5));
      if (found !== undefined) { end = found; break; }
    }
    ends.push(end);
    y0 = end;
  }
  ends.push(total);
  return ends;
}

/** A CSS colour as 0..1 RGB, read back from a 1x1 canvas so any CSS syntax works. */
function rgbOf(color: string): [number, number, number] {
  const c = document.createElement("canvas");
  c.width = c.height = 1;
  const ctx = c.getContext("2d");
  if (!ctx) return [1, 1, 1];
  ctx.fillStyle = "#ffffff";
  ctx.fillStyle = color;
  ctx.fillRect(0, 0, 1, 1);
  const [r, g, b] = ctx.getImageData(0, 0, 1, 1).data;
  return [r / 255, g / 255, b / 255];
}

function canvasJpeg(canvas: HTMLCanvasElement): Promise<Uint8Array | null> {
  return new Promise((resolve) => {
    canvas.toBlob((b) => {
      if (!b || !b.size) return resolve(null);
      void b.arrayBuffer().then((buf) => resolve(new Uint8Array(buf)), () => resolve(null));
    }, "image/jpeg", 0.92);
  });
}

/** A PDF text string that keeps any script (UTF-16BE with a byte-order mark, as hex). */
function pdfText(text: string): string {
  let hex = "FEFF";
  for (let i = 0; i < text.length; i++) hex += text.charCodeAt(i).toString(16).padStart(4, "0").toUpperCase();
  return `<${hex}>`;
}

interface PdfPage { jpeg: Uint8Array; pxW: number; pxH: number; w: number; h: number }

/** A minimal PDF 1.4: one JPEG per page on the page colour, with a "n / N" page number. */
function buildPdf(pages: PdfPage[], background: [number, number, number], title: string): Blob {
  const enc = new TextEncoder();
  const chunks: Uint8Array[] = [];
  const offsets: number[] = [];
  let length = 0;
  const push = (data: string | Uint8Array) => {
    const bytes = typeof data === "string" ? enc.encode(data) : data;
    chunks.push(bytes);
    length += bytes.length;
  };
  const obj = (id: number, dict: string, stream?: Uint8Array) => {
    offsets[id] = length;
    if (stream) {
      push(`${id} 0 obj\n${dict}\nstream\n`);
      push(stream);
      push("\nendstream\nendobj\n");
    } else {
      push(`${id} 0 obj\n${dict}\nendobj\n`);
    }
  };
  const n = (v: number) => (Math.round(v * 100) / 100).toString();
  const { w: W, h: H, margin: M } = A4;
  const pageIds = pages.map((_, i) => 5 + i * 3);

  push("%PDF-1.4\n%âãÏÓ\n");
  obj(1, "<< /Type /Catalog /Pages 2 0 R >>");
  obj(2, `<< /Type /Pages /Count ${pages.length} /Kids [${pageIds.map((id) => `${id} 0 R`).join(" ")}] >>`);
  obj(3, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>");
  obj(4, `<< /Title ${pdfText(title)} /Producer ${pdfText("GMR AI Intelligence")} >>`);
  pages.forEach((page, i) => {
    const [pageId, contentId, imageId] = [pageIds[i], pageIds[i] + 1, pageIds[i] + 2];
    const label = `${i + 1} / ${pages.length}`;
    const content = [
      `${background.map(n).join(" ")} rg 0 0 ${n(W)} ${n(H)} re f`,
      `q ${n(page.w)} 0 0 ${n(page.h)} ${n(M)} ${n(H - M - page.h)} cm /Im0 Do Q`,
      `BT /F1 8 Tf 0.45 0.5 0.58 rg ${n(W - M - label.length * 4.45)} ${n(M / 2 - 3)} Td (${label}) Tj ET`,
    ].join("\n");
    const contentBytes = enc.encode(content);
    obj(pageId, `<< /Type /Page /Parent 2 0 R /MediaBox [0 0 ${n(W)} ${n(H)}] /Resources << /XObject << /Im0 ${imageId} 0 R >> /Font << /F1 3 0 R >> >> /Contents ${contentId} 0 R >>`);
    obj(contentId, `<< /Length ${contentBytes.length} >>`, contentBytes);
    obj(imageId, `<< /Type /XObject /Subtype /Image /Width ${page.pxW} /Height ${page.pxH} /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length ${page.jpeg.length} >>`, page.jpeg);
  });
  const size = 5 + pages.length * 3;
  const xref = length;
  let table = `xref\n0 ${size}\n0000000000 65535 f \n`;
  for (let id = 1; id < size; id++) table += `${String(offsets[id]).padStart(10, "0")} 00000 n \n`;
  push(table);
  push(`trailer\n<< /Size ${size} /Root 1 0 R /Info 4 0 R >>\nstartxref\n${xref}\n%%EOF\n`);
  return new Blob(chunks as BlobPart[], { type: "application/pdf" });
}

/** An A4 PDF of the copy, downloaded directly (see above). Returns the file; the caller saves it. */
export async function renderSnapshotPdf(title: string, parts: SnapshotPart[]): Promise<{ blob: Blob; filename: string }> {
  const blob = await drawSnapshot(title, parts, "PDF", async (clone, options, background) => {
    const canvas = await toCanvas(clone, options);
    if (!canvas.width || !canvas.height) return null;
    const cssW = clone.getBoundingClientRect().width || canvas.width;
    const scale = canvas.width / cssW;  // canvas px per CSS px
    const contentW = A4.w - 2 * A4.margin;
    const pagePt = A4.h - 2 * A4.margin;
    const ptPerPx = contentW / canvas.width;
    const ends = pageBreaks(clone, canvas.height / scale, pagePt / ptPerPx / scale);
    const pages: PdfPage[] = [];
    let start = 0;
    for (const end of ends) {
      const sy = Math.round(start * scale);
      const sh = Math.min(canvas.height, Math.round(end * scale)) - sy;
      start = end;
      if (sh <= 0) continue;
      const slice = document.createElement("canvas");
      slice.width = canvas.width;
      slice.height = sh;
      const ctx = slice.getContext("2d");
      if (!ctx) return null;
      ctx.fillStyle = background;
      ctx.fillRect(0, 0, slice.width, sh);
      ctx.drawImage(canvas, 0, sy, canvas.width, sh, 0, 0, canvas.width, sh);
      const jpeg = await canvasJpeg(slice);
      if (!jpeg) return null;
      pages.push({ jpeg, pxW: slice.width, pxH: sh, w: contentW, h: sh * ptPerPx });
    }
    return pages.length ? buildPdf(pages, rgbOf(background), title) : null;
  });
  return { blob, filename: snapshotFilename(title, "pdf") };
}
