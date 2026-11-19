/**
 * The hover card of the WFM dashboard's charts (HES-FRONTEND-GMR SmartChart.jsx CustomTooltip, origin/uat),
 * for the chatbot's charts:
 *   * the category in bold, and under a week label the dates it covers (getWeekRangeFromLabel)
 *   * one row per series -- colour dot, name, full Indian-format value -- DT, Consumer, Feeder first
 *   * a Total row when the series are parts of one whole (Consumer / DT / Feeder, age bands, Burnt / Faulty /
 *     Others ...), never under a total drawn beside its own parts (Supplied / Installed)
 *   * the card's extra figures (the rates: Productivity, Utilization Rate, % SAT vs MI, Defect Rate ...)
 *   * a donut slice: its value and its share of the whole
 * Every figure is the answer's own number; the Total is the sum of the rows shown above it.
 * The same card is used live (ApexCharts tooltip.custom) and in the downloaded HTML copy (exportTip.ts).
 */
import type { AIChartSpec } from "../../api/client";
import type { ChartLayout } from "../chartOptions";
import { fullNumber } from "./dataviz";
import { monthLabels } from "../../utils/formatters";

export interface HoverRow { name: string; value: string; color?: string; strong?: boolean; divider?: boolean; heading?: boolean }
export interface HoverCard { title: string; sub?: string; rows: HoverRow[] }

const ORDER = ["dt", "dtr", "consumer", "feeder"];
// Names that are parts of one whole on the dashboard cards (its Total row keys + the age / stock bands).
const PART_RE = /^(?:dtr?|consumers?|feeders?|burnt|faulty|meter burnt|meter faulty|others?|ok|failed|upload (?:ok|failed|other))$|days?$|^age\b|\+$/i;
const WHOLE_RE = /\b(?:total|suppl(?:y|ied|ies)|inventory|received|eligible)\b|^escalated tickets$|^(?:installed(?: meters)?|meters installed)$/i;
const PERCENT_RE = /%|\bpct\b|percent|\brate\b|utili[sz]ation|efficiency|share/i;
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

const num = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? v : Number(v) || 0);
const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
const dd = (d: Date) => `${String(d.getUTCDate()).padStart(2, "0")} ${MONTHS[d.getUTCMonth()]}`;

/** '2026-W45' (ISO week) -> '02 Nov - 08 Nov 2026'; 'week2-oct' (the dashboard's own) -> 'Oct 08 - Oct 14'. */
export function weekRange(label: string): string | undefined {
  const iso = /^(\d{4})-w(\d{1,2})$/i.exec(label.trim());
  if (iso) {
    const year = Number(iso[1]);
    const week = Number(iso[2]);
    const jan4 = new Date(Date.UTC(year, 0, 4));
    const monday = new Date(jan4.getTime() - (((jan4.getUTCDay() + 6) % 7) * 86400000) + (week - 1) * 7 * 86400000);
    const sunday = new Date(monday.getTime() + 6 * 86400000);
    return `${dd(monday)} - ${dd(sunday)} ${sunday.getUTCFullYear()}`;
  }
  const dash = /^week(\d+)-([a-z]{3})$/i.exec(label.trim());
  if (dash) {
    const m = MONTHS.findIndex((x) => x.toLowerCase() === dash[2].toLowerCase());
    if (m < 0) return undefined;
    const start = (Number(dash[1]) - 1) * 7 + 1;
    const last = new Date(Date.UTC(new Date().getUTCFullYear(), m + 1, 0)).getUTCDate();
    const end = Math.min(start + 6, last);
    return `${MONTHS[m]} ${String(start).padStart(2, "0")} - ${MONTHS[m]} ${String(end).padStart(2, "0")}`;
  }
  return undefined;
}

function named(chart: AIChartSpec): { name: string; data: number[]; type?: string }[] {
  if (chart.series?.length) return chart.series.map((s) => ({ name: s.name, data: (s.data ?? []).map(num), type: s.type }));
  return [{ name: chart.display_title || chart.title, data: (chart.y ?? []).map(num) }];
}

/** Are the series parts of one whole, so a Total row means something? */
export function seriesAreParts(chart: AIChartSpec, layout: ChartLayout): boolean {
  const s = named(chart);
  if (s.length < 2 || layout.combo || s.some((x) => WHOLE_RE.test(x.name))) return false;
  return layout.stacked || Boolean(chart.stack_total) || s.every((x) => PART_RE.test(x.name.trim()));
}

/** The dashboard hover card for category j. */
export function hoverCard(chart: AIChartSpec, layout: ChartLayout, j: number, colours: string[]): HoverCard {
  const cats = monthLabels((chart.x ?? []).map(String));
  const title = cats[j] ?? "";
  const sub = weekRange(title);
  const unit = chart.unit || undefined;
  if (layout.kind === "pie" || layout.kind === "donut") {
    const values = (chart.y ?? []).map(num);
    const total = values.reduce((t, v) => t + v, 0);
    const v = values[j] ?? 0;
    return {
      title, sub,
      rows: [
        { name: "Value", value: fullNumber(v, unit), color: colours[j] },
        ...(total > 0 ? [{ name: "Share", value: `${((v * 100) / total).toFixed(1)}%` }] : []),
        ...(total > 0 ? [{ name: "Total", value: fullNumber(total, unit), strong: true, divider: true }] : []),
      ],
    };
  }
  const series = named(chart).map((s, i) => ({ ...s, color: colours[i] }));
  const ordered = [...series].sort((a, b) => {
    const ia = ORDER.indexOf(a.name.toLowerCase()), ib = ORDER.indexOf(b.name.toLowerCase());
    return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib);
  });
  const rows: HoverRow[] = ordered.map((s) => ({
    name: s.name,
    value: fullNumber(s.data[j] ?? 0, s.type === "line" ? chart.unit2 || undefined : unit),
    color: s.color,
  }));
  if (seriesAreParts(chart, layout)) {
    const total = series.reduce((t, s) => t + (s.data[j] ?? 0), 0);
    rows.push({ name: "Total", value: fullNumber(total, unit), strong: true, divider: true });
  }
  for (const r of chart.rates ?? []) {
    const v = r.data?.[j];
    if (v === undefined || v === null) continue;
    rows.push({ name: r.name, value: fullNumber(num(v), PERCENT_RE.test(r.name) ? "%" : undefined) });
  }
  // The place split behind this point (the dashboard's detail tooltip rows).
  const details = chart.details?.[j] ?? [];
  if (details.length) {
    rows.push({ name: `By ${chart.details_label || "place"}`, value: "", heading: true, divider: true });
    for (const [name, v] of details) rows.push({ name, value: fullNumber(num(v), unit) });
  }
  return { title, sub, rows };
}

/** The card as HTML (the live tooltip and the saved page draw the same markup). */
export function hoverHtml(card: HoverCard): string {
  const rows = card.rows.map((r) => r.heading
    ? `<div class="hc-row hc-heading${r.divider ? " hc-divider" : ""}">${esc(r.name)}</div>`
    : `<div class="hc-row${r.divider ? " hc-divider" : ""}"><span class="hc-name">`
    + (r.color ? `<span class="hc-dot" style="background:${esc(r.color)}"></span>` : "")
    + `${esc(r.name)}:</span><span class="hc-value${r.strong ? " hc-strong" : ""}">${esc(r.value)}</span></div>`).join("");
  return `<div class="hc-card"><div class="hc-head"><div class="hc-title">${esc(card.title)}</div>`
    + (card.sub ? `<div class="hc-sub">${esc(card.sub)}</div>` : "") + `</div>${rows}</div>`;
}
