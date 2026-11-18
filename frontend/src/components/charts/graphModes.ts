/**
 * "Graphs Modes": the WFM dashboard's chart switcher (HES-FRONTEND-GMR KPIDetailModal.jsx CHART_TYPES +
 * dashboardData.js allowedTrendTypes / allowedDistTypes, origin/uat 23 Oct 2026) for the chatbot's charts.
 *
 * A mode only redraws the figures the answer already has -- no request, no new number. Each card offers
 * the modes the dashboard offers for it (trend list when the axis is dates, comparison list otherwise);
 * a chart from no dashboard card offers the modes its data can show. Modes the data cannot show are left
 * out: a donut needs non-negative parts, a gauge needs percentages, a total beside its own parts
 * (Supplied / Installed, MI / SAT) is never summed into one slice.
 */
import type { AIChartSpec } from "../../api/client";
import type { UIKey } from "../../i18n/uiStrings";

export type GraphMode = "area" | "line" | "bar" | "hbar" | "donut" | "gauge" | "funnel" | "pareto";

export const MODE_ORDER: GraphMode[] = ["area", "line", "bar", "hbar", "donut", "gauge", "funnel", "pareto"];

type Allowed = { trend: GraphMode[]; dist: GraphMode[] };

// dashboardData.js, per card. The dashboard lists "line" for several trends; the chatbot shows it.
const DASHBOARD_MODES: Record<string, Allowed> = {
  mi_progress: { trend: ["area", "line", "bar"], dist: ["bar", "hbar", "donut", "funnel"] },
  mi_productivity: { trend: ["bar", "line", "area"], dist: ["bar"] },
  monthly_productivity: { trend: ["bar", "line", "area"], dist: ["bar"] },
  meter_replacement: { trend: ["bar", "hbar"], dist: ["bar", "hbar", "donut"] },
  supply_utilization: { trend: ["bar", "area", "line"], dist: ["bar", "hbar"] },
  mi_pace_stock: { trend: ["bar", "area", "line"], dist: ["bar", "hbar"] },
  stock_ageing: { trend: ["area", "bar", "line"], dist: ["bar", "hbar"] },
  om_escalated: { trend: ["donut", "hbar"], dist: ["bar", "hbar"] },
  om_productivity: { trend: ["bar", "line", "area"], dist: ["bar"] },
  hhu_reading: { trend: ["bar", "line", "area"], dist: ["bar"] },
  om_active_ageing: { trend: ["donut", "hbar"], dist: ["bar", "hbar"] },
  om_closed: { trend: ["line", "area", "bar"], dist: ["donut", "hbar", "bar"] },
  mi_vs_sat: { trend: ["bar", "hbar"], dist: ["bar", "hbar"] },
  non_sat_ageing: { trend: ["area", "bar", "line"], dist: ["bar", "hbar"] },
  sat_invoice_funnel: { trend: ["bar", "hbar"], dist: ["bar", "hbar"] },
  revenue_ageing: { trend: ["bar", "hbar"], dist: ["bar", "hbar"] },
  meter_journey: { trend: ["bar", "hbar"], dist: ["bar", "hbar"] },
  meter_stage: { trend: ["bar", "hbar"], dist: ["bar", "hbar"] },
};

export const MODE_LABEL_KEY: Record<GraphMode, UIKey> = {
  area: "modeArea", line: "modeLine", bar: "modeBar", hbar: "modeHbar", donut: "modeDonut",
  gauge: "modeGauge", funnel: "modeFunnel", pareto: "modePareto",
};
/** Several series: the dashboard calls these "Multi Area" / "Multi Line". */
export const MULTI_LABEL_KEY: Partial<Record<GraphMode, UIKey>> = { area: "modeMultiArea", line: "modeMultiLine" };

// dashboard_charts._DATE_RE (backend), the dashboard's chartUtils isDate.
const DATE_RE = /^(?:\d{4}-\d{2}-\d{2}|\d{2}-\d{2}-\d{4}|\d{2}-\d{2}-\d{2}|\d{2}\/\d{2}\/\d{4}|\d{2}\/\d{2}\/\d{2}|\d{4}-\d{2}|\d{4}-w\d{1,2}|[a-z]{3}[-\s]\d{2,4}|\d{1,2}[-\s][a-z]{3}[-\s]\d{2,4}|\d{1,2}\s[a-z]{3}\s*-\s*\d{1,2}\s[a-z]{3})$|^week|^w\d+-/i;
const PERCENT_RE = /%|\bpct\b|percent|\brate\b|utili[sz]ation|efficiency|share/i;
// A total that already holds the other series (backend dashboard_charts._WHOLE_RE): never one slice.
const WHOLE_RE = /\b(?:total|suppl(?:y|ied|ies)|inventory|received|eligible)\b|^escalated tickets$|^(?:installed(?: meters)?|meters installed)$/i;

const num = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? v : Number(v) || 0);

function seriesOf(chart: AIChartSpec): { name: string; data: number[] }[] {
  if (chart.series?.length) return chart.series.map((s) => ({ name: s.name, data: (s.data ?? []).map(num) }));
  return [{ name: chart.title, data: (chart.y ?? []).map(num) }];
}

const isDates = (cats: string[]) => cats.length > 0 && cats.every((c) => DATE_RE.test(String(c).trim()));
const isStacked = (t: AIChartSpec["type"]) => t === "stacked_bar" || t === "stacked_horizontal_bar" || t === "percent_stacked_bar";

/** The mode the chart is drawn in as sent (highlighted in the menu until another is picked). */
export function currentMode(chart: AIChartSpec): GraphMode | null {
  switch (chart.type) {
    case "area": return "area";
    case "line": return "line";
    case "bar": case "grouped_bar": case "stacked_bar": case "percent_stacked_bar": return "bar";
    case "horizontal_bar": case "stacked_horizontal_bar": return "hbar";
    case "pie": case "donut": return "donut";
    case "radial": return "gauge";
    case "funnel": return "funnel";
    default: return null;
  }
}

/** Modes this chart can be redrawn in; [] = no switcher (combo / heatmap / one bar). */
export function availableModes(chart: AIChartSpec): GraphMode[] {
  if (chart.type === "combo" || chart.type === "heatmap" || chart.type === "percent_stacked_bar") return [];
  const cats = (chart.x ?? []).map(String);
  const series = seriesOf(chart);
  const n = cats.length;
  const k = series.length;
  if (n < 2 && chart.type !== "radial") return [];
  const values = series.flatMap((s) => s.data);
  const nonNegative = values.every((v) => v >= 0);
  const dated = isDates(cats);
  // Parts of one whole: one series, or series the card already stacks (Consumer / DT / Feeder, age bands).
  const parts = k === 1 || (isStacked(chart.type) && !series.some((s) => WHOLE_RE.test(s.name)));

  let wanted: GraphMode[];
  const card = chart.dashboard ? DASHBOARD_MODES[chart.dashboard] : undefined;
  if (card) {
    wanted = dated ? card.trend : [...new Set([...card.dist, ...card.trend.filter((m) => m !== "line" && m !== "area")])];
  } else {
    wanted = ["bar", "hbar", "line", "area", "donut", "gauge", "funnel", "pareto"];
  }

  const ok = (m: GraphMode): boolean => {
    switch (m) {
      case "line": case "area": return n >= 2 && (dated || !card);
      case "bar": case "hbar": return true;
      case "donut": return parts && nonNegative && n >= 2 && n <= 12 && !dated;
      case "gauge": return k === 1 && n <= 4 && nonNegative && values.every((v) => v <= 100)
        && (chart.unit === "%" || PERCENT_RE.test(`${series[0].name} ${chart.title}`));
      case "funnel": return k === 1 && n >= 3 && nonNegative && !dated
        && series[0].data.every((v, i, a) => i === 0 || v <= a[i - 1]);
      case "pareto": return !card && k === 1 && n >= 3 && nonNegative && !dated && values.some((v) => v > 0);
    }
  };
  const modes = MODE_ORDER.filter((m) => wanted.includes(m) && ok(m));
  const now = currentMode(chart);
  if (now && !modes.includes(now)) modes.unshift(now);
  // Only worth a menu when there is somewhere else to go.
  return modes.length > 1 ? modes : [];
}

/** The same figures drawn in another mode. Values are never changed -- a donut of stacked parts shows each
 * category's stack total; a pareto sorts the bars and adds the running share. */
export function applyMode(chart: AIChartSpec, mode: GraphMode | null): AIChartSpec {
  if (!mode || mode === currentMode(chart)) return chart;
  const series = seriesOf(chart);
  const multi = (chart.series?.length ?? 0) > 1;
  const stacked = isStacked(chart.type) || (chart.labels === "total" && multi);
  const partsFlag = stacked ? { stack_total: true } : {};
  // Keys are dropped, never set to undefined (an undefined key replaces a chart-library default).
  // Per-category colours (Technical boards) belong to bars and slices, not to a line.
  const keepPoints = mode === "bar" || mode === "hbar" || mode === "donut";
  const without = (...keys: (keyof AIChartSpec)[]): AIChartSpec => {
    const copy: AIChartSpec = { ...chart };
    for (const k of [...keys, ...(keepPoints ? [] : ["point_keys" as const])]) delete copy[k];
    return copy;
  };
  switch (mode) {
    case "bar":
      return { ...without(), type: multi ? (stacked ? "stacked_bar" : "grouped_bar") : "bar" };
    case "hbar":
      return { ...without(), type: multi ? (stacked ? "stacked_horizontal_bar" : "horizontal_bar") : "horizontal_bar" };
    case "line":
    case "area":
      return { ...without("labels"), ...partsFlag, type: mode };
    case "donut": {
      const totals = (chart.x ?? []).map((_c, i) => series.reduce((t, s) => t + (s.data[i] ?? 0), 0));
      return { ...without("series", "series_keys", "rates"), type: "donut", y: totals };
    }
    case "gauge":
      return { ...without("rates"), type: "radial" };
    case "funnel":
      return { ...without("rates"), type: "funnel" };
    case "pareto": {
      const cats = (chart.x ?? []).map(String);
      const order = cats.map((_c, i) => i).sort((a, b) => series[0].data[b] - series[0].data[a]);
      const sorted = order.map((i) => series[0].data[i]);
      const total = sorted.reduce((t, v) => t + v, 0);
      let run = 0;
      const cumulative = sorted.map((v) => { run += v; return total > 0 ? Math.round((run * 1000) / total) / 10 : 0; });
      return {
        ...without("rates", "labels", "series_keys"), type: "combo", x: order.map((i) => cats[i]), y: sorted,
        ...(chart.details ? { details: order.map((i) => chart.details![i] ?? []) } : {}),
        series: [{ name: series[0].name, data: sorted, type: "bar" }, { name: "Cumulative %", data: cumulative, type: "line" }],
        unit2: "%",
      };
    }
  }
}

const ISO_DAY_OR_MONTH = /^\d{4}-\d{2}(?:-\d{2})?$/;

/** Time left to right, as on the portal: a chart whose dates arrive newest first is drawn oldest first
 * (answers saved before the backend did the same, 18 Nov). Every per-category array is flipped with it. */
export function chronological(chart: AIChartSpec): AIChartSpec {
  const x = (chart.x ?? []).map(String);
  if (x.length < 2 || !x.every((c) => ISO_DAY_OR_MONTH.test(c)) || !x.every((c, i) => i === 0 || x[i - 1] > c)) return chart;
  const flip = <T,>(a?: T[]) => (a ? [...a].reverse() : a);
  const out: AIChartSpec = { ...chart, x: flip(chart.x)!, y: flip(chart.y)! };
  if (chart.series) out.series = chart.series.map((s) => ({ ...s, data: flip(s.data)! }));
  if (chart.rates) out.rates = chart.rates.map((r) => ({ ...r, data: flip(r.data)! }));
  if (chart.point_keys) out.point_keys = flip(chart.point_keys);
  if (chart.details) out.details = flip(chart.details);
  return out;
}
