import React, { useEffect, useMemo, useRef, useState } from "react";
import ReactApexChart from "react-apexcharts";
import type { AIChartSpec } from "../api/client";
import { buildChartOptions, buildChartSeries, chartCategories, resolveKind } from "./chartOptions";
import { chartTipData } from "./charts/exportTip";
import { hoverCard, hoverHtml } from "./charts/hoverCard";
import { uiText } from "../i18n/languages";

const MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
const DAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];

/** The chart library's own menu (Download PNG / CSV) in the screen language. Month and day
 * names stay English: the axis labels come from the data, which is English. The locale name
 * carries the language so a language change re-applies it. */
function chartLocale(lang?: string | null) {
  const name = `ui-${lang || "en"}`;
  return {
    defaultLocale: name,
    locales: [{
      name,
      options: {
        months: MONTHS, shortMonths: MONTHS.map((m) => m.slice(0, 3)), days: DAYS, shortDays: DAYS.map((d) => d.slice(0, 3)),
        toolbar: {
          menu: uiText(lang, "chartMenu"), exportToPNG: uiText(lang, "chartDownloadPng"),
          exportToCSV: uiText(lang, "chartDownloadCsv"), exportToSVG: uiText(lang, "chartDownloadSvg"),
          selection: "Selection", selectionZoom: "Selection Zoom", zoomIn: "Zoom In", zoomOut: "Zoom Out", pan: "Panning", reset: "Reset Zoom",
        },
      },
    }],
  };
}

const HOVER_KINDS = new Set(["bar", "line", "area", "pie", "donut"]);
const coloursOf = (options: { colors?: unknown[] }) => (options.colors ?? []).map((c) => (typeof c === "string" ? c : ""));

/** The dashboard's hover card (charts/hoverCard.ts) in place of the library's default list. Gauges have no
 * tooltip (as on the dashboard) and heatmap cells keep the library's. */
function withHoverCard(options: ReturnType<typeof buildChartOptions>, chart: AIChartSpec, layout: ReturnType<typeof resolveKind>) {
  if (!HOVER_KINDS.has(layout.kind) || options.tooltip?.enabled === false) return {};
  const colours = coloursOf(options);
  const circular = layout.kind === "pie" || layout.kind === "donut";
  return {
    tooltip: {
      ...options.tooltip,
      custom: ({ seriesIndex, dataPointIndex }: { seriesIndex: number; dataPointIndex: number }) =>
        hoverHtml(hoverCard(chart, layout, circular ? seriesIndex : dataPointIndex, colours)),
    },
  };
}

interface ChartVisualizationProps {
  chart: AIChartSpec;
  height?: number;
  /** Called with the clicked bar/slice's category so the caller can turn a
   * click into a scoped follow-up question. Omit to render read-only. */
  onPointClick?: (label: string) => void;
  /** Entry animations (default on); the dev preview turns them off for screenshots. */
  animate?: boolean;
  /** Screen language for the chart's own menu and empty-state text. */
  lang?: string | null;
}

// ApexCharts styled after the WFM/HES dashboard's chart cards. All option/series
// construction lives in ./chartOptions so it stays a pure function of the backend
// payload and can be exercised against the real library (scripts/verify-charts.mjs).
export const ChartVisualization: React.FC<ChartVisualizationProps> = ({ chart, height = 320, onPointClick, animate = true, lang }) => {
  const layout = useMemo(() => resolveKind(chart.type), [chart.type]);
  const series = useMemo(() => buildChartSeries(chart, layout), [chart, layout]);
  const options = useMemo(() => {
    const raw = buildChartOptions(chart, layout, onPointClick);
    const built = { ...raw, chart: { ...raw.chart, ...chartLocale(lang) }, ...withHoverCard(raw, chart, layout) };
    // Users who ask the OS for reduced motion get static charts (and so do screenshots).
    const reduced = typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    return animate && !reduced ? built : { ...built, chart: { ...built.chart, animations: { enabled: false } } };
  }, [chart, layout, onPointClick, animate, lang]);
  // Hover figures for the downloaded HTML copy, which has no chart library (utils/snapshot.ts).
  const tip = useMemo(() => {
    const cards = HOVER_KINDS.has(layout.kind)
      ? chartCategories(chart).map((_c, j) => hoverHtml(hoverCard(chart, layout, j, coloursOf(options)))) : undefined;
    const data = chartTipData(chartCategories(chart), layout, series as any, options, cards);
    return data ? JSON.stringify(data) : undefined;
  }, [chart, layout, series, options]);

  if (!chartCategories(chart).length || !chart.y?.length) {
    return <div className="table-empty">{uiText(lang, "noChartData")}</div>;
  }
  if (layout.kind === "radialBar") {
    return <Gauges chart={chart} height={height} animate={animate} />;
  }
  // Ranked / horizontal bars grow with the number of rows, like the dashboard's RankedBar.
  const rows = chartCategories(chart).length;
  const h = layout.horizontal && !layout.funnel ? Math.max(height, rows * 30 + 60) : height;

  return (
    <div className={onPointClick ? "chart-clickable" : undefined} data-chart-tip={tip}>
      {/* Keyed by language and chart type: the library reads its locale (menu labels) only when a chart is created, and a Graphs Modes switch (bar -> area) needs a fresh chart. */}
      <ReactApexChart key={`${lang || "en"}-${chart.type}`} options={options} series={series as any} type={layout.kind} height={h} width="100%" />
    </div>
  );
};

/** Gauges: one half ring per percentage (up to 4), side by side in one row. A half
 * ring is twice as wide as it is tall, so each one's height is the largest that
 * fits both the card's height and its share of the card's measured width -- no
 * empty lower half, no wrapping out of the card. */
const Gauges: React.FC<{ chart: AIChartSpec; height: number; animate: boolean }> = ({ chart, height, animate }) => {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width));
    observer.observe(el);
    return () => observer.disconnect();
  }, []);
  const cats = chartCategories(chart);
  const values = chart.y.slice(0, 4);
  const gap = 32;
  const byWidth = width ? (width - 16 - gap * (values.length - 1)) / values.length / 2 : height;
  const size = Math.max(80, Math.floor(Math.min(height * 0.72, byWidth)));
  return (
    <div ref={ref} className="chart-gauges" style={{ minHeight: height, columnGap: gap }}>
      {width
        ? values.map((value, i) => (
            <Gauge key={`${cats[i]}-${i}`} chart={{ ...chart, x: [cats[i] ?? ""], y: [value], series: undefined }}
              height={size} animate={animate} />
          ))
        : null}
    </div>
  );
};

const Gauge: React.FC<{ chart: AIChartSpec; height: number; animate: boolean }> = ({ chart, height, animate }) => {
  const layout = useMemo(() => resolveKind(chart.type), [chart.type]);
  const series = useMemo(() => buildChartSeries(chart, layout), [chart, layout]);
  const options = useMemo(() => {
    const built = buildChartOptions(chart, layout);
    // Text scales with the ring so a small gauge's value stays inside its hollow.
    const labels = (built.plotOptions?.radialBar?.dataLabels ?? {}) as any;
    const valueSize = Math.max(16, Math.min(30, Math.round(height * 0.15)));
    built.plotOptions = {
      ...built.plotOptions,
      radialBar: {
        ...built.plotOptions?.radialBar,
        hollow: { size: height < 170 ? "70%" : "62%" },
        dataLabels: {
          ...labels,
          name: { ...labels.name, fontSize: `${Math.max(10, Math.min(12, Math.round(height * 0.06)))}px`, offsetY: -Math.round(height * 0.04) },
          value: { ...labels.value, fontSize: `${valueSize}px`, offsetY: -Math.round(height * 0.04) - valueSize - 12 },
        },
      },
    };
    const reduced = typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    return animate && !reduced ? built : { ...built, chart: { ...built.chart, animations: { enabled: false } } };
  }, [chart, layout, animate]);
  return (
    // The ring is drawn in a square twice the gauge's height; only its top half
    // (the half ring, centre on the bottom edge) is shown.
    <div className="chart-gauge" style={{ width: height * 2, height: height + 6 }}>
      <ReactApexChart options={options} series={series as any} type="radialBar" height={height * 2} width={height * 2} />
    </div>
  );
};
