/**
 * ApexCharts option/series construction for an `AIChartSpec`, styled after the
 * WFM/HES dashboard (Documents/WFM/HES-FRONTEND-GMR -- KpiCharts.jsx, SmartChart.jsx,
 * theme/dataviz.tokens.js): one hue + grey track for ranked bars, value labels on
 * bars, totals on stacked columns, a band ramp for ordered bands, the categorical
 * palette for genuinely different series, donut with the total in the hole and
 * count + share in the legend, half-ring gauge, dashed light grid, Indian number
 * formats (42,14,678 in labels / 42.15L on axes).
 *
 * Kept separate from the React component so it is a pure function of the
 * backend's chart payload, and can be executed directly against the real
 * ApexCharts library in a test without mounting a component tree.
 *
 * Two rules govern everything here:
 *
 * 1. No figure is derived, except where the dashboard itself shows one (a donut's
 *    total and share %). Values, labels and units come from the backend as sent;
 *    a unit is shown only when the backend detected one.
 * 2. No option key is ever set to `undefined`. ApexCharts merges these options
 *    over its defaults and then reads several paths unconditionally --
 *    `config.markers.size`, `config.xaxis.title.text`,
 *    `config.yaxis[0].title.text`, `config.chart.events`. A key that is present
 *    with the value `undefined` REPLACES the library default instead of leaving
 *    it in place, which is a hard TypeError at render time. Optional options are
 *    spread in conditionally so the key is absent entirely when it does not apply.
 *
 * Chart types (chosen by the backend LLM from few-shot examples in
 * backend/gmr_agents/schema/data/charts.yaml):
 *   bar, grouped_bar, stacked_bar, percent_stacked_bar, horizontal_bar,
 *   stacked_horizontal_bar, line, area, combo (bars + line on a 2nd axis),
 *   pie, donut, funnel, radial (gauge of %), heatmap.
 */
import type { ApexOptions } from "apexcharts";
import type { AIChartSpec } from "../api/client";
import { DV, MONO, axisNumber, fullNumber, niceAxis, seriesColour, truncate } from "./charts/dataviz";
import { canonicalKey, dashboardColour, getContrastText } from "./charts/dashboardPalette";

// Kept for callers/tests that refer to the categorical palette by this name.
export const SERIES_COLORS = DV.palette;

// Category names in real data run long ("EDD LAXMINAGAR MATHURA"). Axis ticks are
// shortened to stay readable; the tooltip always shows the untruncated name.
const MAX_CATEGORY_CHARS = 18;
const MAX_SIDE_CATEGORY_CHARS = 26;

export type ApexKind = "bar" | "pie" | "donut" | "line" | "area" | "radialBar" | "heatmap";

export interface ChartLayout {
  kind: ApexKind;
  horizontal: boolean;
  stacked: boolean;
  percent: boolean; // 100% stacked
  funnel: boolean;
  combo: boolean;   // bar series + line series, line on a second y axis
}

const L = (kind: ApexKind, flags: Partial<ChartLayout> = {}): ChartLayout => ({
  kind, horizontal: false, stacked: false, percent: false, funnel: false, combo: false, ...flags,
});

const LAYOUT: Record<string, ChartLayout> = {
  bar: L("bar"),
  grouped_bar: L("bar"),
  stacked_bar: L("bar", { stacked: true }),
  percent_stacked_bar: L("bar", { stacked: true, percent: true }),
  horizontal_bar: L("bar", { horizontal: true }),
  stacked_horizontal_bar: L("bar", { horizontal: true, stacked: true }),
  line: L("line"),
  area: L("area"),
  combo: L("line", { combo: true }),
  pie: L("pie"),
  donut: L("donut"),
  funnel: L("bar", { horizontal: true, funnel: true }),
  radial: L("radialBar"),
  heatmap: L("heatmap"),
};

function safeNumber(value: unknown): number {
  const num = typeof value === "number" ? value : Number(value);
  return Number.isFinite(num) ? num : 0;
}

/** Backend chart type -> ApexCharts kind plus layout flags (unknown -> bar). */
export function resolveKind(type: AIChartSpec["type"]): ChartLayout {
  return LAYOUT[type] ?? LAYOUT.bar;
}

export function chartCategories(chart: AIChartSpec): string[] {
  return (chart.x ?? []).map((x) => String(x ?? ""));
}

type Named = { name: string; data: number[]; type?: "bar" | "line" };

/** Named series as sent; a single-series chart is built from `y`. */
function namedSeries(chart: AIChartSpec): Named[] {
  if (chart.series?.length) {
    return chart.series.map((s) => ({ name: s.name, data: (s.data ?? []).map(safeNumber), type: s.type }));
  }
  return [{ name: chart.title, data: (chart.y ?? []).map(safeNumber) }];
}

/** Series colours by the dashboard's rules: one hue for a single magnitude series,
 * band ramp for ordered bands (age buckets), categorical palette otherwise. */
function colours(chart: AIChartSpec, layout: ChartLayout): string[] {
  const series = namedSeries(chart);
  const kpi = chart.dashboard;
  if (kpi && layout.kind !== "radialBar" && layout.kind !== "heatmap") {
    // The dashboard card's fixed colour per series name (per slice / step for donuts and funnels).
    if (layout.kind === "pie" || layout.kind === "donut" || layout.funnel) {
      return chartCategories(chart).map((c, i) => dashboardColour(kpi, c, i, true));
    }
    // One unnamed series is the card's `value` (blue); its name is only the chart's title --
    // seen live, 'Revenue-Not-Realized Meters by Current Stage' turned the bars revenue red.
    const first = chart.series?.length ? series[0].name : "value";
    const names = [first, ...series.slice(1).map((s) => s.name),
      ...(layout.kind === "area" ? (chart.rates ?? []).map((r) => r.name) : [])];
    return names.map((n, i) => dashboardColour(kpi, n, i));
  }
  if (layout.kind === "pie" || layout.kind === "donut" || layout.funnel) return DV.palette;
  if (layout.kind === "radialBar") return [DV.good, ...DV.palette.slice(1)];
  if (layout.kind === "heatmap") return [DV.series];
  if (series.length === 1) return [DV.series];
  const names = series.map((s) => s.name);
  if (layout.combo) {
    return series.map((s, i) => (s.type === "line" ? (i === series.length - 1 ? DV.accent : seriesColour(i, names)) : seriesColour(i, names)));
  }
  return series.map((_s, i) => seriesColour(i, names));
}

/**
 * Series payload. Circular/radial charts take a flat number[]; axis charts
 * take named series; heatmap takes one series per row of {x, y} cells.
 * Values pass through untouched -- they are already the backend's figures.
 */
export function buildChartSeries(
  chart: AIChartSpec,
  layout: ChartLayout
): number[] | { name: string; type?: string; data: number[] }[] | { name: string; data: { x: string; y: number }[] }[] {
  if (layout.kind === "pie" || layout.kind === "donut" || layout.kind === "radialBar") {
    return (chart.y ?? []).map(safeNumber);
  }
  const categories = chartCategories(chart);
  if (layout.kind === "heatmap") {
    return namedSeries(chart).map((s) => ({ name: s.name, data: categories.map((x, i) => ({ x, y: s.data[i] ?? 0 })) }));
  }
  if (layout.funnel) {
    const s = namedSeries(chart)[0];
    return [{ name: s.name, data: s.data }];
  }
  if (layout.combo) {
    // Series the backend marked "line" ride the second axis; the rest are bars.
    return namedSeries(chart).map((s) => ({ name: s.name, type: s.type === "line" ? "line" : "column", data: s.data }));
  }
  if (chart.dashboard && layout.kind === "area" && chart.rates?.length) {
    // Dashboard area trend: a rate (Productivity) is a line on its own right-hand axis.
    return [
      ...namedSeries(chart).map((s) => ({ name: s.name, type: "area", data: s.data })),
      ...chart.rates.map((r) => ({ name: r.name, type: "line", data: (r.data ?? []).map(safeNumber) })),
    ];
  }
  return namedSeries(chart).map((s) => ({ name: s.name, data: s.data }));
}

const AXIS_LABEL = { colors: DV.axis, fontSize: "11px", fontWeight: 500 };
const MONO_LABEL = { colors: DV.axis, fontSize: "10.5px", fontFamily: MONO };

function axisTooltipX(categories: string[], horizontal: boolean) {
  // Full category name on hover, however short the axis tick was cut.
  return horizontal
    ? {}
    : { x: { formatter: (_value: string, opts: any) => categories[opts?.dataPointIndex ?? 0] ?? "" } };
}

export function buildChartOptions(
  chart: AIChartSpec,
  layout: ChartLayout,
  onPointClick?: (label: string) => void
): ApexOptions {
  const { kind, horizontal, stacked } = layout;
  const isCircular = kind === "pie" || kind === "donut";
  const categories = chartCategories(chart);
  const unit = chart.unit || undefined;
  const unit2 = chart.unit2 || undefined;
  const series = namedSeries(chart);
  const seriesCount = series.length;
  const palette = colours(chart, layout);

  const base: ApexOptions = {
    chart: {
      type: kind,
      stacked,
      ...(layout.percent ? { stackType: "100%" as const } : {}),
      toolbar: { show: true, tools: { download: true, selection: false, zoom: false, zoomin: false, zoomout: false, pan: false, reset: false } },
      fontFamily: "inherit",
      foreColor: DV.axis,
      animations: { enabled: true, speed: 500, animateGradually: { enabled: true, delay: 60 }, dynamicAnimation: { enabled: true, speed: 300 } },
      ...(onPointClick
        ? {
            events: {
              dataPointSelection: (_event: unknown, _ctx: unknown, config: any) => {
                const index = config?.dataPointIndex ?? -1;
                const label = categories[index];
                if (label) onPointClick(label);
              },
            },
          }
        : {}),
    },
    colors: palette,
    // Always a real object: size 0 means "no point markers", which is what bar
    // and pie charts want, while keeping `config.markers.size` readable.
    markers: { size: kind === "line" || kind === "area" ? (categories.length > 14 ? 0 : 4) : 0, strokeWidth: 0, hover: { size: 6 } },
    legend: {
      show: isCircular || seriesCount > 1,
      position: isCircular ? "right" : "top",
      horizontalAlign: "center",
      fontSize: "12px",
      labels: { colors: DV.text },
      markers: { size: 6, shape: "square" as const },
      itemMargin: { horizontal: 10, vertical: 3 },
    },
    states: { hover: { filter: { type: "darken" } } },
    noData: { text: "No data for this chart." },
  };

  const dashboard = chart.dashboard ? dashboardOptions(chart, layout, base, categories, series, palette, unit) : null;
  if (dashboard) return dashboard;

  if (isCircular) {
    // Dashboard donut: total in the hole, each slice named with its count AND share.
    const total = (chart.y ?? []).reduce((t, v) => t + safeNumber(v), 0);
    return {
      ...base,
      labels: categories,
      legend: {
        ...base.legend,
        formatter: (name: string, opts: any) => {
          const v = safeNumber(opts?.w?.globals?.series?.[opts?.seriesIndex]);
          const share = total > 0 ? ` · ${((v * 100) / total).toFixed(1)}%` : "";
          return `${truncate(name, 22)}  ${fullNumber(v, unit)}${share}`;
        },
      },
      dataLabels: { enabled: false },
      plotOptions: {
        pie: {
          expandOnClick: false,
          donut: {
            size: kind === "donut" ? "60%" : "0%",
            labels: {
              show: kind === "donut",
              name: { show: true, fontSize: "11px", color: "#94a3b8", offsetY: 18 },
              value: { show: true, fontSize: "18px", fontWeight: 700, fontFamily: MONO, color: "#0f172a", offsetY: -10,
                formatter: (val: string) => fullNumber(safeNumber(val), unit) },
              total: { show: true, showAlways: true, label: "total", fontSize: "11px", color: "#94a3b8",
                formatter: () => fullNumber(total, unit) },
            },
          },
        },
      },
      stroke: { width: 2, colors: [DV.surface] },
      tooltip: { y: { formatter: (val: number) => fullNumber(val, unit) } },
    };
  }

  if (kind === "radialBar") {
    // Dashboard gauge (WFM SmartChart GaugeVariant): one half ring per percentage,
    // filled share in green on a grey track, flat ends, the value large and bold in
    // the middle with the metric's name under it. ChartVisualization renders several
    // percentages as side-by-side gauges, so this always styles exactly one.
    return {
      ...base,
      chart: { ...base.chart, sparkline: { enabled: true }, toolbar: { show: false } },
      labels: categories.slice(0, 1),
      legend: { ...base.legend, show: false },
      plotOptions: {
        radialBar: {
          startAngle: -90,
          endAngle: 90,
          hollow: { size: "62%" },
          track: { background: DV.grid, strokeWidth: "100%", margin: 0 },
          dataLabels: {
            // Both labels sit above the centre line: the lower half of the ring is cropped.
            name: { show: true, fontSize: "12px", fontWeight: 500, color: DV.axis, offsetY: -8 },
            value: { show: true, fontSize: "30px", fontWeight: 800, fontFamily: "inherit", color: "#0f172a", offsetY: -52,
              formatter: (val: number) => fullNumber(val, unit ?? "%") },
          },
        },
      },
      grid: { padding: { top: 4, bottom: 4, left: 8, right: 8 } },
      stroke: { lineCap: "butt" },
      tooltip: { enabled: false },
    };
  }

  if (kind === "heatmap") {
    // ApexCharts shades cells from the lowest to the highest value, so text contrast follows the same scale.
    const heatValues = series.flatMap((s) => s.data);
    const heatMax = Math.max(...heatValues);
    const heatMin = Math.min(...heatValues);
    return {
      ...base,
      dataLabels: {
        enabled: categories.length <= 12,
        formatter: (v: number) => axisNumber(v, unit),
        // Text follows the cell's shade: white on dark cells, dark on light ones.
        style: {
          fontSize: "11px", fontFamily: MONO,
          colors: [({ seriesIndex, dataPointIndex, w }: any) => {
            const value = safeNumber(w?.globals?.series?.[seriesIndex]?.[dataPointIndex]);
            return heatMax > heatMin && (value - heatMin) / (heatMax - heatMin) > 0.45 ? "#fff" : DV.text;
          }] as any,
        },
      },
      plotOptions: { heatmap: { shadeIntensity: 0.65, radius: 3, enableShades: true, useFillColorAsStroke: false } },
      stroke: { width: 2, colors: [DV.surface] },
      tooltip: { y: { formatter: (val: number) => fullNumber(val, unit) } },
      xaxis: { categories, labels: { style: AXIS_LABEL, formatter: (value: string) => truncate(String(value ?? ""), MAX_CATEGORY_CHARS) } },
      yaxis: { labels: { style: AXIS_LABEL } },
    };
  }

  if (layout.funnel) {
    // Dashboard funnel: steps largest first, each with its name and full value.
    return {
      ...base,
      legend: { ...base.legend, show: false },
      dataLabels: {
        enabled: true,
        formatter: (val: number, opts: any) => `${categories[opts?.dataPointIndex ?? 0] ?? ""}: ${fullNumber(val, unit)}`,
        style: { fontSize: "12px", fontWeight: 700, fontFamily: MONO, colors: ["#fff"] },
        dropShadow: { enabled: false },
      },
      plotOptions: { bar: { horizontal: true, isFunnel: true, distributed: true, borderRadius: 4, barHeight: "82%" } },
      tooltip: { y: { formatter: (val: number) => fullNumber(val, unit) } },
      xaxis: { categories, labels: { show: false } },
      yaxis: { labels: { show: false } },
      grid: { show: false },
    };
  }

  const grid = {
    borderColor: DV.grid,
    strokeDashArray: 4,
    xaxis: { lines: { show: horizontal && !layout.funnel } },
    yaxis: { lines: { show: !horizontal } },
  };
  const rotate = !horizontal && (categories.length > 6 || categories.some((c) => c.length > 12));

  if (layout.combo) {
    const firstLine = series.findIndex((s) => s.type === "line");
    const firstBar = series.findIndex((s) => s.type !== "line");
    return {
      ...base,
      dataLabels: {
        // Dashboard dual-axis: only the rate line carries labels; bars read off the left axis.
        enabled: categories.length <= 12,
        enabledOnSeries: series.map((s, i) => (s.type === "line" ? i : -1)).filter((i) => i >= 0),
        // Short Indian figures on bars (16.36L) and the rate as sent on the line -- no unit text on marks.
        formatter: (val: number, opts: any) =>
          series[opts?.seriesIndex ?? 0]?.type === "line" ? fullNumber(val, unit2 === "%" ? "%" : undefined) : axisNumber(val),
        offsetY: -6,
        style: { fontSize: "10.5px", fontFamily: MONO, fontWeight: 700, colors: palette },
        background: { enabled: false },
      },
      plotOptions: { bar: { borderRadius: 4, borderRadiusApplication: "end", columnWidth: categories.length > 8 ? "60%" : "42%", dataLabels: { position: "top" } } },
      stroke: { width: series.map((s) => (s.type === "line" ? 3 : 0)), curve: "smooth" },
      grid,
      tooltip: {
        shared: true,
        intersect: false,
        y: { formatter: (val: number, opts: any) => fullNumber(val, series[opts?.seriesIndex ?? 0]?.type === "line" ? unit2 : unit) },
        ...axisTooltipX(categories, false),
      },
      xaxis: {
        categories,
        labels: { style: AXIS_LABEL, rotate: -32, rotateAlways: rotate, formatter: (value: string) => truncate(String(value ?? ""), MAX_CATEGORY_CHARS) },
        axisBorder: { color: DV.grid },
        axisTicks: { show: false },
      },
      // One y axis per series: bars share the left axis, lines share the right.
      yaxis: series.map((s, i) => {
        const isLine = s.type === "line";
        const owner = isLine ? firstLine : firstBar;
        const axisUnit = isLine ? unit2 : unit;
        return {
          seriesName: series[owner]?.name ?? s.name,
          opposite: isLine,
          show: i === owner,
          labels: { style: MONO_LABEL, formatter: (value: number) => axisNumber(value, axisUnit) },
          ...(axisUnit ? { title: { text: axisUnit, style: { color: DV.axis, fontWeight: 500 } } } : {}),
        };
      }),
    };
  }

  // Bars (vertical / horizontal / stacked / 100%) and lines / areas.
  const isBar = kind === "bar";
  const singleBar = isBar && seriesCount === 1;
  const labelsOn = isBar && !layout.percent && (horizontal ? categories.length <= 30 : categories.length * seriesCount <= 16);
  // Mark labels carry the number only (the unit is on the axis / tooltip), like the dashboard.
  // Several series side by side -> short Indian figures (16.36L) so neighbours don't collide.
  const markLabel = (val: number) => (seriesCount > 1 || categories.length > 10 ? axisNumber(val) : fullNumber(val));
  const columnTotals = categories.map((_c, i) => series.reduce((t, s) => t + (s.data[i] ?? 0), 0));
  const maxTotal = Math.max(0, ...columnTotals);
  return {
    ...base,
    dataLabels: {
      enabled: labelsOn || (layout.percent && categories.length <= 8),
      ...(layout.percent
        ? { formatter: (val: number) => (val >= 6 ? `${val.toFixed(1)}%` : "") }
        : stacked
          // Dashboard StackedColumn labels only the total; a segment gets its own label only when it is big enough to hold it.
          ? { formatter: (val: number) => (maxTotal > 0 && val / maxTotal >= 0.12 ? axisNumber(val) : "") }
          : { formatter: markLabel }),
      // Ranked bars: the label starts just past the bar end (dashboard RankedBar), never over it.
      ...(horizontal && !stacked ? { textAnchor: "start" as const, offsetX: 6 } : { offsetX: 0 }),
      offsetY: !horizontal && !stacked ? -18 : 0,
      style: {
        fontSize: "11px", fontFamily: MONO, fontWeight: 600,
        colors: stacked ? ["#fff"] : [DV.axis],
      },
      background: { enabled: false },
      dropShadow: { enabled: false },
    },
    plotOptions: {
      bar: {
        horizontal,
        borderRadius: stacked ? 0 : 4,
        borderRadiusApplication: "end",
        columnWidth: categories.length > 8 ? "62%" : seriesCount > 1 ? "55%" : "42%",
        barHeight: horizontal ? (categories.length > 12 ? "72%" : "60%") : "70%",
        dataLabels: {
          position: stacked ? "center" : "top",
          // Dashboard StackedColumn: the stack's total sits on top of each column.
          ...(stacked && !layout.percent
            ? { total: { enabled: true, formatter: (val?: string) => fullNumber(safeNumber(val), unit),
                style: { fontSize: "11px", fontWeight: 700, color: DV.axis } } }
            : {}),
        },
        // Dashboard RankedBar: a grey track behind every single-hue bar.
        ...(singleBar && horizontal ? { colors: { backgroundBarColors: [DV.surfaceAlt], backgroundBarRadius: 4 } } : {}),
      },
    },
    stroke: kind === "line" || kind === "area"
      ? { width: seriesCount > 1 ? 2.5 : 3, curve: "smooth" }
      : stacked ? { width: 1, colors: [DV.surface] } : { width: 0 },
    ...(kind === "area" ? { fill: { type: "gradient", gradient: { shadeIntensity: 0.2, opacityFrom: 0.38, opacityTo: 0.04 } } } : {}),
    grid,
    tooltip: {
      // The value shown on hover is the backend's number with the unit the
      // backend detected -- and no unit at all when it detected none.
      shared: seriesCount > 1 && !horizontal,
      intersect: false,
      y: { formatter: (val: number) => fullNumber(val, layout.percent ? undefined : unit) },
      ...axisTooltipX(categories, horizontal),
    },
    xaxis: {
      categories,
      // With horizontal bars ApexCharts swaps the axes: the x axis becomes the
      // value axis and the y axis carries the categories.
      labels: {
        style: horizontal ? MONO_LABEL : AXIS_LABEL,
        rotate: horizontal ? 0 : -32,
        rotateAlways: rotate,
        trim: false,
        hideOverlappingLabels: false,
        // The dashboard hides the value axis on ranked bars: the value label on each bar carries it.
        show: !(singleBar && horizontal),
        formatter: horizontal
          ? (value: string) => axisNumber(safeNumber(value), layout.percent ? "%" : unit)
          : (value: string) => truncate(String(value ?? ""), MAX_CATEGORY_CHARS),
      },
      axisBorder: { show: !horizontal, color: DV.grid },
      axisTicks: { show: false },
      // Only label an axis with a unit the backend actually detected.
      ...(horizontal && unit && !layout.percent && !singleBar
        ? { title: { text: unit, style: { color: DV.axis, fontWeight: 500 } } }
        : {}),
    },
    yaxis: {
      // Ranked bars: headroom past the longest bar so its value label is never clipped.
      ...(singleBar && horizontal && maxTotal > 0 ? { max: maxTotal * 1.18, min: 0 } : {}),
      labels: {
        style: horizontal ? { ...AXIS_LABEL, fontSize: "11.5px" } : MONO_LABEL,
        maxWidth: horizontal ? 200 : 90,
        formatter: horizontal
          ? (value: any) => truncate(String(value ?? ""), MAX_SIDE_CATEGORY_CHARS)
          : (value: number) => axisNumber(value, layout.percent ? "%" : unit),
      },
      ...(!horizontal && unit && !layout.percent
        ? { title: { text: unit, style: { color: DV.axis, fontWeight: 500 } } }
        : {}),
    },
  };
}

// ---------------------------------------------------------------------------
// Dashboard cards (chart.dashboard set): drawn the way the WFM dashboard's
// SmartChart.jsx draws that KPI -- one stack per category with a glossy
// top-to-bottom gradient and a 1px outline in the series colour, rounded top on
// the last segment, value labels inside the segments (or only the stack total /
// the rate on top), dashed light grid, legend on top with round swatches; area
// trends with a 0.4 -> 0.05 gradient; donuts with the share % inside each slice.
// Other kinds (line, combo, heatmap, funnel, gauge) keep the builder above, in
// the card's colours.
// ---------------------------------------------------------------------------

const DASH_AXIS = { colors: DV.axis, fontSize: "12px", fontWeight: 700 };
const DENSE_LABEL_BARS = 20; // SmartChart MAX_DENSE_LABEL_BARS: past this, value labels are dropped
// Grouped MI / SAT bars: each count bar carries its own progress % (SmartChart MI_SAT_LABEL_MAP).
const RATE_FOR: Record<string, string> = { "total mi": "% sat vs mi", "eligible mi for sat": "% sat vs eligible mi" };

function rateLabel(name: string, value: number | undefined): string {
  if (value === undefined || !Number.isFinite(value)) return "";
  return `${value.toFixed(1)}${/%|rate|pct|percent/i.test(name) ? "%" : ""}`;
}

function dashboardOptions(
  chart: AIChartSpec,
  layout: ChartLayout,
  base: ApexOptions,
  categories: string[],
  series: Named[],
  palette: string[],
  unit: string | undefined
): ApexOptions | null {
  const kpi = chart.dashboard as string;
  const { kind, horizontal } = layout;
  const rates = (chart.rates ?? []).map((r) => ({ name: r.name, data: (r.data ?? []).map(safeNumber) }));
  const n = categories.length;
  const legend = {
    ...base.legend,
    position: "top" as const,
    fontWeight: 800,
    markers: { size: 5, shape: "circle" as const },
    itemMargin: { horizontal: 12, vertical: 4 },
  };
  const tooltip = { y: { formatter: (val: number) => fullNumber(val, unit) } };

  if (kind === "pie" || kind === "donut") {
    // SmartChart DonutVariant: inner radius 45% of a 75% ring, 3deg gaps between slices,
    // the share % in white inside each slice of 5% or more, legend under the ring.
    return {
      ...base,
      labels: categories,
      legend: { ...legend, show: true, position: "bottom" },
      dataLabels: {
        enabled: true,
        formatter: (pct: number) => (pct < 5 ? "" : `${pct.toFixed(0)}%`),
        style: { fontSize: "13px", fontWeight: 800, colors: ["#fff"] },
        dropShadow: { enabled: false },
      },
      plotOptions: { pie: { expandOnClick: false, donut: { size: kind === "donut" ? "60%" : "0%", labels: { show: false } } } },
      stroke: { width: 3, colors: [DV.surface] },
      tooltip,
    };
  }

  if (kind === "area") {
    // SmartChart AreaVariant: unstacked areas, gradient 0.4 -> 0.05, 3px line, dots up to 50 points;
    // a rate rides a right-hand axis as a plain line (RIGHT_AXIS_CONFIGS).
    const dense = n > 14;
    const all = [...series.map((s) => s.name), ...rates.map((r) => r.name)];
    return {
      ...base,
      chart: { ...base.chart, type: "area", stacked: false },
      legend: { ...legend, show: all.length > 1 },
      dataLabels: { enabled: false },
      stroke: { width: all.map((_n, i) => (i < series.length && n > 100 ? 1.5 : 3)), curve: "smooth" },
      fill: {
        type: all.map((_n, i) => (i < series.length ? "gradient" : "solid")),
        gradient: { shadeIntensity: 0, opacityFrom: 0.4, opacityTo: 0.05, stops: [5, 95] },
      },
      markers: { size: n > 50 ? 0 : 4, strokeWidth: 0, hover: { size: 6 } },
      grid: { borderColor: DV.grid, strokeDashArray: 3, xaxis: { lines: { show: false } }, yaxis: { lines: { show: true } } },
      tooltip: { shared: true, intersect: false, ...tooltip, ...axisTooltipX(categories, false) },
      xaxis: {
        categories,
        ...(n > 14 ? { tickAmount: n > 60 ? 6 : 10 } : {}),
        labels: { style: DASH_AXIS, rotate: dense ? -35 : 0, rotateAlways: dense, hideOverlappingLabels: true },
        axisBorder: { show: false },
        axisTicks: { show: false },
      },
      yaxis: rates.length
        ? all.map((name, i) => {
            const isRate = i >= series.length;
            return {
              seriesName: isRate ? rates[0].name : series[0].name,
              opposite: isRate,
              show: i === 0 || i === series.length,
              labels: { style: DASH_AXIS, formatter: (v: number) => axisNumber(v) },
              ...(isRate ? { title: { text: name, style: { color: DV.axis, fontWeight: 700 } } } : {}),
            };
          })
        : { labels: { style: DASH_AXIS, formatter: (v: number) => axisNumber(v) } },
    };
  }

  if (kind !== "bar" || layout.funnel || layout.percent) return null;

  // SmartChart BarVariant / HBarVariant. Every series shares one stack, except the grouped
  // MI vs SAT bars (and other totals the backend keeps side by side).
  const grouped = !layout.stacked && series.length > 1;
  const mode = chart.labels ?? "segments";
  const crowded = n > DENSE_LABEL_BARS;
  const totals = categories.map((_c, i) => series.reduce((t, s) => t + (s.data[i] ?? 0), 0));
  const maxTotal = Math.max(0, ...totals);
  const rateText = (i: number) => rates.map((r) => rateLabel(r.name, r.data[i])).filter(Boolean).join(" · ");
  const pairedRate = series.map((s) => {
    const want = RATE_FOR[canonicalKey(kpi, s.name)];
    return want ? rates.find((r) => canonicalKey(kpi, r.name) === want) : undefined;
  });
  const showTotal = !grouped && !crowded && (rates.length > 0 || mode === "total");
  const rotate = !horizontal && n > 6;

  const dataLabels: ApexOptions["dataLabels"] = grouped
    ? {
        // Above each bar: its own progress % when it has one, else the count.
        enabled: !crowded && mode !== "none",
        formatter: (val: number, opts: any) => {
          const s = opts?.seriesIndex ?? 0;
          const i = opts?.dataPointIndex ?? 0;
          const rate = pairedRate[s];
          if (rate) return rateLabel(rate.name, rate.data[i]);
          // A rate no bar owns (Utilization Rate beside Supplied / Installed) sits on the last bar.
          if (rates.length && !pairedRate.some(Boolean) && s === series.length - 1) return rateText(i);
          return axisNumber(val);
        },
        offsetY: horizontal ? 0 : -18,
        offsetX: horizontal ? 6 : 0,
        ...(horizontal ? { textAnchor: "start" as const } : {}),
        style: { fontSize: "11px", fontWeight: 800, colors: [DV.axis] },
        background: { enabled: false },
        dropShadow: { enabled: false },
      }
    : {
        // Inside each segment in ink that reads on its fill; a sliver too thin to hold one stays blank.
        enabled: mode === "segments" && !crowded,
        formatter: (val: number) => (maxTotal > 0 && val / maxTotal >= 0.06 ? axisNumber(val) : ""),
        style: { fontSize: "12px", fontWeight: 800, colors: palette.map(getContrastText) },
        background: { enabled: false },
        dropShadow: { enabled: false },
      };

  return {
    ...base,
    chart: { ...base.chart, stacked: !grouped },
    legend: { ...legend, show: series.length > 1 },
    dataLabels,
    plotOptions: {
      bar: {
        horizontal,
        borderRadius: grouped ? 0 : horizontal ? 4 : 6,
        borderRadiusApplication: "end",
        borderRadiusWhenStacked: "last",
        columnWidth: n > 10 ? "70%" : grouped ? "60%" : "45%",
        barHeight: n > 15 ? "70%" : "55%",
        dataLabels: {
          position: grouped ? "top" : "center",
          total: {
            enabled: showTotal,
            // The rate (Utilization Rate, Productivity, SAT %) when there is one, else the stack's total.
            formatter: (val?: string, opts?: any) =>
              rates.length ? rateText(opts?.dataPointIndex ?? 0) : axisNumber(safeNumber(val)),
            offsetY: horizontal ? 0 : -4,
            offsetX: horizontal ? 6 : 0,
            style: {
              fontSize: "12px",
              fontWeight: 800,
              color: rates.length ? dashboardColour(kpi, rates[0].name, 0) : DV.axis,
            },
          },
        },
      },
    },
    // Glossy gradient on single / side-by-side bars; a stack is solid -- ApexCharts runs the
    // gradient per segment, which washed a tall bottom segment out to pink at its top.
    fill: series.length > 1 && !grouped
      ? { type: "solid", opacity: 0.95 }
      : {
          type: "gradient",
          gradient: {
            type: horizontal ? "horizontal" : "vertical",
            shadeIntensity: 0,
            opacityFrom: horizontal ? 0.8 : 1,
            opacityTo: horizontal ? 1 : 0.7,
            stops: [0, 100],
          },
        },
    stroke: { show: true, width: 1, colors: palette },
    grid: {
      borderColor: DV.grid,
      strokeDashArray: 3,
      xaxis: { lines: { show: horizontal } },
      yaxis: { lines: { show: !horizontal } },
    },
    tooltip: {
      shared: series.length > 1 && !horizontal,
      intersect: false,
      y: {
        formatter: (val: number, opts: any) => {
          const i = opts?.dataPointIndex ?? 0;
          const extra = rates.length && opts?.seriesIndex === series.length - 1 ? `  (${rateText(i)})` : "";
          return `${fullNumber(val, unit)}${extra}`;
        },
      },
      ...axisTooltipX(categories, horizontal),
    },
    xaxis: {
      categories,
      labels: {
        style: DASH_AXIS,
        rotate: horizontal ? 0 : -45,
        rotateAlways: rotate,
        trim: false,
        hideOverlappingLabels: horizontal,
        formatter: horizontal
          ? (value: string) => axisNumber(safeNumber(value))
          : (value: string) => truncate(String(value ?? ""), 12),
      },
      axisBorder: { show: false },
      axisTicks: { show: false },
    },
    yaxis: {
      // Room past the longest stack so its total / rate label is never clipped, on round ticks.
      ...(showTotal && maxTotal > 0 ? { ...niceAxis(maxTotal * (horizontal ? 1.15 : 1.12)), min: 0 } : {}),
      labels: {
        style: DASH_AXIS,
        maxWidth: horizontal ? 150 : 90,
        formatter: horizontal
          ? (value: any) => truncate(String(value ?? ""), 20)
          : (value: number) => axisNumber(value),
      },
    },
  };
}
