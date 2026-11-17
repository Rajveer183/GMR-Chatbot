/** Hover data for a chart in the downloaded HTML copy of an answer (utils/snapshot.ts).
 *
 * The saved page has no ApexCharts library, so its tooltips cannot be drawn the normal way. The
 * figures the live tooltip would show are formatted here, with the chart's own tooltip formatter,
 * and stored on the chart's wrapper as JSON; a small script in the saved page shows them on hover. */
import type { ApexOptions } from "apexcharts";
import type { ChartLayout } from "../chartOptions";
import { fullNumber } from "./dataviz";

export interface ChartTip {
  /** How the hovered point is found: by the plot x position with points on the ticks (line/area),
   * between the ticks (vertical bars), or from the hovered mark itself (horizontal bars, pie, heatmap). */
  mode: "on" | "between" | "hit" | "slice";
  cats: string[];
  series: { name: string; color?: string; values: string[] }[];
}

type AnySeries = number[] | { name: string; data: (number | { x: string; y: number })[] }[];

export function chartTipData(categories: string[], layout: ChartLayout, series: AnySeries, options: ApexOptions): ChartTip | null {
  if (layout.kind === "radialBar" || options.tooltip?.enabled === false) return null;  // the live gauge has no tooltip either
  const yOpt = options.tooltip?.y;
  const formatter = !Array.isArray(yOpt) && typeof yOpt?.formatter === "function" ? yOpt.formatter : null;
  const format = (val: number, seriesIndex: number, dataPointIndex: number): string => {
    if (formatter) {
      try {
        return String(formatter(val, { seriesIndex, dataPointIndex } as any));
      } catch {
        /* fall back to the plain number */
      }
    }
    return fullNumber(val);
  };
  const colors = (options.colors ?? []).map((c) => (typeof c === "string" ? c : undefined));

  if (layout.kind === "pie" || layout.kind === "donut") {
    const values = (series as number[]).map((v, j) => format(v, 0, j));
    return { mode: "slice", cats: categories, series: [{ name: "", values }] };
  }
  const named = series as { name: string; data: (number | { x: string; y: number })[] }[];
  const mode = layout.kind === "heatmap" || layout.horizontal ? "hit" : layout.kind === "bar" ? "between" : "on";
  return {
    mode,
    cats: categories,
    series: named.map((s, i) => ({
      name: s.name,
      ...(colors[i] ? { color: colors[i] } : {}),
      values: s.data.map((d, j) => format(typeof d === "number" ? d : d.y, i, j)),
    })),
  };
}
