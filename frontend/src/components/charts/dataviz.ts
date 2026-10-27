/**
 * Chart design tokens and number formats, taken from the WFM/HES dashboard
 * (Documents/WFM/HES-FRONTEND-GMR/src/theme/dataviz.tokens.js -- the LIGHT block,
 * which is the dashboard's reference palette -- and src/utils/chartUtils.js).
 * Kept identical so a chart in the chatbot reads like the same card on the dashboard.
 */

export const DV = {
  surface: "#FFFFFF",
  surfaceAlt: "#F1F5F9", // bar track behind ranked bars
  grid: "#E2E8F0",
  axis: "#6B7789",
  text: "#334155",
  // §3 Rule 1: ONE hue for every ranked magnitude bar.
  series: "#1A5FA8",
  seriesSoft: "#DBE7F5",
  // §3 Rule 4: ordered bands (age buckets, stages) take one hue, light -> dark.
  bandRamp: ["#8FB8E2", "#5590CB", "#2A66AB", "#0B3D7B"],
  // §3 Rule 5: genuinely categorical series -- fixed 8 hues, never cycled.
  palette: ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
  // Dashboard gauge / pareto accents (chartUtils.js GMR).
  good: "#0d9488",
  accent: "#c2410c",
};

export const MONO = '"SF Mono", ui-monospace, Consolas, monospace';
export const AXIS_TICK = { fontSize: 11, fontWeight: 500, fill: DV.axis };

/** Full figure in Indian grouping (42,14,678), as the dashboard's labels and tooltips show it. */
export function fullNumber(value: number, unit?: string): string {
  if (!Number.isFinite(value)) return "—";
  const text = Number.isInteger(value)
    ? value.toLocaleString("en-IN")
    : value.toLocaleString("en-IN", { maximumFractionDigits: 2 });
  return unit ? (unit === "%" ? `${text}%` : `${text} ${unit}`) : text;
}

/** Axis ticks: the dashboard's formatIndianNumber (1.2Cr / 42.15L / 9,876). */
export function axisNumber(value: number, unit?: string): string {
  if (!Number.isFinite(value)) return "";
  const abs = Math.abs(value);
  let text: string;
  if (abs >= 10000000) text = `${(value / 10000000).toFixed(1)}Cr`;
  else if (abs >= 100000) text = `${(value / 100000).toFixed(2)}L`;
  else text = value.toLocaleString("en-IN", { maximumFractionDigits: abs >= 1000 ? 0 : 2 });
  return unit === "%" ? `${text}%` : text;
}

/** Axis top past `value` on a round step (1 / 2 / 2.5 / 5 x 10^n), about five ticks: the
 * dashboard's 0 / 10,000 / 20,000 ... instead of 0 / 9,882.66 / 19,765.31. */
export function niceAxis(value: number): { max: number; tickAmount: number } {
  if (!(value > 0)) return { max: 1, tickAmount: 1 };
  const raw = value / 5;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? 10 * mag;
  const ticks = Math.ceil(value / step);
  return { max: ticks * step, tickAmount: ticks };
}

export function truncate(text: string, max: number): string {
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
}

/** Ordered bands (age buckets, "0-30 days", "SAT-1".. , "91-120") use the band ramp; anything else is categorical. */
export function isOrderedBands(names: string[]): boolean {
  if (names.length < 2) return false;
  return names.every((n) => /\d/.test(n) && /(day|band|age|bucket|\d+\s*[-–+]|\d+\s*plus|sat-?\d)/i.test(n));
}

/** Colour for series i of n: band ramp (spread light -> dark) for ordered bands, else the categorical palette. */
export function seriesColour(i: number, names: string[]): string {
  if (isOrderedBands(names)) {
    const ramp = DV.bandRamp;
    if (names.length <= ramp.length) return ramp[Math.round((i * (ramp.length - 1)) / Math.max(names.length - 1, 1))];
    return ramp[Math.min(ramp.length - 1, Math.floor((i * ramp.length) / names.length))];
  }
  return DV.palette[i % DV.palette.length];
}
