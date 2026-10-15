/**
 * Presentation helpers only.
 *
 * Every figure in a response is computed AND formatted by the backend
 * (`format_number` in backend/agent/executor.py, applied throughout
 * backend/agent/shape.py). Nothing in this file may re-derive, re-scale or
 * re-label a value.
 *
 * This is not a stylistic rule. The previous version of this file appended a
 * hardcoded " Rs/Cr" to any column whose name merely *contained* "amount",
 * "dues", "billing" or "ros" -- so a record count, a quantity in tonnes, or a
 * column called "GROSS WEIGHT" was displayed as a rupee-crore figure. The unit
 * shown must be the one the backend actually detected from the column header
 * (`_detect_unit` in backend/profiler.py), and when it detected none, none is
 * shown.
 */

/** Render a table cell exactly as the backend formatted it. */
export function formatTableCell(value: unknown): string {
  if (value == null) return "";
  return String(value);
}

/**
 * The numeric value behind a backend-formatted string ("1,234.00" -> 1234).
 *
 * Used ONLY to order rows when a column header is clicked, and to keep that
 * ordering numeric instead of lexicographic (where "9" sorts after "10"). The
 * cell always keeps its original backend-formatted text -- this value is never
 * displayed. Returns null for anything that is not purely a number, so mixed
 * text columns fall back to string comparison.
 */
export function numericValueOf(value: unknown): number | null {
  if (typeof value === "number") return Number.isFinite(value) ? value : null;
  if (typeof value !== "string") return null;
  const cleaned = value.replace(/,/g, "").trim();
  if (!/^[+-]?\d*\.?\d+$/.test(cleaned)) return null;
  const num = Number(cleaned);
  return Number.isFinite(num) ? num : null;
}

/**
 * Value text for a chart axis or tooltip.
 *
 * `unit` is whatever the backend detected from the column header and is often
 * absent -- when it is, no unit is appended rather than a guessed one.
 */
export function formatChartValue(value: number, unit?: string): string {
  if (!Number.isFinite(value)) return "";
  const text = value.toLocaleString(undefined, { maximumFractionDigits: 2 });
  return unit ? `${text} ${unit}` : text;
}
