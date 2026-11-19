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

const MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const MONTH_START_RE = /^(\d{4})-(0[1-9]|1[0-2])(?:-01)?$/;
const MONTH_LABEL_RE = /^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) (\d{4})$/;

/**
 * Chart categories that are all months ('2025-11-01' / '2025-11') read 'Nov 2025', as the table's
 * month columns do (backend blocks.month_cols). Display only: chart.x keeps the ISO text, which the
 * date checks (chronological, graph modes) depend on. Days and mixed axes are left as sent.
 */
export function monthLabels(cats: string[]): string[] {
  if (cats.length < 2 || !cats.every((c) => MONTH_START_RE.test(c))) return cats;
  return cats.map(monthLabel);
}

/** '2025-11-01' / '2025-11' -> 'Nov 2025'; anything else unchanged. */
export function monthLabel(text: string): string {
  const m = MONTH_START_RE.exec(text);
  return m ? `${MONTH_NAMES[Number(m[2]) - 1]} ${m[1]}` : text;
}

/**
 * Table columns that hold months, by the backend's rule (blocks.month_cols): every value apart from a
 * Total row is the 1st of a month (or 'YYYY-MM'), and the column is named as a month or has 2+ rows.
 * Done here too so answers saved before the backend did it (and cached ones) read 'Nov 2025' as well.
 */
export function monthColumns(columns: string[], rows: Record<string, unknown>[]): Set<string> {
  const out = new Set<string>();
  for (const name of columns) {
    const values = rows
      .map((row) => (row[name] == null ? "" : String(row[name]).trim()))
      .filter((v) => v !== "" && !/^(grand )?total$/i.test(v));
    if (values.length && values.every((v) => MONTH_START_RE.test(v)) && (/month/i.test(name) || values.length > 1)) out.add(name);
  }
  return out;
}

/** 'Nov 2025' -> a sortable number (year * 12 + month); null for anything else. */
export function monthValueOf(value: unknown): number | null {
  const m = typeof value === "string" ? MONTH_LABEL_RE.exec(value.trim()) : null;
  return m ? Number(m[2]) * 12 + MONTH_NAMES.indexOf(m[1]) : null;
}
