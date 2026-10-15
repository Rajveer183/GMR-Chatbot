import type { AIQueryResponse, AITableSpec } from "../api/client";
import { formatTableCell } from "./formatters";

// TTS is billed per character, so tables are read as a short preview rather
// than cell-by-cell: a 500-row table would otherwise cost thousands of chars.
const SPOKEN_ROWS = 3;
const SPOKEN_COLUMNS = 4;

const stripMarkdown = (text: string) => text.replace(/\*\*(.*?)\*\*/g, "$1").replace(/\[(.*?)\]\((.*?)\)/g, "$1");

const spokenHeader = (name: string) => name.replace(/_/g, " ").toLowerCase();

const sentence = (text: string) => {
  const t = text.trim();
  return /[.!?]$/.test(t) ? t : `${t}.`;
};

function tableToSpeech(table: AITableSpec): string {
  const [lead, ...rest] = table.columns.slice(0, SPOKEN_COLUMNS);
  // A grand-total row only repeats what the summary already said.
  const isTotal = (row: Record<string, unknown>) =>
    lead !== undefined && formatTableCell(row[lead]).trim().toLowerCase() === "total";
  const rows = (table.rows ?? []).filter((row) => !isTotal(row)).slice(0, SPOKEN_ROWS);
  const total = table.total_rows;
  const parts = [sentence(`Table ${table.title}, ${total} ${total === 1 ? "row" : "rows"}`)];
  if (rows.length && lead !== undefined) {
    // "Jan 2026: total 406,285, feeder 48." -- the first column names the row,
    // so its header isn't repeated on every line.
    const lines = rows.map((row) => {
      const values = rest
        .map((c) => ({ c, v: formatTableCell(row[c]).trim() }))
        .filter(({ v }) => v)
        .map(({ c, v }) => `${spokenHeader(c)} ${v}`)
        .join(", ");
      const name = formatTableCell(row[lead]).trim();
      return sentence(values ? `${name}: ${values}` : name);
    });
    parts.push(total > rows.length ? `First ${rows.length}: ${lines.join(" ")}` : lines.join(" "));
  }
  return parts.join(" ");
}

/** What the Play button reads: summary, then metric cards, then a preview of each table. */
export function resultToSpeech(result: AIQueryResponse): string {
  const parts = [sentence(stripMarkdown(result.summary))];

  const kpis = result.kpis ?? [];
  if (kpis.length) {
    parts.push(`Key metrics: ${kpis.map((k) => sentence(`${k.label}, ${k.formatted_value}`)).join(" ")}`);
  }

  for (const table of result.tables ?? []) parts.push(tableToSpeech(table));

  return parts.join(" ");
}
