import type { AIQueryResponse, AITableSpec } from "../api/client";
import { formatTableCell } from "./formatters";

// TTS is billed per character, so tables are read as a short preview rather
// than cell-by-cell: a 500-row table would otherwise cost thousands of chars.
// The backend (shared/tts.py) cuts the text at a character cap and rewrites
// symbols / dates / markup for the voice, so parts are ordered most-useful
// first: summary, metric cards, insights, then the table preview.
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
  const title = table.display_title || table.title;
  const parts = [sentence(`Table ${title}, ${total} ${total === 1 ? "row" : "rows"}`)];
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

/** What the Play button reads: what the answer shows, in the order it matters. */
export function resultToSpeech(result: AIQueryResponse): string {
  const summary = stripMarkdown(result.summary ?? "");
  const parts = [sentence(summary)];

  // Metric cards as shown (a feeder comparison hides them); a card whose value the
  // summary already said is skipped -- it would only repeat the opening sentence.
  const kpis = (result.feeder_compare ? [] : result.kpis ?? []).filter(
    (k) => !(k.formatted_value && summary.includes(k.formatted_value)),
  );
  if (kpis.length) {
    parts.push(`Key metrics: ${kpis.map((k) => sentence(`${k.display_label || k.label}, ${k.formatted_value}`)).join(" ")}`);
  }

  const insights = (result.insights ?? []).map((s) => s.trim()).filter(Boolean);
  if (insights.length) parts.push(`Insights: ${insights.map(sentence).join(" ")}`);

  for (const table of result.tables ?? []) parts.push(tableToSpeech(table));

  return parts.join(" ");
}
