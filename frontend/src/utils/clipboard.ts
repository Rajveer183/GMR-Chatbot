import type { AIKPI, AIQueryResponse, AITableSpec } from "../api/client";
import { formatTableCell } from "./formatters";

/**
 * A copy payload in two flavours. Rich targets (Excel, Word, Outlook, Google
 * Docs/Sheets, Teams) read `html` and keep the table as a real grid; plain
 * targets (Notepad, WhatsApp, terminals) read `text`, where tables are
 * tab-separated so a paste into a spreadsheet still lands in columns.
 */
export interface CopyPayload {
  text: string;
  html: string;
}

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function stripMarkdown(text: string): string {
  return text.replace(/\*\*(.*?)\*\*/g, "$1").replace(/\[(.*?)\]\((.*?)\)/g, "$1");
}

function markdownToHtml(text: string): string {
  return escapeHtml(text).replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>").replace(/\n/g, "<br>");
}

// A tab or newline inside a cell would break the column/row grid on paste.
const tsvCell = (value: unknown) => formatTableCell(value).replace(/[\t\r\n]+/g, " ");

const headerLabel = (name: string) => name.replace(/_/g, " ").toUpperCase();

const CELL_STYLE = "border:1px solid #cbd5e1;padding:4px 8px;";

export function tableToPayload(table: AITableSpec): CopyPayload {
  const rows = table.rows ?? [];
  const text = [
    table.columns.map(headerLabel).join("\t"),
    ...rows.map((row) => table.columns.map((c) => tsvCell(row[c])).join("\t")),
  ].join("\n");

  const head = table.columns
    .map((c) => `<th style="${CELL_STYLE}background:#f1f5f9;text-align:left;">${escapeHtml(headerLabel(c))}</th>`)
    .join("");
  const body = rows
    .map(
      (row) =>
        `<tr>${table.columns
          .map((c) => `<td style="${CELL_STYLE}">${escapeHtml(formatTableCell(row[c]))}</td>`)
          .join("")}</tr>`
    )
    .join("");
  const html = `<table style="border-collapse:collapse;font-family:Arial,sans-serif;font-size:12px;"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;

  return { text, html };
}

export function kpiToPayload(kpi: Pick<AIKPI, "label" | "formatted_value" | "description">): CopyPayload {
  const text = `${kpi.label}: ${kpi.formatted_value}${kpi.description ? ` (${kpi.description})` : ""}`;
  const html = `<p><strong>${escapeHtml(kpi.label)}:</strong> ${escapeHtml(kpi.formatted_value)}${
    kpi.description ? ` <em>(${escapeHtml(kpi.description)})</em>` : ""
  }</p>`;
  return { text, html };
}

function kpisToPayload(kpis: AIKPI[]): CopyPayload {
  const text = kpis.map((k) => `${k.label}\t${k.formatted_value}`).join("\n");
  const rows = kpis
    .map(
      (k) =>
        `<tr><td style="${CELL_STYLE}font-weight:bold;">${escapeHtml(k.label)}</td><td style="${CELL_STYLE}">${escapeHtml(
          k.formatted_value
        )}</td></tr>`
    )
    .join("");
  const html = `<table style="border-collapse:collapse;font-family:Arial,sans-serif;font-size:12px;"><tbody>${rows}</tbody></table>`;
  return { text, html };
}

export function summaryToPayload(summary: string): CopyPayload {
  return { text: stripMarkdown(summary), html: `<p>${markdownToHtml(summary)}</p>` };
}

/** The whole answer -- summary, key metrics, every table and insights -- in reading order. */
export function resultToPayload(result: AIQueryResponse): CopyPayload {
  const summary = summaryToPayload(result.summary);
  const textParts: string[] = [summary.text];
  const htmlParts: string[] = [summary.html];

  const kpis = result.kpis ?? [];
  if (kpis.length) {
    const p = kpisToPayload(kpis);
    textParts.push(`Key Metrics\n${p.text}`);
    htmlParts.push(`<h4>Key Metrics</h4>${p.html}`);
  }

  for (const table of result.tables ?? []) {
    const p = tableToPayload(table);
    const truncated = table.total_rows > (table.rows?.length ?? 0);
    const note = truncated ? ` (showing ${table.rows.length} of ${table.total_rows.toLocaleString()} rows)` : "";
    textParts.push(`${table.title}${note}\n${p.text}`);
    htmlParts.push(`<h4>${escapeHtml(table.title)}${escapeHtml(note)}</h4>${p.html}`);
  }

  const insights = result.insights ?? [];
  if (insights.length) {
    textParts.push(`Insights\n${insights.map((i) => `- ${stripMarkdown(i)}`).join("\n")}`);
    htmlParts.push(`<h4>Insights</h4><ul>${insights.map((i) => `<li>${markdownToHtml(i)}</li>`).join("")}</ul>`);
  }

  return { text: textParts.join("\n\n"), html: htmlParts.join("") };
}

/**
 * Writes both flavours when the browser supports ClipboardItem, falling back
 * to plain text, then to the legacy hidden-textarea copy (non-HTTPS contexts).
 */
export async function copyToClipboard(payload: CopyPayload): Promise<void> {
  try {
    if (typeof ClipboardItem !== "undefined" && navigator.clipboard?.write) {
      await navigator.clipboard.write([
        new ClipboardItem({
          "text/plain": new Blob([payload.text], { type: "text/plain" }),
          "text/html": new Blob([payload.html], { type: "text/html" }),
        }),
      ]);
      return;
    }
    await navigator.clipboard.writeText(payload.text);
  } catch {
    const textarea = document.createElement("textarea");
    textarea.value = payload.text;
    textarea.style.position = "fixed";
    textarea.style.opacity = "0";
    document.body.appendChild(textarea);
    textarea.select();
    try {
      document.execCommand("copy");
    } catch {
      // give up silently -- copied state simply won't show
    }
    document.body.removeChild(textarea);
  }
}
