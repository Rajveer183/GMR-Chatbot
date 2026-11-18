import React, { useEffect, useMemo, useRef, useState } from "react";
import { CalendarClock, Database, RefreshCw, X } from "lucide-react";
import type { AIQueryResponse } from "../api/client";

interface SourcesButtonProps {
  citations: NonNullable<AIQueryResponse["citations"]>;
  label?: string;
  title?: string;
}

interface SourceTable {
  qualified: string;
  name: string;
  schema: string;
  upTo: string | null;
}

const ACRONYMS = new Set(["mi", "sat", "om", "pf", "hes", "mdm", "wfm", "kpi", "dt", "sku", "oem", "dcu", "ct", "pt", "id", "1ph", "3ph", "lt", "ht", "puvvnl", "discom", "mv"]);

/** "gold_mi_progress_daily" -> "MI Progress Daily". */
function humanize(table: string): string {
  return table
    .replace(/^(gold|silver|bronze)_/i, "")
    .split(/[_\s]+/)
    .filter(Boolean)
    .map((w) => (ACRONYMS.has(w.toLowerCase()) ? w.toUpperCase() : w.charAt(0).toUpperCase() + w.slice(1)))
    .join(" ");
}

/** Split on commas that are not inside parentheses. */
function splitTop(text: string): string[] {
  const out: string[] = [];
  let depth = 0;
  let cur = "";
  for (const ch of text) {
    if (ch === "(") depth++;
    if (ch === ")") depth = Math.max(0, depth - 1);
    if (ch === "," && depth === 0) {
      out.push(cur.trim());
      cur = "";
    } else cur += ch;
  }
  if (cur.trim()) out.push(cur.trim());
  return out;
}

/** The backend's live-query citation is one string:
 * "schema.table (data up to Aug 2026), schema.other (data refreshed 18 Nov 2026 10:00)". */
function parse(location: string): { tables: SourceTable[]; refreshed: string | null } {
  let text = location.trim();
  let refreshed: string | null = null;
  const m = text.match(/\s*\(data refreshed ([^)]*)\)\s*$/i);
  if (m) {
    refreshed = m[1].trim();
    text = text.slice(0, m.index).trim();
  }
  const tables = splitTop(text).map((part) => {
    const pm = part.match(/^(.*?)\s*\((?:data up to\s*)?([^)]*)\)\s*$/i);
    const qualified = (pm ? pm[1] : part).trim();
    const dot = qualified.lastIndexOf(".");
    const raw = dot >= 0 ? qualified.slice(dot + 1) : qualified;
    return {
      qualified,
      name: humanize(raw),
      schema: dot >= 0 ? qualified.slice(0, dot) : "",
      upTo: pm ? pm[2].trim() : null,
    };
  });
  return { tables, refreshed };
}

/** "Sources" action under an answer (beside Export): click to see the data tables the live
 * query actually read, with how recent each one's data is. Verified-question / KPI-rule
 * citations are not shown. Built by the backend (gmr_agents/checks/grounding.py
 * data_citations), never by the LLM. Renders nothing when the answer read no table. */
export const SourcesButton: React.FC<SourcesButtonProps> = ({ citations, label = "Sources", title = "View sources" }) => {
  const [open, setOpen] = useState(false);
  const wrapRef = useRef<HTMLDivElement>(null);

  const { tables, refreshed } = useMemo(() => {
    const all: SourceTable[] = [];
    let refreshedAt: string | null = null;
    for (const c of citations) {
      if (c.source !== "Live query" || !c.location?.trim()) continue;
      const p = parse(c.location);
      for (const t of p.tables) if (!all.some((a) => a.qualified === t.qualified)) all.push(t);
      refreshedAt = refreshedAt ?? p.refreshed;
    }
    return { tables: all, refreshed: refreshedAt };
  }, [citations]);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", onDown);
    window.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      window.removeEventListener("keydown", onKey);
    };
  }, [open]);

  if (!tables.length) return null;
  return (
    <div className="export-menu" ref={wrapRef}>
      <button
        type="button"
        className={`answer-action-btn${open ? " is-active" : ""}`}
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        aria-haspopup="dialog"
        aria-label={title}
      >
        <Database size={13} />
        <span>{label}</span>
      </button>
      {open ? (
        <div className="sources-pop" role="dialog" aria-label={label}>
          <div className="sources-pop-head">
            <span className="sources-pop-title">
              {label}
              <span className="sources-pop-count">{tables.length}</span>
            </span>
            <button type="button" className="sources-pop-close" onClick={() => setOpen(false)} aria-label="Close">
              <X size={14} />
            </button>
          </div>
          <ul className="sources-pop-list">
            {tables.map((t, i) => (
              <li key={t.qualified} className="sources-pop-item">
                <span className="sources-pop-num">{i + 1}</span>
                <div className="sources-pop-body">
                  <div className="sources-pop-name">{t.name}</div>
                  <div className="sources-pop-table">{t.qualified}</div>
                  {t.upTo ? (
                    <div className="sources-pop-upto">
                      <CalendarClock size={11} aria-hidden="true" />
                      Data up to {t.upTo}
                    </div>
                  ) : null}
                </div>
              </li>
            ))}
          </ul>
          {refreshed ? (
            <div className="sources-pop-foot">
              <RefreshCw size={11} aria-hidden="true" />
              Data refreshed {refreshed}
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
};
