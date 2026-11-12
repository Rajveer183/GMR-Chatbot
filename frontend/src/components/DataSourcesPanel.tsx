import React, { useEffect, useMemo, useState } from "react";
import { Search, X } from "lucide-react";
import { apiClient } from "../api/client";
import type { DatasetSummary, FreshnessRow } from "../api/client";

/** Data Sources pop-up: every table the assistant can query (from /api/datasets), with its
 * row count and, where tracked, how recent its data is (from /api/data/freshness). */

const DOMAIN_LABELS: Record<string, string> = { mdm: "MDM", wfm: "WFM", hes: "HES" };

export const DataSourcesPanel: React.FC<{ open: boolean; onClose: () => void; datasets: DatasetSummary[] }> = ({
  open,
  onClose,
  datasets,
}) => {
  const [query, setQuery] = useState("");
  const [fresh, setFresh] = useState<Map<string, FreshnessRow>>(new Map());

  useEffect(() => {
    if (!open || fresh.size) return;
    apiClient
      .getFreshness()
      .then((f) => setFresh(new Map((f.tables ?? []).map((r) => [r.table, r]))))
      .catch(() => undefined); // freshness is optional; the list still shows
  }, [open, fresh.size]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  const groups = useMemo(() => {
    const q = query.trim().toLowerCase();
    const byDomain = new Map<string, DatasetSummary[]>();
    for (const ds of datasets) {
      if (q && !ds.display_name.toLowerCase().includes(q) && !ds.dataset_id.toLowerCase().includes(q)) continue;
      const key = ds.domain || "other";
      if (!byDomain.has(key)) byDomain.set(key, []);
      byDomain.get(key)!.push(ds);
    }
    return [...byDomain.entries()]
      .sort((a, b) => a[0].localeCompare(b[0]))
      .map(([domain, items]) => [domain, items.sort((a, b) => a.display_name.localeCompare(b.display_name))] as const);
  }, [datasets, query]);

  if (!open) return null;
  const shown = groups.reduce((n, [, items]) => n + items.length, 0);

  return (
    <div className="ds-overlay" onClick={onClose}>
      <div className="ds-modal" role="dialog" aria-label="Data Sources" onClick={(e) => e.stopPropagation()}>
        <div className="ds-top">
          <h3>
            Data Sources <span className="ds-total">{datasets.length}</span>
          </h3>
          <button type="button" className="ins-icon-btn" onClick={onClose} title="Close" aria-label="Close">
            <X size={16} />
          </button>
        </div>
        <div className="ds-search">
          <Search size={13} />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search tables"
            aria-label="Search tables"
            autoFocus
          />
        </div>
        <div className="ds-body">
          {!datasets.length ? <p className="ins-empty">No data sources loaded yet.</p> : null}
          {datasets.length && !shown ? <p className="ins-empty">No tables match “{query}”.</p> : null}
          {groups.map(([domain, items]) => (
            <section key={domain}>
              <h4 className="ins-h">
                {DOMAIN_LABELS[domain] ?? domain.toUpperCase()} · {items.length}
              </h4>
              <table className="ins-table">
                <thead>
                  <tr>
                    <th>Table</th>
                    <th>Rows</th>
                    <th>Data up to</th>
                  </tr>
                </thead>
                <tbody>
                  {items.map((ds) => {
                    const f = fresh.get(ds.display_name);
                    return (
                      <tr key={ds.dataset_id}>
                        <td className="ins-mono" title={ds.dataset_id}>{ds.display_name}</td>
                        <td>{ds.row_count.toLocaleString("en-IN")}</td>
                        <td>
                          {f ? (
                            <span className={`ins-pill ${f.stale ? "bad" : "ok"}`} title={f.stale ? `${f.behind} behind` : "current"}>
                              {f.label.replace(/^data up to /, "")}
                            </span>
                          ) : (
                            "–"
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </section>
          ))}
        </div>
      </div>
    </div>
  );
};

export default DataSourcesPanel;
