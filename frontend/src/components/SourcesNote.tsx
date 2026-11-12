import React from "react";
import { Database } from "lucide-react";
import type { AIQueryResponse } from "../api/client";

interface SourcesNoteProps {
  citations: NonNullable<AIQueryResponse["citations"]>;
  title?: string;
}

/** Only the data tables the live query actually read (with how recent their data is).
 * Verified-question / KPI-rule citations are not shown. Built by the backend
 * (gmr_agents/checks/grounding.py data_citations), never by the LLM. */
export const SourcesNote: React.FC<SourcesNoteProps> = ({ citations, title = "Sources" }) => {
  const items = citations.filter((c) => c.source === "Live query" && c.location?.trim());
  if (!items.length) return null;
  return (
    <section className="chat-result-section">
      <div className="chat-sources">
        <Database size={12} aria-hidden="true" />
        <strong>{title}:</strong>
        {items.map((c) => (
          <span key={c.id} className="chat-sources-item" title={c.source}>
            {c.location}
          </span>
        ))}
      </div>
    </section>
  );
};
