import React from "react";
import { Database } from "lucide-react";
import type { AIQueryResponse } from "../api/client";

interface SourcesNoteProps {
  citations: NonNullable<AIQueryResponse["citations"]>;
  title?: string;
}

/** Where the answer's figures came from: the tables the live query read (with how recent
 * their data is), and the verified pattern / business rule it followed. Built by the backend
 * from what actually ran (gmr_agents/checks/grounding.py data_citations), never by the LLM. */
const LABELS: Record<string, string> = {
  "examples.yaml (verified)": "Verified question",
  "business_rules.yaml": "KPI rule",
};

export const SourcesNote: React.FC<SourcesNoteProps> = ({ citations, title = "Sources" }) => {
  const items = citations.filter((c) => c.location?.trim());
  if (!items.length) return null;
  return (
    <section className="chat-result-section">
      <div className="chat-sources">
        <Database size={12} aria-hidden="true" />
        <strong>{title}:</strong>
        {items.map((c) => (
          <span key={c.id} className="chat-sources-item" title={c.source}>
            {c.source === "Live query" ? c.location : `${LABELS[c.source] ?? c.source}: ${c.location}`}
          </span>
        ))}
      </div>
    </section>
  );
};
