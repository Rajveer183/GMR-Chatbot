import React from "react";
import { AlertTriangle } from "lucide-react";

interface WarningBoxProps {
  warnings: string[];
  title?: string;
}

export const WarningBox: React.FC<WarningBoxProps> = ({ warnings, title = "Warnings" }) => {
  if (!warnings?.length) return null;
  return (
    <section className="chat-result-section">
      <div className="chat-warning-card">
        <div className="chat-warning-header">
          <AlertTriangle size={16} />
          <strong>{title}</strong>
        </div>
        <ul className="chat-warning-list">
          {warnings.map((w, i) => (
            <li key={i}>{w}</li>
          ))}
        </ul>
      </div>
    </section>
  );
};
