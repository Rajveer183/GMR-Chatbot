import React from "react";
import { AlertTriangle } from "lucide-react";

interface WarningBoxProps {
  warnings: string[];
}

export const WarningBox: React.FC<WarningBoxProps> = ({ warnings }) => {
  if (!warnings?.length) return null;
  return (
    <section className="chat-result-section">
      <div className="chat-warning-card">
        <div className="chat-warning-header">
          <AlertTriangle size={16} />
          <strong>Warnings</strong>
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
