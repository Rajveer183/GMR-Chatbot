import React from "react";
import { Maximize2 } from "lucide-react";

interface ChartCardProps {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
  onExpand?: () => void;
}

export const ChartCard: React.FC<ChartCardProps> = ({ title, subtitle, children, onExpand }) => {
  return (
    <div className="chart-card chat-chart-card">
      <div className="chart-card-header">
        <div>
          <h3 className="chart-card-title">{title}</h3>
          {subtitle ? <p className="chart-card-subtitle">{subtitle}</p> : null}
        </div>
        {onExpand ? (
          <button type="button" className="chart-expand-btn" onClick={onExpand} aria-label={`Expand ${title}`}>
            <Maximize2 size={16} />
          </button>
        ) : null}
      </div>
      <div className="chart-card-body">
        <div className="chart-plot chat-chart-plot">{children}</div>
      </div>
    </div>
  );
};
