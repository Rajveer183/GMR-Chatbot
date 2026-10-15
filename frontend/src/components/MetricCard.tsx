import React from "react";
import { kpiToPayload } from "../utils/clipboard";
import { CopyButton } from "./CopyButton";

interface MetricCardProps {
  label: string;
  /** Already computed and formatted by the backend (kpi.formatted_value, built
   * by `format_number` with the column's detected unit). Rendered verbatim. */
  formattedValue: string;
  subtitle?: string;
}

// This card deliberately has no formatting logic. It used to strip a trailing
// " Rs/Cr" off the backend's formatted value and then re-append that same unit
// only when it matched that one hardcoded string -- so any other unit the
// profiler detected survived only by accident, and an `isCurrency` prop guessed
// at currency-ness from whether the label contained the word "unique". The
// backend already knows the unit; there is nothing to decide here.
export const MetricCard: React.FC<MetricCardProps> = ({ label, formattedValue, subtitle }) => {
  return (
    <div className="kpi-card">
      <div className="kpi-card-header">
        <span className="kpi-label">{label}</span>
        <CopyButton
          compact
          className="kpi-copy-btn"
          title={`Copy ${label}`}
          getPayload={() => kpiToPayload({ label, formatted_value: formattedValue, description: subtitle })}
        />
      </div>
      <div className="kpi-card-body">
        <span className="kpi-value">{formattedValue}</span>
        {subtitle && <span className="kpi-subtitle">{subtitle}</span>}
      </div>
    </div>
  );
};
