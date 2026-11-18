import React, { useEffect, useRef, useState } from "react";
import { Activity, ChevronDown } from "lucide-react";
import type { GraphMode } from "./charts/graphModes";
import { MODE_LABEL_KEY, MULTI_LABEL_KEY } from "./charts/graphModes";
import { uiText } from "../i18n/languages";

interface GraphModeMenuProps {
  modes: GraphMode[];
  /** The picked mode; null = the chart as the answer drew it. */
  active: GraphMode | null;
  /** The mode the answer drew it in (marked while nothing else is picked). */
  original: GraphMode | null;
  /** Several series: Area / Line read "Multi Area" / "Multi Line", as on the dashboard. */
  multi: boolean;
  onPick: (mode: GraphMode | null) => void;
  lang?: string | null;
}

/** The dashboard's "Graphs Modes" pill: the first entry goes back to the answer's own chart. */
export const GraphModeMenu: React.FC<GraphModeMenuProps> = ({ modes, active, original, multi, onPick, lang }) => {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const away = (e: MouseEvent) => { if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false); };
    const esc = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(false); };
    document.addEventListener("mousedown", away);
    document.addEventListener("keydown", esc);
    return () => { document.removeEventListener("mousedown", away); document.removeEventListener("keydown", esc); };
  }, [open]);
  const label = (m: GraphMode) => uiText(lang, (multi && MULTI_LABEL_KEY[m]) || MODE_LABEL_KEY[m]);
  const shown = active ?? original;
  const pick = (m: GraphMode | null) => { onPick(m); setOpen(false); };
  return (
    <div className="graph-mode no-export" ref={ref}>
      <button type="button" className="graph-mode-btn" aria-haspopup="menu" aria-expanded={open}
        title={uiText(lang, "graphModesTitle")} onClick={() => setOpen((o) => !o)}>
        <Activity size={13} aria-hidden />
        <span>{active ? label(active) : uiText(lang, "graphModes")}</span>
        <ChevronDown size={12} className={open ? "graph-mode-chevron open" : "graph-mode-chevron"} aria-hidden />
      </button>
      {open ? (
        <div className="graph-mode-menu" role="menu">
          <button type="button" role="menuitem" className={active === null ? "graph-mode-item heading active" : "graph-mode-item heading"}
            onClick={() => pick(null)}>
            {uiText(lang, "graphModes")}
          </button>
          {modes.map((m) => (
            <button key={m} type="button" role="menuitemradio" aria-checked={shown === m}
              className={shown === m ? "graph-mode-item active" : "graph-mode-item"}
              onClick={() => pick(m === original ? null : m)}>
              {label(m)}
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
};
