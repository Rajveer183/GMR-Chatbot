import React, { useState } from "react";
import { ChevronDown, FlaskConical, History, RefreshCw } from "lucide-react";
import { apiClient } from "../api/client";
import type { Investigation, RelatedInvestigation } from "../api/client";

/** 'Why did X change?' answers: the explanations the backend tested on the rows
 * (presentation/investigate.py -- computed by code, not written by the LLM), the confidence,
 * and earlier investigations of a similar question with a read-only "re-check now". */

const STATUS_CLASS: Record<string, string> = {
  SUPPORTED: "ok",
  "NOT SUPPORTED": "no",
  PARTLY: "partly",
  CHECK: "partly",
  "NOT TESTABLE": "no",
};

const fmt = (v?: number | null) =>
  v === undefined || v === null ? "" : Number.isInteger(v) ? v.toLocaleString("en-IN") : v.toLocaleString("en-IN", { maximumFractionDigits: 2 });

function leadText(lead: Investigation["lead"] | undefined | null): string {
  if (!lead) return "";
  const share = lead.share !== undefined ? `, ${Math.round(lead.share)}% of the change` : "";
  const move = lead.prev !== undefined && lead.curr !== undefined ? ` (${fmt(lead.prev)} → ${fmt(lead.curr)}${share})` : "";
  return `${lead.by} = ${lead.name}${move}`;
}

const RelatedItem: React.FC<{ item: RelatedInvestigation }> = ({ item }) => {
  const [state, setState] = useState<{ busy: boolean; text?: string }>({ busy: false });
  const recheck = async () => {
    setState({ busy: true });
    try {
      const got = await apiClient.recheckInvestigation(item.id);
      setState({ busy: false, text: got.ok ? got.summary : `Could not re-check: ${got.error ?? "unknown error"}` });
    } catch (err) {
      setState({ busy: false, text: `Could not re-check: ${err instanceof Error ? err.message : String(err)}` });
    }
  };
  return (
    <li className="inv-related-item">
      <div>
        <span className="inv-related-when">{item.at?.slice(0, 10)}</span> “{item.question}”
        {item.lead ? <span className="inv-related-lead"> — lead: {leadText(item.lead)}</span> : null}
        {item.confidence ? <span className={`inv-conf inv-conf-${item.confidence}`}>{item.confidence}</span> : null}
      </div>
      {item.action ? <div className="inv-related-action">Recommended: {item.action}</div> : null}
      <button type="button" className="inv-recheck-btn" onClick={recheck} disabled={state.busy}
              title="Re-run that investigation's query now and compare (no AI call)">
        <RefreshCw size={12} className={state.busy ? "spin" : ""} /> {state.busy ? "Checking…" : "Has it changed since?"}
      </button>
      {state.text ? <div className="inv-recheck-result">{state.text}</div> : null}
    </li>
  );
};

export const InvestigationPanel: React.FC<{
  investigation?: Investigation | null;
  related?: RelatedInvestigation[];
}> = ({ investigation, related }) => {
  const [open, setOpen] = useState(false);
  const rel = related ?? [];
  if (!investigation && !rel.length) return null;
  const conf = investigation?.confidence;
  return (
    <section className="chat-result-section inv-panel">
      {investigation ? (
        <>
          <button type="button" className="inv-toggle" onClick={() => setOpen((o) => !o)} aria-expanded={open}>
            <FlaskConical size={14} aria-hidden="true" />
            <strong>How this was tested</strong>
            {conf ? <span className={`inv-conf inv-conf-${conf.level}`}>confidence: {conf.level}</span> : null}
            {investigation.lead ? <span className="inv-lead">lead: {leadText(investigation.lead)}</span> : null}
            <ChevronDown size={14} className={`inv-chevron ${open ? "expanded" : ""}`} />
          </button>
          {open ? (
            <div className="inv-body">
              {investigation.premise ? <p className="inv-note warn">{investigation.premise.replace(/^PREMISE CHECK:\s*/, "Premise check: ").replace(/ Correct the premise.*$/, "")}</p> : null}
              {investigation.coverage_gap ? (
                <p className="inv-note warn">The rows cover one period only, so the change could not be measured.</p>
              ) : null}
              {investigation.dimensions.map((d, i) => (
                <div key={`${d.by}-${i}`} className="inv-dim">
                  <div className="inv-dim-head">
                    By <strong>{d.by}</strong>: {d.measure}, {d.periods?.[0]} → {d.periods?.[1]}
                    {!d.rate && d.total_prev !== undefined ? (
                      <> — {fmt(d.total_prev)} → {fmt(d.total_curr)}{d.pct !== undefined && d.pct !== null ? ` (${d.pct > 0 ? "+" : ""}${d.pct.toFixed(1)}%)` : ""}</>
                    ) : null}
                  </div>
                  <ul className="inv-hyps">
                    {d.hypotheses.map((h, j) => (
                      <li key={j}>
                        <span className={`inv-status inv-status-${STATUS_CLASS[h.status] ?? "no"}`}>{h.status.toLowerCase()}</span>
                        <span className="inv-h">{h.h}</span>
                        <span className="inv-ev">{h.evidence}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
              {investigation.contradictions.map((c, i) => (
                <p key={i} className="inv-note warn">Contradiction: {c.replace(/ -- quote the.*$/, "")}</p>
              ))}
              {conf?.reasons?.length ? <p className="inv-note">Why {conf.level}: {conf.reasons.join("; ")}.</p> : null}
              <p className="inv-foot">Computed by code from the query results — the AI writes the answer from these figures and does not invent causes.</p>
            </div>
          ) : null}
        </>
      ) : null}
      {rel.length ? (
        <div className="inv-related">
          <div className="inv-related-title"><History size={13} aria-hidden="true" /> Earlier investigations of a similar question</div>
          <ul>{rel.map((r) => <RelatedItem key={r.id} item={r} />)}</ul>
        </div>
      ) : null}
    </section>
  );
};
