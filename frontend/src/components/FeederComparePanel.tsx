import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowRight, ArrowRightLeft, CircleCheck, CircleHelp, CircleSlash, Lightbulb, TriangleAlert, Zap, ZapOff,
} from "lucide-react";
import type { FeederCompare, FeederCompareFeeder, FeederCompareOutage, FeederCompareSide } from "../api/client";
import "../styles/feeder-compare.css";
import { uiParts, uiText } from "../i18n/languages";
import type { UIKey } from "../i18n/uiStrings";

/** One consumer vs every feeder of its substation (backend presentation/feeder_compare.py):
 *  verdict header + month switcher, current -> recommended feeder, a feeder leaderboard
 *  (outages the consumer went dark with vs stayed powered) and an outage timeline that lines
 *  the consumer's dark periods up against each feeder's outages. Every figure comes from the
 *  payload; the plain tables (and downloads) stay below the panel. */

const VERDICT: Record<string, { cls: string; label: UIKey; Icon: React.FC<{ size?: number }> }> = {
  CONFIRMED: { cls: "ok", label: "fc_confirmed", Icon: CircleCheck },
  REASSIGN: { cls: "move", label: "fc_reassign", Icon: ArrowRightLeft },
  REVIEW: { cls: "review", label: "fc_review", Icon: CircleHelp },
  "NO DATA": { cls: "none", label: "fc_noData", Icon: CircleSlash },
};

/** Screen language of the panel (the answer's language). */
type Lang = string | null | undefined;

/** The backend writes " -- " for a dash. */
const dash = (t: string) => t.replace(/ -- /g, " — ");

const pct = (dark: number, seen: number) => (seen ? Math.round((dark * 100) / seen) : null);

/* ---------------------------------------------------------------------------------------------- */

const Ring: React.FC<{ dark: number; seen: number; tone: "ok" | "move" | "muted"; lang: Lang }> = ({ dark, seen, tone, lang }) => {
  const r = 30, c = 2 * Math.PI * r;
  const p = pct(dark, seen);
  return (
    <svg className={`fcp-ring fcp-ring--${tone}`} width="76" height="76" viewBox="0 0 76 76" role="img"
      aria-label={seen ? uiText(lang, "fc_ringAria", { dark, seen }) : uiText(lang, "fc_noOutagesObserved")}>
      <circle cx="38" cy="38" r={r} className="fcp-ring-track" />
      {seen ? (
        <circle cx="38" cy="38" r={r} className="fcp-ring-value"
          strokeDasharray={`${(c * dark) / seen} ${c}`} transform="rotate(-90 38 38)" />
      ) : null}
      <text x="38" y="36" textAnchor="middle" className="fcp-ring-num">{seen ? `${p}%` : "–"}</text>
      <text x="38" y="51" textAnchor="middle" className="fcp-ring-sub">{seen ? uiText(lang, "fc_ofSeen", { dark, seen }) : uiText(lang, "fc_noOutages")}</text>
    </svg>
  );
};

const SideCard: React.FC<{ title: string; side: FeederCompareSide; tone: "ok" | "move" | "muted"; hint?: string; lang: Lang }> = ({ title, side, tone, hint, lang }) => (
  <div className={`fcp-side fcp-side--${tone}`}>
    <Ring dark={side.dark} seen={side.seen} tone={tone} lang={lang} />
    <div className="fcp-side-text">
      <span className="fcp-eyebrow">{title}</span>
      <strong className="fcp-side-name">{side.name || side.code}</strong>
      <span className="fcp-code">{side.code}</span>
      {hint ? <span className="fcp-side-hint">{hint}</span> : null}
    </div>
  </div>
);

/* ---------------------------------------------------------------------------------------------- */

const SplitBar: React.FC<{ f: FeederCompareFeeder; maxSeen: number; lang: Lang }> = ({ f, maxSeen, lang }) => {
  if (!f.seen) return <span className="fcp-bar-empty">{uiText(lang, "fc_noOutageToCompare")}</span>;
  const width = Math.max(14, (f.seen * 100) / Math.max(maxSeen, 1));
  return (
    <div className="fcp-bar" style={{ width: `${width}%` }}
      title={uiText(lang, "fc_barTitle", { dark: f.dark, powered: f.powered })}>
      {f.dark ? <span className="fcp-seg fcp-seg--dark" style={{ flexGrow: f.dark }}>{f.dark}</span> : null}
      {f.powered ? <span className="fcp-seg fcp-seg--powered" style={{ flexGrow: f.powered }}>{f.powered}</span> : null}
    </div>
  );
};

const Leaderboard: React.FC<{
  feeders: FeederCompareFeeder[]; focus: string | null; onFocus: (code: string | null) => void; lang: Lang;
}> = ({ feeders, focus, onFocus, lang }) => {
  const t = (key: UIKey, vars?: Record<string, string | number>) => uiText(lang, key, vars);
  const [order, setOrder] = useState<"rank" | "match">("rank");
  const scored = feeders.filter((f) => f.rank !== null);
  const unscored = feeders.filter((f) => f.rank === null);
  const rows = order === "rank" ? scored
    : [...scored].sort((a, b) => (b.match_pct ?? -1) - (a.match_pct ?? -1) || b.seen - a.seen);
  const maxSeen = Math.max(1, ...feeders.map((f) => f.seen));
  return (
    <div className="fcp-card">
      <div className="fcp-card-head">
        <div>
          <h5 className="fcp-card-title">{t("fc_everyFeeder")}</h5>
          <p className="fcp-card-sub">{t("fc_everyFeederSub")}</p>
        </div>
        <div className="fcp-toggle" role="group" aria-label={t("fc_orderBy")}>
          <button type="button" className={order === "rank" ? "is-on" : ""} onClick={() => setOrder("rank")}>{t("fc_auditRank")}</button>
          <button type="button" className={order === "match" ? "is-on" : ""} onClick={() => setOrder("match")}>{t("fc_matchPct")}</button>
        </div>
      </div>
      <div className="fcp-legend">
        <span><i className="fcp-swatch fcp-swatch--dark" />{t("fc_legendDark")}</span>
        <span><i className="fcp-swatch fcp-swatch--powered" />{t("fc_legendPowered")}</span>
      </div>
      <div className="fcp-rows" role="list">
        {rows.map((f, i) => (
          <button type="button" role="listitem" key={f.code}
            className={`fcp-row${focus === f.code ? " is-focus" : ""}${focus && focus !== f.code ? " is-dim" : ""}`}
            onClick={() => onFocus(focus === f.code ? null : f.code)}>
            <span className={`fcp-rank${(order === "rank" ? f.rank : i + 1) === 1 ? " is-top" : ""}`}>
              {order === "rank" ? f.rank : i + 1}
            </span>
            <span className="fcp-feeder">
              <span className="fcp-feeder-name">{f.name || f.code}</span>
              <span className="fcp-feeder-meta">
                <span className="fcp-code">{f.code}</span>
                {f.declared ? <span className="fcp-chip fcp-chip--current">{t("fc_currentTag")}</span> : null}
                {f.recommended && !f.declared ? <span className="fcp-chip fcp-chip--rec">{t("fc_recommended")}</span> : null}
                {f.best && !f.recommended && !f.declared ? <span className="fcp-chip">{t("fc_strongest")}</span> : null}
                {f.other_substation ? <span className="fcp-chip fcp-chip--other">{f.other_substation}</span> : null}
              </span>
            </span>
            <span className="fcp-bar-cell"><SplitBar f={f} maxSeen={maxSeen} lang={lang} /></span>
            <span className="fcp-match">
              <strong>{f.match_pct === null ? "–" : `${Math.round(f.match_pct)}%`}</strong>
              <small>{t("fc_match")}</small>
            </span>
            <span className={`fcp-net ${f.net && f.net > 0 ? "is-pos" : f.net && f.net < 0 ? "is-neg" : ""}`}
              title={t("fc_netTitle")}>
              {f.net === null ? "–" : f.net > 0 ? `+${f.net}` : f.net}
            </span>
          </button>
        ))}
        {unscored.map((f) => (
          <div role="listitem" key={f.code} className="fcp-row is-unscored">
            <span className="fcp-rank">–</span>
            <span className="fcp-feeder">
              <span className="fcp-feeder-name">{f.name || f.code}</span>
              <span className="fcp-feeder-meta">
                <span className="fcp-code">{f.code}</span>
                {f.declared ? <span className="fcp-chip fcp-chip--current">{t("fc_currentTag")}</span> : null}
              </span>
            </span>
            <span className="fcp-bar-cell"><SplitBar f={f} maxSeen={maxSeen} lang={lang} /></span>
            <span className="fcp-match"><strong>–</strong><small>{t("fc_match")}</small></span>
            <span className="fcp-net">–</span>
          </div>
        ))}
      </div>
    </div>
  );
};

/* ---------------------------------------------------------------------------------------------- */

function useWidth<T extends HTMLElement>(): [React.RefObject<T | null>, number] {
  const ref = useRef<T>(null);
  const [w, setW] = useState(800);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    setW(Math.max(320, Math.round(el.getBoundingClientRect().width)));
    if (typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(320, Math.round(e.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, w];
}

const HOUR = 3600_000;
const ms = (iso: string | null) => (iso ? new Date(iso).getTime() : NaN);   // naive IST -> same offset for all
const dayLabel = (t: number) => new Date(t).toLocaleDateString("en-GB", { day: "2-digit", month: "short" });
const hourLabel = (t: number) => new Date(t).toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" });

type Tip = { x: number; y: number; title: string; lines: string[] };

const Timeline: React.FC<{
  outages: FeederCompareOutage[]; feeders: FeederCompareFeeder[]; consumerDark?: { start: string; end: string }[];
  focus: string | null; onFocus: (c: string | null) => void; lang: Lang;
}> = ({ outages, feeders, consumerDark, focus, onFocus, lang }) => {
  const t = (key: UIKey, vars?: Record<string, string | number>) => uiText(lang, key, vars);
  const [wrapRef, width] = useWidth<HTMLDivElement>();
  const [tip, setTip] = useState<Tip | null>(null);
  const valid = outages.filter((o) => Number.isFinite(ms(o.start)) && Number.isFinite(ms(o.end)));
  const lanes = useMemo(() => {
    const has = new Set(valid.map((o) => o.feeder));
    return feeders.filter((f) => has.has(f.code));
  }, [valid, feeders]);
  if (!valid.length) return null;

  // The axis runs on clock hours (the times are IST wall-clock, parsed and printed in the same zone).
  const floorTo = (t: number, h: number) => { const d = new Date(t); d.setMinutes(0, 0, 0); d.setHours(Math.floor(d.getHours() / h) * h); return d.getTime(); };
  const tMin = Math.min(...valid.map((o) => ms(o.start)));
  const tMax = Math.max(...valid.map((o) => ms(o.end)));
  const labelW = Math.min(170, Math.round(width * 0.28));
  const plotW = width - labelW - 12;
  // Ticks every 3/6/12/24/48 h so that about one label per 90px fits.
  const spanH = (tMax - tMin) / HOUR + 3;
  const step = [3, 6, 12, 24, 48, 96].find((s) => spanH / s <= Math.max(4, plotW / 90)) ?? 168;
  const t0 = floorTo(tMin, Math.min(step, 24));
  let t1 = t0;
  while (t1 < tMax) t1 += step * HOUR;
  const laneH = 30, top = 30, consumerGap = 10;
  const maxChars = Math.max(8, Math.floor((labelW - 20) / 7.4));   // ~7.4px a character at 12px
  const x = (t: number) => labelW + ((t - t0) / Math.max(t1 - t0, HOUR)) * plotW;
  const height = top + laneH * (lanes.length + 1) + consumerGap + 8;
  const ticks: number[] = [];
  for (let t = t0; t <= t1; t += step * HOUR) ticks.push(t);

  // The consumer lane: the half-hours it had no supply (backend), else the outages it went dark in.
  const merged: [number, number][] = consumerDark?.length
    ? consumerDark.map((w) => [ms(w.start), ms(w.end)] as [number, number]).filter(([a, b]) => Number.isFinite(a) && Number.isFinite(b))
    : (() => {
        const out: [number, number][] = [];
        for (const iv of valid.filter((o) => o.dark).map((o) => [ms(o.start), ms(o.end)] as [number, number]).sort((a, b) => a[0] - b[0])) {
          const last = out[out.length - 1];
          if (last && iv[0] <= last[1]) last[1] = Math.max(last[1], iv[1]);
          else out.push([...iv]);
        }
        return out;
      })();
  const exact = Boolean(consumerDark?.length);
  // The part of an outage [a, b) during which the consumer was dark.
  const overlap = (a: number, b: number) =>
    merged.map(([c, d]) => [Math.max(a, c), Math.min(b, d)] as [number, number]).filter(([p, q]) => q > p);
  const minutes = (a: number, b: number) => Math.round((b - a) / 60000);

  const show = (e: React.MouseEvent, title: string, lines: string[]) => {
    const box = wrapRef.current?.getBoundingClientRect();
    if (box) setTip({ x: e.clientX - box.left, y: e.clientY - box.top, title, lines });
  };

  return (
    <div className="fcp-card">
      <div className="fcp-card-head">
        <div>
          <h5 className="fcp-card-title">{t("fc_timeline")}</h5>
          <p className="fcp-card-sub">{t("fc_timelineSub")}</p>
        </div>
      </div>
      <div className="fcp-legend">
        <span><i className="fcp-swatch fcp-swatch--consumer" />{t("fc_lgConsumer")}</span>
        <span><i className="fcp-swatch fcp-swatch--dark" />{t("fc_lgDark")}</span>
        <span><i className="fcp-swatch fcp-swatch--frame" />{t("fc_lgFrame")}</span>
        <span><i className="fcp-swatch fcp-swatch--hollow" />{t("fc_lgHollow")}</span>
      </div>
      <div className="fcp-timeline" ref={wrapRef} onMouseLeave={() => setTip(null)}>
        <svg width={width} height={height} style={{ display: "block" }} role="img" aria-label={t("fc_timelineAria")}>
          {ticks.map((t) => (
            <g key={t}>
              <line x1={x(t)} x2={x(t)} y1={top - 6} y2={height - 6} className={new Date(t).getHours() === 0 ? "fcp-grid fcp-grid--day" : "fcp-grid"} />
              <text x={x(t)} y={top - 12} className="fcp-tick"
                textAnchor={x(t) > width - 34 ? "end" : x(t) < labelW + 20 ? "start" : "middle"}>
                {new Date(t).getHours() === 0 || step >= 24 ? dayLabel(t) : hourLabel(t)}
              </text>
            </g>
          ))}
          {/* Consumer lane */}
          <text x={8} y={top + laneH / 2 + 4} className="fcp-lane-label fcp-lane-label--consumer">{t("fc_thisConsumer")}</text>
          <rect x={labelW} y={top + 4} width={plotW} height={laneH - 8} rx="4" className="fcp-lane-bg" />
          {merged.map(([a, b]) => (
            <rect key={`c${a}`} x={x(a)} y={top + 5} width={Math.max(3, x(b) - x(a))} height={laneH - 10} rx="4"
              className="fcp-block fcp-block--consumer"
              onMouseMove={(e) => show(e, t("fc_consumerDark"), [`${dayLabel(a)} ${hourLabel(a)} – ${dayLabel(b)} ${hourLabel(b)}`])} />
          ))}
          {lanes.map((f, i) => {
            const y = top + laneH * (i + 1) + consumerGap;
            const dim = focus && focus !== f.code;
            return (
              <g key={f.code} className={`fcp-lane${dim ? " is-dim" : ""}${focus === f.code ? " is-focus" : ""}`}
                onClick={() => onFocus(focus === f.code ? null : f.code)}>
                <rect x={0} y={y} width={width} height={laneH} className="fcp-lane-hit" />
                <text x={8} y={y + laneH / 2 + 4} className="fcp-lane-label">
                  {(f.name || f.code).length > maxChars ? `${(f.name || f.code).slice(0, maxChars - 1)}…` : f.name || f.code}
                  {f.declared ? " •" : ""}
                </text>
                <rect x={labelW} y={y + 4} width={plotW} height={laneH - 8} rx="4" className="fcp-lane-bg" />
                {valid.filter((o) => o.feeder === f.code).map((o) => {
                  const a = ms(o.start), b = ms(o.end);
                  const parts = o.dark && exact ? overlap(a, b) : o.dark ? [[a, b] as [number, number]] : [];
                  const darkMin = parts.reduce((s, [p, q]) => s + minutes(p, q), 0);
                  const tipLines = [
                    `${o.start_label} – ${o.end_label} (${t("fc_minutes", { n: o.half_hours * 30 })})`,
                    o.dark ? (exact ? t("fc_tipDarkFor", { n: darkMin }) : t("fc_tipDark")) : t("fc_tipPowered")];
                  const bx = x(a) + 1, bw = Math.max(4, x(b) - x(a) - 2);
                  return (
                    <g key={`${o.feeder}-${o.no}`} onMouseMove={(e) => show(e, t("fc_outageNo", { name: f.name || f.code, no: o.no }), tipLines)}>
                      <rect x={bx} y={y + 6} width={bw} height={laneH - 12} rx="4"
                        className={`fcp-block ${o.dark ? "fcp-block--dark-frame" : "fcp-block--powered"}`} />
                      {parts.map(([p, q]) => (
                        <rect key={p} x={Math.max(bx, x(p))} y={y + 6}
                          width={Math.max(3, Math.min(bx + bw, x(q)) - Math.max(bx, x(p)))} height={laneH - 12} rx="3"
                          className="fcp-block fcp-block--dark" />
                      ))}
                    </g>
                  );
                })}
              </g>
            );
          })}
        </svg>
        {tip ? (
          <div className="fcp-tip" style={{ left: Math.min(tip.x + 12, width - 230), top: tip.y + 12 }}>
            <strong>{tip.title}</strong>
            {tip.lines.map((l) => <span key={l}>{l}</span>)}
          </div>
        ) : null}
      </div>
      <p className="fcp-foot">{t("fc_footDot")}</p>
    </div>
  );
};

/* ---------------------------------------------------------------------------------------------- */

/** "<strong>3</strong> of 12 feeders compared" from uiParts(). */
const Bold: React.FC<{ parts: [string, string, string] }> = ({ parts: [before, bold, after] }) => (
  <span>{before}<strong>{bold}</strong>{after}</span>
);

export const FeederComparePanel: React.FC<{
  data: FeederCompare; onAsk?: (question: string) => void; lang?: Lang;
}> = ({ data, onAsk, lang }) => {
  const t = (key: UIKey, vars?: Record<string, string | number>) => uiText(lang, key, vars);
  const [focus, setFocus] = useState<string | null>(null);
  const v = VERDICT[data.verdict] ?? VERDICT["NO DATA"];
  const place = [data.category, data.sub_station, data.zone ? t("fc_zone", { zone: data.zone }) : null].filter(Boolean).join(" · ");
  const sameFeeder = data.declared && data.recommended && data.declared.code === data.recommended.code;

  return (
    <section className={`fcp fcp--${v.cls}`} aria-label={t("fc_mappingAria", { meter: data.meter })}>
      <header className="fcp-hero">
        <div className="fcp-hero-main">
          <span className="fcp-eyebrow fcp-eyebrow--hero">{t("fc_eyebrow", { month: data.month_label })}</span>
          <h4 className="fcp-meter">{data.meter}</h4>
          <span className="fcp-place">{place}</span>
        </div>
        <div className={`fcp-verdict fcp-verdict--${v.cls}`}>
          <v.Icon size={22} />
          <div>
            <strong>{t(v.label)}</strong>
            <span>{dash(data.verdict_text)}</span>
          </div>
        </div>
        {data.months.length > 1 ? (
          <div className="fcp-months" role="group" aria-label={t("fc_auditMonth")}>
            {data.months.map((m) => {
              const mv = VERDICT[m.verdict] ?? VERDICT["NO DATA"];
              const on = m.month === data.month;
              return (
                <button type="button" key={m.month} className={`fcp-month${on ? " is-on" : ""}`} disabled={on || !onAsk}
                  onClick={() => onAsk?.(`mapping of meter ${data.meter} in ${m.label}`)}
                  title={on ? t("fc_showingMonth") : t("fc_showMonth", { month: m.label })}>
                  <i className={`fcp-dot fcp-dot--${mv.cls}`} />{m.label}<small>{t(mv.label)}</small>
                </button>
              );
            })}
          </div>
        ) : null}
      </header>

      <div className="fcp-flow">
        {data.declared ? (
          <SideCard title={sameFeeder ? t("fc_taggedConfirmed") : t("fc_currentTag")} side={data.declared}
            tone={data.verdict === "CONFIRMED" ? "ok" : "muted"} lang={lang} />
        ) : <div className="fcp-side fcp-side--muted"><div className="fcp-side-text"><span className="fcp-eyebrow">{t("fc_currentTag")}</span><strong>{t("fc_noFeeder")}</strong></div></div>}
        {data.recommended && !sameFeeder ? (
          <>
            <div className="fcp-arrow" aria-hidden><ArrowRight size={22} /><span>{t("fc_retag")}</span></div>
            <SideCard title={t("fc_recommendedAuto")} side={data.recommended} tone="move" lang={lang} />
          </>
        ) : null}
        {!data.recommended && data.strongest ? (
          <>
            <div className="fcp-arrow fcp-arrow--soft" aria-hidden><ArrowRight size={22} /><span>{t("fc_maybe")}</span></div>
            <SideCard title={t("fc_strongest")} side={data.strongest} tone="muted" hint={t("fc_notDecisive")} lang={lang} />
          </>
        ) : null}
        <div className="fcp-facts">
          <div><Zap size={16} /><Bold parts={uiParts(lang, "fc_feedersCompared", "n", { n: data.feeders.filter((f) => f.rank !== null).length, total: data.feeders_in_substation })} /></div>
          {data.outages.length ? (
            <div><ZapOff size={16} /><Bold parts={uiParts(lang, "fc_outagesDark", "n", { n: data.outages.filter((o) => o.dark).length, total: data.outages.length })} /></div>
          ) : null}
          {data.action ? <div className="fcp-action"><Lightbulb size={16} /><span>{dash(data.action)}</span></div> : null}
        </div>
      </div>

      {data.tied || data.notes.length ? (
        <div className={`fcp-notes${data.tied ? " is-warn" : ""}`}>
          <TriangleAlert size={16} />
          {data.notes.length === 1 ? <p className="fcp-note-one">{dash(data.notes[0].replace(/^(Note|Caution):\s*/, ""))}</p>
            : <ul>{data.notes.map((n) => <li key={n}>{dash(n.replace(/^(Note|Caution):\s*/, ""))}</li>)}</ul>}
        </div>
      ) : null}

      {data.feeders.length ? <Leaderboard feeders={data.feeders} focus={focus} onFocus={setFocus} lang={lang} /> : null}
      {data.outages.length ? (
        <Timeline outages={data.outages} feeders={data.feeders} consumerDark={data.consumer_dark} focus={focus} onFocus={setFocus} lang={lang} />
      ) : null}
    </section>
  );
};
