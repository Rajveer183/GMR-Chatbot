/**
 * The WFM dashboard's series colours, copied from its source (Documents/WFM/HES-FRONTEND-GMR,
 * branch vapt-rbac-remediation): src/utils/chartUtils.js (GMR, GMR_PALETTE, KEY_COLORS,
 * getColor, getContrastText) and the per-KPI overrides in src/components/charts/SmartChart.jsx.
 * KEY_COLORS repeats some keys there; the value JavaScript keeps (the last one) is the one here.
 *
 * The backend tags a chart with the dashboard KPI it answers (`chart.dashboard`,
 * backend/gmr_agents/dashboard_charts.py). Chatbot series are named by the planner
 * ("Dtr", "days_120_plus", "Total Sat"), so `canonicalKey` maps each name onto the
 * dashboard's own key for that KPI before the colour is looked up.
 */

export const GMR = {
  blue: "#2563eb",
  green: "#0d9488",
  orange: "#c2410c",
  purple: "#a21caf",
  pink: "#e11d48",
  yellow: "#d97706",
  red: "#ef4444",
};

export const GMR_PALETTE = [GMR.blue, GMR.green, GMR.orange, GMR.purple, GMR.pink, GMR.yellow, GMR.red];

const NAVY = "#0B3C6D";
const GOLD = "#FBBC04";
const RED = "#E31E24";

export const KEY_COLORS: Record<string, string> = {
  "total monthly installations": GMR.blue,
  "total installed": GMR.green,
  "cumulative sat": GMR.orange,
  critical: "#dc2626", major: "#f59e0b", minor: "#10b981",
  high: "#dc2626", medium: "#f59e0b", low: "#10b981",
  count: GMR.blue, value: GMR.blue, total: GMR.blue,
  consumer: NAVY, dt: GOLD, feeder: RED,
  "mi installed": GMR.orange,
  "sat done": GMR.green,
  invoiced: GMR.purple,
  burnt: RED, faulty: NAVY, others: GOLD,
  aew: NAVY, cap: RED, hpl: GMR.green, seipl: GOLD, unknown: GMR.pink,
  "daily installations": NAVY, "total installations": NAVY, installations: NAVY,
  "closed tickets": GMR.green, "open tickets": GMR.orange,
  closed: GMR.green, open: GMR.orange, "total open": GMR.orange,
  "auto ticketing": GMR.blue, "1912 helpdesk": GMR.yellow,
  pace: GMR.blue, revenue: GMR.purple,
  installed: GMR.green,
  remaining: "#94a3b8",
  supplies: GMR.blue, supplied: GMR.blue, "total supplies": GMR.blue, inventory: GMR.blue, "total inventory": GMR.blue,
  "remaining stock": GMR.orange,
  "utilization rate": GMR.purple,
  stock: GMR.blue,
  productivity: RED,
  agencies: GMR.red, teams: GMR.red,
  technicians: GOLD,
  "active days": NAVY, active_days: NAVY,
  "di to agency": "#199e70",
  "agency to supervisor": "#F2665D",
  "supervisor to technician": "#d55181",
  "technician to installation": "#3987E5",
  "installation to sat": RED,
  "sat to revenue": GOLD,
  "total journey": NAVY,
  "avg days": GMR.blue,
  realized: GMR.green,
  mi: GMR.blue, sat: GMR.green,
  lumpsum_invoice: GMR.purple, pmpm_invoice: GMR.pink,
  total_mi: GMR.blue, total_sat: GMR.green, total_lumpsum_invoice: GMR.purple, total_pmpm_invoice: GMR.pink,
  agra: GMR.blue, kashi: GMR.orange, triveni: GMR.purple,
  "0-30 days": "#3987E5", "31-60 days": GOLD, "61-90 days": NAVY, "90+ days": RED,
  "age 90+": RED, "age 0-30": "#F2665D", "age 31-60": "#3987E5", "age 61-90": NAVY, "age 91-120": GOLD, "age 120+": RED,
  "total mi": NAVY,
  "total mi (till cutoff date)": GMR.yellow,
  "installed, sat pending": GMR.orange,
  "sat done, revenue pending": GMR.purple,
  "total sat": RED,
  "sat progress %": GMR.purple,
  "sat progress % (till cutoff date)": GMR.pink,
  "% sat vs mi": "#c026d3",
  "% sat vs eligible mi": "#f43f5e",
  "eligible mi for sat": GOLD,
  "sat progress": GMR.purple,
  "mi progress": GMR.orange,
  "total lumpsum invoice": GMR.purple, "total pmpm invoice": GMR.pink,
  meters: "#3987E5",
  "certified (sat)": RED,
  "lumpsum invoiced": GMR.purple, "pmpm invoiced": GMR.pink,
  "% lumpsum of sat": "#0ea5e9", "% pmpm of sat": "#f43f5e",
  "1ph-consumer_meter": NAVY, "3ph-consumer_meter": GOLD, "ltct-consumer_meter": RED,
  "htct-consumer_meter": "#3987E5", "other-consumer_meter": "#F2665D",
};

// SmartChart.jsx per-KPI overrides, looked up before KEY_COLORS.
const SUPPLIES_UTIL_COLORS: Record<string, string> = { supplied: NAVY, installed: GOLD, "utilization rate": RED };
const MI_PACE_STOCK_COLORS: Record<string, string> = { "total supplies": NAVY, "total installed": GOLD, "remaining stock": RED };
const METER_STAGE_COLORS: Record<string, string> = {
  supplies: NAVY, agency: "#F2665D", supervisor: "#9085e9", technician: "#15803d",
  "installed, sat pending": GOLD, "sat done, revenue pending": RED,
};
// Technical (HES) boards -- HES-FRONTEND-GMR src/theme/dataviz.tokens.js LIGHT tokens via src/kpi/colour.js
// (statusColour / severityColour / dtBandColour) and DtLoadBoard.jsx (load status, rise bands). The backend
// (presentation/hes_charts.py) sends these keys as `point_keys` (per bar / slice) and `series_keys`.
const T = {
  bad: "#B91C1C", warn: "#D97706", good: "#15803D", grey: "#64748B", series: "#1A5FA8",
  ramp1: "#E07C40", ramp2: "#CE5220", ramp3: "#B91C1C",
  underload: "#E8734A", nearOptimal: "#F5A81C", approaching: "#F97316",
  band0: "#8FB8E2", band1: "#5590CB", band2: "#2A66AB", band3: "#0B3D7B", goodAlt: "#4E9E63",
};
// Layout-V voltage boards: statusColour, a flat tier map (Mild = Moderate = Elevated amber).
const PQ_STATUS: Record<string, string> = {
  poor: T.bad, severe: T.bad, unstable: T.bad, critical: T.bad,
  moderate: T.warn, elevated: T.warn, mild: T.warn,
  good: T.good, ok: T.good, stable: T.good,
  "1-phase": T.grey, "offline/noload": T.grey, "phase missing": T.grey, "no valid slot": T.grey, offline: T.grey,
  mean: T.grey,                                 // MeanBars: one neutral hue for every bar
  series: T.series, overload: T.bad, highlight: T.bad,
  dips: "#2a78d6", surges: "#e87ba4",           // DirectionCells (VFI)
  "under-voltage": T.bad, "over-voltage": T.warn,
};
// Layout A / E: severityColour, by rank in the KPI's own scale.
const PQ_SEVERITY: Record<string, string> = {
  ...PQ_STATUS,
  chronicviolation: T.ramp3, noncompliant: T.ramp2, borderline: T.ramp1, compliant: T.good,
  critical: T.ramp3, poor: T.ramp2, average: T.ramp1,
  "3-30 days": T.band0, "31-60 days": T.band1, "61-90 days": T.band2, "more than 90 days": T.band3,
};
const PCI_SEVERITY: Record<string, string> = { ...PQ_SEVERITY, noncompliant: T.ramp3, borderline: T.ramp2 };
// DT boards: loading bands by group, load status, load-change bands.
const DT_COLOURS: Record<string, string> = {
  "<20%": T.underload, "20-35%": T.underload, "35-55%": T.underload, "55-70%": T.nearOptimal, "70-85%": T.good,
  "85-100%": T.bad, "100-120%": T.bad, ">120%": T.bad,
  "overloaded >100%": T.bad, "overloaded 90-100%": T.approaching, normal: T.good, "underloaded <20%": T.underload,
  "fell >10 pp": T.band3, "fell 5-10 pp": T.band1, "stable (±5 pp)": T.goodAlt, "rose 5-10 pp": T.ramp1,
  "rose 10-20 pp": T.ramp2, "rose >20 pp": T.ramp3, "no previous month": T.grey,
  "last month": T.band0, "this month": T.series,
  series: T.series, overload: T.bad, highlight: T.bad, mean: T.grey,
};
const HES_COLOURS: Record<string, Record<string, string>> = {
  pq_vdi: PQ_STATUS, pq_vfi: PQ_STATUS, pq_vui: PQ_STATUS, pq_drop: PQ_STATUS, pq_cu: PQ_STATUS,
  pq_oob: PQ_SEVERITY, pq_pci: PCI_SEVERITY, lv_pockets: PQ_SEVERITY,
  dt_loading: DT_COLOURS, dt_peak: DT_COLOURS, dt_peak_feeder: DT_COLOURS, dt_kva: DT_COLOURS, dt_rise: DT_COLOURS,
  load_violation: { series: T.series, highlight: T.bad },
};

/** True for a Technical (HES) board tag: its keys are the backend's canonical names, used as sent. */
export function isHesBoard(kpi: string | undefined): boolean {
  return Boolean(kpi && HES_COLOURS[kpi]);
}

const KPI_OVERRIDES: Record<string, Record<string, string>> = {
  supply_utilization: SUPPLIES_UTIL_COLORS,
  mi_pace_stock: MI_PACE_STOCK_COLORS,
  meter_stage: METER_STAGE_COLORS,
  ...HES_COLOURS,
};

/** The dashboard card each `chart.dashboard` tag reproduces, by its name on the dashboard. */
export const DASHBOARD_CARDS: Record<string, string> = {
  mi_progress: "MI-Progress",
  mi_productivity: "MI-Productivity per team",
  monthly_productivity: "Monthly Productivity trend",
  meter_replacement: "Meter Replacement",
  supply_utilization: "Supply utilization rate",
  mi_pace_stock: "MI pace Vs Stock availability",
  stock_ageing: "Un-utilized stock ageing",
  mi_vs_sat: "MI Vs SAT-Progress",
  sat_invoice_funnel: "MI vs SAT vs Invoice Funnel",
  non_sat_ageing: "Non-SAT ageing",
  revenue_ageing: "Revenue not realized ageing",
  meter_journey: "Meters Journey Avg time",
  meter_stage: "Meters Current Stage",
  om_active_ageing: "O&M Active Ticket Ageing",
  om_escalated: "O&M Escalated to Discom",
  om_productivity: "O&M-Productivity per team",
  om_closed: "O&M Ticket Closed Analysis",
  hhu_reading: "HHU Reading Trend",
  pq_vdi: "Voltage Deviation Index",
  pq_vfi: "Voltage Fluctuation Index",
  pq_vui: "Voltage Unbalance Index",
  pq_drop: "Voltage Drop",
  pq_cu: "Meter Current Unbalance",
  pq_oob: "% Time beyond voltage tolerance band",
  pq_pci: "% Time with unacceptable current imbalance",
  lv_pockets: "Low-voltage pockets",
  dt_loading: "% DT Loading",
  dt_peak: "% DT Peak Loading",
  dt_peak_feeder: "% DT Peak Loading",
  dt_kva: "DT Load",
  dt_rise: "DT Load Rise",
  load_violation: "% Consumers with Load Violation",
};

/** chartUtils.js getColor: the key's fixed colour, else the palette slot for its position. */
export function getColor(key: string, index: number): string {
  const k = key.toLowerCase().trim();
  return KEY_COLORS[k] ?? GMR_PALETTE[index % GMR_PALETTE.length];
}

/** chartUtils.js getContrastText: white ink on dark fills, near-black on light ones. */
export function getContrastText(bgHex: string): string {
  if (!bgHex || bgHex[0] !== "#") return "#ffffff";
  const hex = bgHex.length === 4 ? bgHex.replace(/#(.)(.)(.)/, "#$1$1$2$2$3$3") : bgHex;
  const lin = (c: number) => (c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));
  const [r, g, b] = [1, 3, 5].map((i) => lin(parseInt(hex.slice(i, i + 2), 16) / 255));
  return 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.45 ? "#0f172a" : "#ffffff";
}

/** Age band in the dashboard's spelling for this KPI: '0-30 days' (stock), 'age 91-120' (non-SAT / revenue). */
function band(kpi: string, k: string): string | null {
  const range = k.match(/(\d+)\s*(?:-|to|\s)\s*(\d+)/);
  const plus = k.match(/(\d+)\s*(?:\+|plus|and above|above|or more)/) ?? k.match(/(?:>\s*=?|above|over|more than)\s*(\d+)/);
  if (!range && !plus) return null;
  const text = range ? `${range[1]}-${range[2]}` : `${plus![1]}+`;
  if (kpi === "non_sat_ageing" || kpi === "revenue_ageing") return `age ${text}`;
  return `${text} days`;
}

const has = (k: string, re: RegExp) => re.test(k);

const JOURNEY_STAGES: [RegExp, RegExp, string][] = [
  [/\bdi\b|inventory|warehouse/, /agency/, "di to agency"],
  [/agency/, /supervisor/, "agency to supervisor"],
  [/supervisor/, /technician/, "supervisor to technician"],
  [/technician/, /install/, "technician to installation"],
  [/install/, /\bsat\b/, "installation to sat"],
  [/\bsat\b/, /revenue|realiz/, "sat to revenue"],
];

/** The dashboard's key for a chatbot series / category name, for the given dashboard KPI.
 * `exact`: donut slices are coloured by their own name only, as the dashboard does -- seen
 * live, 'BYPASS BY CONSUMER' and 'CONSUMER REFUSAL' both took Consumer navy by partial match. */
export function canonicalKey(kpi: string, name: string, exact = false): string {
  const raw = name.toLowerCase().trim();
  // Technical boards: the backend already sends canonical keys ('>120%', 'Poor') -- the age-band
  // parsing below would turn '>120%' into '120+ days'.
  if (isHesBoard(kpi)) return raw;
  const k = raw.replace(/_/g, " ").replace(/\s+/g, " ");
  if (/(age|day|band|bucket|plus|\+|>)/.test(k) || /^\d+\s*-\s*\d+$/.test(k)) {
    const b = band(kpi, k);
    if (b) return b;
  }
  switch (kpi) {
    case "supply_utilization":
      if (has(k, /utili[sz]|rate|%/)) return "utilization rate";
      if (has(k, /install/)) return "installed";
      if (has(k, /suppl|inventory|received|stock/)) return "supplied";
      break;
    case "mi_pace_stock":
      if (has(k, /remaining|balance|available/)) return "remaining stock";
      if (has(k, /install/)) return "total installed";
      if (has(k, /suppl|inventory|received|stock/)) return "total supplies";
      break;
    case "mi_vs_sat":
      if (has(k, /%|pct|percent|progress/)) return has(k, /eligible/) ? "% sat vs eligible mi" : "% sat vs mi";
      if (has(k, /eligible/)) return "eligible mi for sat";
      if (has(k, /\bsat\b/)) return "total sat";
      if (has(k, /\bmi\b|install/)) return "total mi";
      break;
    case "sat_invoice_funnel":
      if (has(k, /lumpsum/)) return has(k, /%|pct/) ? "% lumpsum of sat" : "lumpsum invoiced";
      if (has(k, /pmpm/)) return has(k, /%|pct/) ? "% pmpm of sat" : "pmpm invoiced";
      if (has(k, /\bsat\b|certif/)) return "certified (sat)";
      if (has(k, /\bmi\b|install/)) return "installed";
      break;
    case "meter_replacement":
      if (has(k, /burn/)) return "burnt";
      if (has(k, /fault/)) return "faulty";
      if (has(k, /other/)) return "others";
      break;
    case "meter_journey":
      if (has(k, /total/)) return "total journey";
      for (const [a, b, key] of JOURNEY_STAGES) if (has(k, a) && has(k, b)) return key;
      break;
    case "meter_stage":
      // Stages named as journey legs ('DI to Agency') -- seen live -- keep the journey's distinct colours.
      if (has(k, / to /)) {
        for (const [a, b, key] of JOURNEY_STAGES) if (has(k, a) && has(k, b)) return key;
      }
      if (has(k, /revenue/)) return "sat done, revenue pending";
      if (has(k, /sat pending|install/)) return "installed, sat pending";
      if (has(k, /suppl|inventory|warehouse/)) return "supplies";
      if (has(k, /agency/)) return "agency";
      if (has(k, /supervisor/)) return "supervisor";
      if (has(k, /technician/)) return "technician";
      break;
  }
  if (KEY_COLORS[raw]) return raw;
  if (KEY_COLORS[k] || exact) return k;
  if (has(k, /^dtr?\b|\bdtr?$|\bdt meters?\b|transformer/)) return "dt";
  if (has(k, /consumer/)) return "consumer";
  if (has(k, /feeder/)) return "feeder";
  if (has(k, /productivity/)) return "productivity";
  if (has(k, /closed/)) return "closed tickets";
  if (has(k, /open/)) return "open tickets";
  if (has(k, /technician/)) return "technicians";
  if (has(k, /agenc/)) return "agencies";
  if (has(k, /\bteams?\b/)) return "teams";
  if (has(k, /active days/)) return "active days";
  if (has(k, /avg|average/) && has(k, /day/)) return "avg days";
  if (has(k, /install/)) return "installations";
  return k;
}

/** Colour of series / slice `name` at position `index` on the dashboard card for `kpi`. */
export function dashboardColour(kpi: string, name: string, index: number, exact = false): string {
  const key = canonicalKey(kpi, name, exact);
  return KPI_OVERRIDES[kpi]?.[key] ?? getColor(key, index);
}
