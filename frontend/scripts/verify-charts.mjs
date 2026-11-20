/**
 * Regression checks on the ApexCharts options this app builds
 * (src/components/chartOptions.ts, styled after the WFM/HES dashboard).
 *
 * WHAT THIS COVERS, and why it exists: the bug it guards against is invisible
 * to TypeScript and to `vite build`. ApexCharts merges our options over its
 * own defaults and then reads several paths unconditionally --
 * `config.markers.size`, `config.xaxis.title.text`, `config.yaxis[0].title.text`,
 * `config.chart.events`. A key that is *present with the value `undefined`*
 * REPLACES the library's default rather than leaving it in place, so it becomes
 * a hard TypeError at render time ("Cannot read properties of undefined
 * (reading 'size')" shipped exactly like that while every static check passed).
 *
 * It also checks: values reach the chart unmodified, a unit is displayed only
 * when the backend detected one, combo lines sit on the second axis, every
 * chart type the backend can send is handled, and the dashboard number formats.
 *
 * WHAT THIS DOES NOT COVER: painting (svg.js needs real SVG layout) -- check
 * that visually by asking a chart question in the running app.
 *
 * Run with:  npm run verify:charts
 */
import { execFileSync } from "node:child_process";
import { pathToFileURL } from "node:url";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const outDir = path.join(os.tmpdir(), "gmr-chartcheck");
fs.rmSync(outDir, { recursive: true, force: true });
try {
  execFileSync(
    process.execPath,
    [
      path.join("node_modules", "typescript", "bin", "tsc"), "--ignoreConfig",
      "src/components/chartOptions.ts", "src/components/charts/dataviz.ts", "src/components/charts/graphModes.ts", "src/components/charts/hoverCard.ts",
      "--outDir", outDir, "--module", "esnext", "--moduleResolution", "bundler",
      "--target", "es2022", "--skipLibCheck",
    ],
    { cwd: process.cwd(), stdio: "pipe" }
  );
} catch {
  // tsc may exit non-zero over the type-only api/client import; a missing emit is caught below.
}
for (const file of fs.globSync("**/*.js", { cwd: outDir })) {
  const full = path.join(outDir, file);
  fs.writeFileSync(
    full,
    fs.readFileSync(full, "utf8").replace(/(from\s+")(\.[^"]*?)(")/g, (_m, a, spec, c) =>
      path.extname(spec) ? `${a}${spec}${c}` : `${a}${spec}.js${c}`
    )
  );
}
const optsJs = fs.globSync("**/chartOptions.js", { cwd: outDir })[0];
const vizJs = fs.globSync("**/dataviz.js", { cwd: outDir })[0];
if (!optsJs || !vizJs) {
  console.error(`chart modules did not compile into ${outDir}`);
  process.exit(1);
}
const { resolveKind, buildChartOptions, buildChartSeries } = await import(pathToFileURL(path.join(outDir, optsJs)).href);
const { fullNumber, axisNumber, isOrderedBands, DV } = await import(pathToFileURL(path.join(outDir, vizJs)).href);
const modesJs = fs.globSync("**/graphModes.js", { cwd: outDir })[0];
const { availableModes, applyMode, chronological } = await import(pathToFileURL(path.join(outDir, modesJs)).href);
const hoverJs = fs.globSync("**/hoverCard.js", { cwd: outDir })[0];
const { hoverCard, hoverHtml, weekRange } = await import(pathToFileURL(path.join(outDir, hoverJs)).href);

const CASES = [
  { name: "bar, SAT stages", chart: { id: "c", title: "SAT by stage", type: "bar",
      x: ["SAT-1", "SAT-2", "SAT-3"], y: [498911, 243645, 266735] } },
  { name: "horizontal_bar, 12 long names", chart: { id: "c", title: "Divisions", type: "horizontal_bar",
      x: Array.from({ length: 12 }, (_, i) => `EDD DIVISION WITH A LONG NAME ${i}`), y: Array.from({ length: 12 }, (_, i) => i * 1000) } },
  { name: "horizontal_bar, detected unit", chart: { id: "c", title: "Dues", type: "horizontal_bar", x: ["A", "B"], y: [3, 9], unit: "lakh" } },
  { name: "grouped_bar", chart: { id: "c", title: "MI vs SAT", type: "grouped_bar", x: ["AGRA", "KASHI"], y: [1, 2],
      series: [{ name: "MI", data: [1636209, 1352702] }, { name: "SAT", data: [1457191, 1190471] }] } },
  { name: "stacked_bar, age bands", chart: { id: "c", title: "Ageing", type: "stacked_bar", x: ["AGRA", "KASHI"], y: [0, 0],
      series: [{ name: "91-120 days", data: [5424, 2201] }, { name: "120+ days", data: [106588, 78834] }] } },
  { name: "stacked_horizontal_bar", chart: { id: "c", title: "Stage", type: "stacked_horizontal_bar", x: ["Agra", "Kashi"], y: [0, 0],
      series: [{ name: "A", data: [1, 2] }, { name: "B", data: [3, 4] }] } },
  { name: "percent_stacked_bar", chart: { id: "c", title: "Mix", type: "percent_stacked_bar", x: ["AGRA"], y: [0],
      series: [{ name: "Consumer", data: [95] }, { name: "DTR", data: [5] }] } },
  { name: "line, multi-series", chart: { id: "c", title: "Supply vs installs", type: "line", x: ["Jan", "Feb", "Mar"], y: [1, 5, 2],
      series: [{ name: "Supplied", data: [1, 5, 2] }, { name: "Installed", data: [2, 4, 3] }] } },
  { name: "area", chart: { id: "c", title: "Installs", type: "area", x: ["Jan", "Feb", "Mar"], y: [321188, 320929, 406285] } },
  { name: "combo, bars + % line", chart: { id: "c", title: "SAT progress", type: "combo", x: ["AGRA", "KASHI"], y: [1, 2], unit2: "%",
      series: [{ name: "Total MI", data: [1636209, 1352702], type: "bar" }, { name: "SAT %", data: [89.06, 88.01], type: "line" }] } },
  { name: "donut", chart: { id: "c", title: "Status", type: "donut", x: ["Confirmed", "Reassign", "Review", "No data"],
      y: [1727536, 732376, 1190464, 396625] } },
  { name: "pie", chart: { id: "c", title: "Defects", type: "pie", x: ["Faulty", "Burnt", "Others"], y: [37870, 11350, 2205] } },
  { name: "funnel", chart: { id: "c", title: "Funnel", type: "funnel", x: ["Supplied", "Installed", "SAT"], y: [4372802, 4266091, 3754574] } },
  { name: "radial", chart: { id: "c", title: "Utilization", type: "radial", x: ["Utilization"], y: [97.56], unit: "%" } },
  { name: "heatmap", chart: { id: "c", title: "Weekday", type: "heatmap", x: ["Mon", "Tue"], y: [1, 2],
      series: [{ name: "agra", data: [1, 2] }, { name: "kashi", data: [3, 4] }] } },
  { name: "single category", chart: { id: "c", title: "One", type: "bar", x: ["Only"], y: [42] } },
  { name: "zero / negative values", chart: { id: "c", title: "Net", type: "bar", x: ["A", "B", "C"], y: [0, -12.5, 8] } },
  { name: "unknown type -> bar", chart: { id: "c", title: "?", type: "mystery", x: ["A"], y: [1] } },
];

function undefinedPaths(node, prefix = "") {
  const out = [];
  if (node === null || typeof node !== "object" || typeof node === "function") return out;
  const entries = Array.isArray(node) ? node.map((v, i) => [`[${i}]`, v]) : Object.entries(node);
  for (const [key, value] of entries) {
    const at = Array.isArray(node) ? `${prefix}${key}` : prefix ? `${prefix}.${key}` : key;
    if (value === undefined) out.push(at);
    else if (typeof value === "object") out.push(...undefinedPaths(value, at));
  }
  return out;
}

const UNGUARDED = [
  { path: ["markers", "size"], required: true },
  { path: ["xaxis", "title", "text"], required: false },
  { path: ["yaxis", "title", "text"], required: false },
  { path: ["chart", "events"], required: false },
];

function checkUnguarded(options, problems) {
  for (const { path: keys, required } of UNGUARDED) {
    let node = options;
    for (let i = 0; i < keys.length; i += 1) {
      const present = node !== null && typeof node === "object" && keys[i] in node;
      if (!present) {
        if (i === 0 && required) problems.push(`"${keys.join(".")}" is missing (must always be set)`);
        break;
      }
      node = node[keys[i]];
      if (node === undefined) {
        problems.push(`"${keys.slice(0, i + 1).join(".")}" is present but undefined`);
        break;
      }
    }
  }
}

let failures = 0;
for (const { name, chart } of CASES) {
  for (const clickable of [false, true]) {
    const label = `${name}${clickable ? " [drilldown]" : ""}`;
    const problems = [];
    try {
      const layout = resolveKind(chart.type);
      const options = buildChartOptions(chart, layout, clickable ? () => {} : undefined);
      const series = buildChartSeries(chart, layout);
      const flatKinds = ["pie", "donut", "radialBar"];

      for (const at of undefinedPaths(options)) problems.push(`option "${at}" is present but undefined`);
      checkUnguarded(options, problems);

      const hasEvents = typeof options.chart?.events?.dataPointSelection === "function";
      if (hasEvents !== clickable) problems.push(`drilldown handler ${hasEvents ? "set" : "absent"} but ${clickable ? "was" : "was not"} supplied`);

      // Values reach the chart unmodified.
      const sent = chart.series?.length ? chart.series.map((s) => s.data) : [chart.y];
      let got;
      if (flatKinds.includes(layout.kind)) got = [series];
      else if (layout.kind === "heatmap") got = series.map((s) => s.data.map((c) => c.y));
      else got = series.map((s) => s.data);
      const want = layout.funnel ? sent.slice(0, 1) : sent;
      if (JSON.stringify(got) !== JSON.stringify(want)) problems.push(`series data altered: ${JSON.stringify(got)} vs ${JSON.stringify(want)}`);

      // Combo: exactly the line series ride the opposite axis.
      if (layout.combo) {
        const lines = series.filter((s) => s.type === "line").map((s) => s.name);
        if (JSON.stringify(lines) !== JSON.stringify(["SAT %"])) problems.push(`combo line series wrong: ${lines}`);
        if (!Array.isArray(options.yaxis) || !options.yaxis.some((a) => a.opposite)) problems.push("combo has no second (opposite) axis");
      }

      // Units only when the backend detected one.
      const valueText = layout.kind === "radialBar"
        ? options.plotOptions?.radialBar?.dataLabels?.value?.formatter?.(1234.5)
        : options.tooltip?.y?.formatter?.(1234.5, { seriesIndex: 0 });
      if (typeof valueText !== "string") problems.push("value formatter did not return a string");
      else if (layout.kind === "radialBar") {
        if (!valueText.endsWith("%")) problems.push("radial value must show the % as sent");
      } else if (chart.unit) {
        if (!valueText.endsWith(chart.unit)) problems.push(`tooltip ${JSON.stringify(valueText)} dropped unit ${chart.unit}`);
      } else if (/[A-Za-z]/.test(valueText)) {
        problems.push(`tooltip ${JSON.stringify(valueText)} invented a unit`);
      }

      // Category names never dropped: tooltip shows the full name on vertical category axes.
      if (!flatKinds.includes(layout.kind) && layout.kind !== "heatmap" && !layout.horizontal) {
        const full = options.tooltip?.x?.formatter?.("", { dataPointIndex: 0 });
        if (full !== String(chart.x[0])) problems.push(`tooltip x label ${JSON.stringify(full)} != ${JSON.stringify(chart.x[0])}`);
      }

      // Dashboard colour rules.
      // Portal colours everywhere (chartUtils.js getColor): one unnamed series is the portal's 'value' blue.
      if (layout.kind === "bar" && !layout.funnel && !chart.series?.length && options.colors?.[0] !== "#2563eb") {
        problems.push("single-series bar must use the portal's value blue #2563eb");
      }
      const PORTAL_BANDS = ["#3987E5", "#FBBC04", "#0B3C6D", "#E31E24", "#F2665D"];
      if (chart.series?.length > 1 && isOrderedBands(chart.series.map((s) => s.name)) && !options.colors.every((c) => PORTAL_BANDS.includes(c))) {
        problems.push("ordered bands must use the portal's age-band colours");
      }
    } catch (err) {
      problems.push(`threw: ${err?.message ?? err}`);
    }
    failures += problems.length ? 1 : 0;
    console.log(`  ${problems.length ? "FAIL" : "ok  "}  ${label}`);
    for (const p of problems) console.log(`          - ${p}`);
  }
}

// Dashboard cards (chart.dashboard): the WFM dashboard's colour per series name and its
// stacking / label rules (SmartChart.jsx, chartUtils.js KEY_COLORS).
const NAVY = "#0B3C6D", GOLD = "#FBBC04", RED = "#E31E24";
const DASHBOARD = [
  { name: "MI progress by project: Consumer / DT / Feeder stack, total on top",
    chart: { id: "c", title: "MI", type: "stacked_bar", dashboard: "mi_progress", labels: "total", x: ["AGRA", "KASHI", "TRIVENI"], y: [0],
      series: [{ name: "Consumer", data: [1, 2, 3] }, { name: "Dtr", data: [1, 1, 1] }, { name: "feeder", data: [0, 1, 0] }] },
    colors: [NAVY, GOLD, RED], stacked: true, total: true },
  { name: "Non-SAT ageing by project: age band colours",
    chart: { id: "c", title: "Non-SAT", type: "stacked_bar", dashboard: "non_sat_ageing", labels: "segments", x: ["AGRA", "KASHI"], y: [0],
      series: ["days_0_30", "31-60 days", "61 to 90 days", "91-120", "days_120_plus"].map((n) => ({ name: n, data: [5, 6] })) },
    colors: ["#F2665D", "#3987E5", NAVY, GOLD, RED], stacked: true, total: false },
  { name: "Stock ageing by warehouse: stock band colours, no labels",
    chart: { id: "c", title: "Stock", type: "stacked_bar", dashboard: "stock_ageing", labels: "none", x: ["W1", "W2"], y: [0],
      series: ["0-30 days", "31-60 days", "61-90 days", "90+ days"].map((n) => ({ name: n, data: [5, 6] })) },
    colors: ["#3987E5", GOLD, NAVY, RED], stacked: true, total: false, labelsOff: true },
  { name: "MI vs SAT by project: grouped, % over Total MI",
    chart: { id: "c", title: "MI vs SAT", type: "grouped_bar", dashboard: "mi_vs_sat", labels: "segments", x: ["AGRA", "KASHI"], y: [0],
      series: [{ name: "Total MI", data: [100, 200] }, { name: "Eligible MI", data: [90, 150] }, { name: "Total SAT", data: [80, 120] }],
      rates: [{ name: "SAT % vs MI", data: [80, 60] }] },
    colors: [NAVY, GOLD, RED], stacked: false, rateOn: [0, "80.0%"] },
  { name: "Supply utilization: Supplied / Installed side by side, rate on the last bar",
    chart: { id: "c", title: "Supply", type: "grouped_bar", dashboard: "supply_utilization", labels: "segments", x: ["AGRA"], y: [0],
      series: [{ name: "Supplied", data: [100] }, { name: "Installed", data: [97] }], rates: [{ name: "Utilization Rate", data: [97] }] },
    colors: [NAVY, GOLD], stacked: false, rateOn: [1, "97.0%"] },
  { name: "Meter replacement by OEM: Burnt / Faulty / Others",
    chart: { id: "c", title: "Defects", type: "stacked_bar", dashboard: "meter_replacement", labels: "total", x: ["HPL", "AEW"], y: [0],
      series: [{ name: "Meter Burnt", data: [1, 2] }, { name: "Meter Faulty", data: [3, 4] }, { name: "Others", data: [1, 1] }] },
    colors: [RED, NAVY, GOLD], stacked: true, total: true },
  { name: "Meter journey stages",
    chart: { id: "c", title: "Journey", type: "stacked_bar", dashboard: "meter_journey", labels: "segments", x: ["AGRA"], y: [0],
      series: ["Inventory To Agency", "Agency To Supervisor", "Installation To Sat", "Sat To Revenue Realized"].map((n) => ({ name: n, data: [3] })) },
    colors: ["#199e70", "#F2665D", RED, GOLD], stacked: true, total: false },
  { name: "MI productivity: one bar, productivity above it",
    chart: { id: "c", title: "Productivity", type: "bar", dashboard: "mi_productivity", labels: "none", x: ["AGRA", "KASHI"], y: [500, 400],
      rates: [{ name: "Productivity", data: [3.2, 2.9] }] },
    stacked: true, total: true, labelsOff: true },
  { name: "O&M open tickets by complaint type: donut, % inside",
    chart: { id: "c", title: "Open", type: "donut", dashboard: "om_active_ageing", labels: "segments", x: ["No display", "Others (3 types)"], y: [90, 10] } },
  { name: "Installs over 40 days: area, productivity on a right axis",
    chart: { id: "c", title: "Trend", type: "area", dashboard: "mi_productivity", labels: "none",
      x: Array.from({ length: 40 }, (_, i) => `2026-09-${String((i % 30) + 1).padStart(2, "0")}`), y: Array.from({ length: 40 }, () => 5),
      series: [{ name: "Installations", data: Array.from({ length: 40 }, () => 5) }, { name: "Agencies", data: Array.from({ length: 40 }, () => 2) }],
      rates: [{ name: "Productivity", data: Array.from({ length: 40 }, () => 3) }] },
    colors: [NAVY, "#ef4444", RED] },
];

for (const { name, chart, colors, stacked, total, labelsOff, rateOn } of DASHBOARD) {
  const problems = [];
  try {
    const layout = resolveKind(chart.type);
    const options = buildChartOptions(chart, layout);
    const series = buildChartSeries(chart, layout);
    for (const at of undefinedPaths(options)) problems.push(`option "${at}" is present but undefined`);
    checkUnguarded(options, problems);
    if (colors && JSON.stringify(options.colors.slice(0, colors.length)) !== JSON.stringify(colors)) {
      problems.push(`colours ${JSON.stringify(options.colors)} != dashboard ${JSON.stringify(colors)}`);
    }
    if (stacked !== undefined && Boolean(options.chart?.stacked) !== stacked) problems.push(`stacked=${options.chart?.stacked}, want ${stacked}`);
    const totalOn = Boolean(options.plotOptions?.bar?.dataLabels?.total?.enabled);
    if (total !== undefined && totalOn !== total) problems.push(`stack total label ${totalOn ? "on" : "off"}, want ${total ? "on" : "off"}`);
    if (labelsOff && options.dataLabels?.enabled) problems.push("value labels must be off on this card");
    if (rateOn && options.dataLabels.formatter(1, { seriesIndex: rateOn[0], dataPointIndex: 0 }) !== rateOn[1]) {
      problems.push(`rate label on series ${rateOn[0]}: ${options.dataLabels.formatter(1, { seriesIndex: rateOn[0], dataPointIndex: 0 })}`);
    }
    if (chart.type === "bar" && chart.rates && options.plotOptions.bar.dataLabels.total.formatter("500", { dataPointIndex: 0 }) !== "3.2") {
      problems.push("productivity must sit on top of the bar");
    }
    if (chart.type === "donut") {
      if (options.dataLabels.formatter(3) !== "" || options.dataLabels.formatter(42.4) !== "42%") problems.push("donut slices must show their % (5% or more)");
      if (options.legend.position !== "bottom") problems.push("donut legend goes under the ring");
    }
    if (chart.type === "area") {
      if (!series.some((s) => s.type === "line" && s.name === "Productivity")) problems.push("rate must be a line series");
      if (!Array.isArray(options.yaxis) || !options.yaxis.some((a) => a.opposite)) problems.push("rate needs a right-hand axis");
    }
  } catch (err) {
    problems.push(`threw: ${err?.message ?? err}`);
  }
  failures += problems.length ? 1 : 0;
  console.log(`  ${problems.length ? "FAIL" : "ok  "}  dashboard: ${name}`);
  for (const p of problems) console.log(`          - ${p}`);
}

// Graphs Modes (charts/graphModes.ts): every mode a chart offers builds clean options and shows the
// same figures -- bars / lines / areas the same series, a donut each category's total, a pareto the
// same bars largest first.
const total = (c) => (c.series?.length ? c.series.filter((s) => s.type !== "line" || c.type !== "combo") : [{ data: c.y }])
  .reduce((t, s) => t + s.data.reduce((a, v) => a + Number(v || 0), 0), 0);
const MODE_CHARTS = [...CASES.map((c) => c.chart), ...DASHBOARD.map((d) => d.chart),
  { id: "m1", title: "MI progress trend", type: "stacked_bar", dashboard: "mi_progress", labels: "total",
    x: ["2026-11-08", "2026-11-09", "2026-11-10"], y: [1, 2, 3],
    series: [{ name: "Consumer", data: [3700, 2400, 5100] }, { name: "DT", data: [40, 60, 30] }, { name: "Feeder", data: [0, 1, 0] }] },
  { id: "m2", title: "Comparison by Project", type: "stacked_bar", dashboard: "mi_progress", labels: "total",
    x: ["kashi", "agra", "triveni"], y: [1, 2, 3],
    series: [{ name: "Consumer", data: [7000, 6000, 4900] }, { name: "DT", data: [200, 150, 86] }] },
  { id: "m3", title: "Supplied vs installed", type: "grouped_bar", dashboard: "supply_utilization",
    x: ["AGRA", "KASHI"], y: [1, 2], series: [{ name: "Supplied", data: [10, 20] }, { name: "Installed", data: [8, 15] }] },
  { id: "m4", title: "Tickets by type", type: "bar", x: ["A", "B", "C", "D"], y: [40, 25, 10, 5] },
  { id: "m5", title: "VDI by zone", type: "horizontal_bar", dashboard: "pq_vdi", x: ["Z1", "Z2", "Z3"], y: [3.1, 2.2, 1.5],
    point_keys: ["mean", "mean", "mean"] },
];
let modeRuns = 0;
for (const chart of MODE_CHARTS) {
  const modes = availableModes(chart);
  for (const mode of modes) {
    const problems = [];
    try {
      const shown = applyMode(chart, mode);
      modeRuns += 1;
      for (const [k, v] of Object.entries(shown)) if (v === undefined) problems.push(`spec key "${k}" is undefined`);
      const layout = resolveKind(shown.type);
      const options = buildChartOptions(shown, layout);
      buildChartSeries(shown, layout);
      for (const at of undefinedPaths(options)) problems.push(`option "${at}" is present but undefined`);
      checkUnguarded(options, problems);
      const want = total(chart);
      const got = total(shown);
      if (Math.abs(got - want) > 1e-6 * Math.max(1, Math.abs(want))) problems.push(`figures changed: ${got} vs ${want}`);
      if ((mode === "line" || mode === "area") && shown.point_keys) problems.push("a line keeps no per-bar colours");
    } catch (err) {
      problems.push(`threw: ${err?.message ?? err}`);
    }
    failures += problems.length ? 1 : 0;
    if (problems.length) console.log(`  FAIL  mode ${mode}: ${chart.title}`);
    for (const p of problems) console.log(`          - ${p}`);
  }
}
const expectModes = [
  ["m1", ["area", "line", "bar"]], ["m2", ["bar", "hbar", "donut"]], ["m3", ["bar", "hbar"]],
  ["m4", ["line", "area", "bar", "hbar", "donut", "funnel", "pareto"]],
];
for (const [id, want] of expectModes) {
  const got = availableModes(MODE_CHARTS.find((c) => c.id === id));
  const ok = JSON.stringify(got) === JSON.stringify(["area", "line", "bar", "hbar", "donut", "gauge", "funnel", "pareto"].filter((m) => want.includes(m)));
  failures += ok ? 0 : 1;
  console.log(`  ${ok ? "ok  " : "FAIL"}  modes offered for ${id}: ${JSON.stringify(got)}`);
}
console.log(`  ok    ${modeRuns} graph-mode renders checked`);

// Hover card (charts/hoverCard.ts): the dashboard's CustomTooltip -- DT / Consumer / Feeder first, a Total only
// for parts of one whole (kept when a stacked chart is switched to lines), rates as extra rows, donut share.
{
  const mi = MODE_CHARTS.find((c) => c.id === "m1");
  const card = (chart, j = 0) => hoverCard(chart, resolveKind(chart.type), j, ["#1", "#2", "#3", "#4"]);
  const HOVER = [
    ["order DT, Consumer, Feeder", () => card(mi).rows.slice(0, 3).map((r) => r.name).join(","), "DT,Consumer,Feeder"],
    ["Total = sum of the parts", () => card(mi).rows.find((r) => r.name === "Total")?.value, "3,740"],
    ["Total kept after Line mode", () => card(applyMode(mi, "line")).rows.some((r) => r.name === "Total"), true],
    ["no Total beside Supplied", () => card(MODE_CHARTS.find((c) => c.id === "m3")).rows.some((r) => r.name === "Total"), false],
    ["donut share", () => card({ id: "d", title: "t", type: "donut", x: ["A", "B"], y: [30, 10] }).rows.find((r) => r.name === "Share")?.value, "75.0%"],
    ["rate row", () => card({ id: "r", title: "t", type: "bar", x: ["A"], y: [5], rates: [{ name: "% SAT vs MI", data: [89.06] }] }).rows.at(-1)?.value, "89.06%"],
    ["ISO week range", () => weekRange("2026-W45"), "02 Nov - 08 Nov 2026"],
    ["escapes names", () => hoverHtml({ title: "<b>x</b>", rows: [] }).includes("<b>x"), false],
    ["place rows under 'By Project'", () => card({ ...mi, details: [[["kashi", 2000], ["agra", 1740]], [], []], details_label: "Project" })
      .rows.slice(-3).map((r) => `${r.name}=${r.value}`).join(","), "By Project=,kashi=2,000,agra=1,740"],
    ["newest-first dates drawn oldest first", () => JSON.stringify(chronological({ id: "c", title: "t", type: "bar",
      x: ["2026-11-17", "2026-11-16"], y: [3, 2], series: [{ name: "agra", data: [3, 2] }], details: [[["a", 3]], [["a", 2]]] })),
      JSON.stringify({ id: "c", title: "t", type: "bar", x: ["2026-11-16", "2026-11-17"], y: [2, 3],
        series: [{ name: "agra", data: [2, 3] }], details: [[["a", 2]], [["a", 3]]] })],
    ["pareto keeps place rows aligned", () => applyMode({ id: "p", title: "t", type: "bar", x: ["A", "B"], y: [1, 9],
      details: [[["a", 1]], [["b", 9]]] }, "pareto").details[0][0][0], "b"],
  ];
  for (const [name, run, want] of HOVER) {
    let got;
    try { got = run(); } catch (err) { got = `threw: ${err?.message ?? err}`; }
    const ok = JSON.stringify(got) === JSON.stringify(want);
    failures += ok ? 0 : 1;
    console.log(`  ${ok ? "ok  " : "FAIL"}  hover: ${name}${ok ? "" : ` got ${JSON.stringify(got)}`}`);
  }
}

// Portal colours exactly as the dashboard draws them (20 Nov, user: no chatbot-only rules): a named key keeps its
// colour, anything else takes the 7 portal slots by position, repeating; two series on one key share its colour.
{
  const PORTAL = ["#2563eb", "#0d9488", "#c2410c", "#a21caf", "#e11d48", "#d97706", "#ef4444"];
  const ZONES = ["VARANASI II", "VARANASI I", "AZAMGARH", "PRAYAGRAJ II", "ETAH", "FIROZABAD", "ALIGARH", "PRAYAGRAJ I", "MIRZAPUR", "AGRA", "MATHURA"];
  const cols = (chart) => buildChartOptions(chart, resolveKind(chart.type)).colors;
  const zoneChart = (order) => ({ id: "z", title: "t", type: "stacked_bar", dashboard: "om_closed", x: ["2026-11-11", "2026-11-12"], y: [1, 1],
    series: order.map((n) => ({ name: n, data: [1, 2] })) });
  const a = cols(zoneChart(ZONES));
  const COLOUR = [
    ["zones take the 7 portal slots by position (Agra keeps blue)", () => a.slice(0, ZONES.length),
      ZONES.map((z, i) => (z === "AGRA" ? "#2563eb" : PORTAL[i % 7]))],
    // 3-part stacked bars: the dashboard's brand trio navy / gold / red (20 Nov, user).
    ...[["mi_progress", ["AGRA", "KASHI", "TRIVENI"], ["#0B3C6D", "#FBBC04", "#E31E24"]],
        ["mi_vs_sat", ["SAT-11", "SAT-12", "SAT-13"], ["#0B3C6D", "#FBBC04", "#E31E24"]],
        [undefined, ["Feeder", "DTR", "Consumer"], ["#E31E24", "#FBBC04", "#0B3C6D"]],
        [undefined, ["Others", "Meter Burnt", "Meter Faulty"], ["#FBBC04", "#E31E24", "#0B3C6D"]],
        ["hhu_reading", ["Upload OK", "Upload Failed", "Upload Other"], ["#2563eb", "#0d9488", "#c2410c"]],
        ["stock_ageing", ["0-30 days", "31-60 days", "61-90 days"], ["#3987E5", "#FBBC04", "#0B3C6D"]]]
      .map(([kpi, names, want]) => [`3-part stack ${names.join(" / ")}`, () => cols({ id: "t", title: "t", type: "stacked_bar", dashboard: kpi,
        x: ["2026-11-10"], y: [1], series: names.map((n) => ({ name: n, data: [1] })) }).slice(0, 3), want]),
    ["Triveni is the portal's fuchsia", () => cols({ id: "p", title: "t", type: "bar", x: ["x"], y: [1],
      series: [{ name: "Agra", data: [1] }, { name: "Kashi", data: [2] }, { name: "Triveni", data: [3] }] }).slice(0, 3),
      ["#2563eb", "#c2410c", "#a21caf"]],
    ["SAT-11 / Total SAT / SAT-12 all take Total SAT red, as on the dashboard", () => cols({ id: "s", title: "t", type: "grouped_bar", dashboard: "mi_vs_sat", x: ["AGRA"], y: [1],
      series: [{ name: "SAT-11", data: [1] }, { name: "Total SAT", data: [2] }, { name: "SAT-12", data: [3] }] }).slice(0, 3), ["#E31E24", "#E31E24", "#E31E24"]],
    ["Agra stays blue beside Total", () => cols({ id: "a", title: "t", type: "bar", x: ["x"], y: [1],
      series: [{ name: "Total", data: [1] }, { name: "AGRA", data: [2] }] })[1], "#2563eb"],
  ];
  for (const [name, run, want] of COLOUR) {
    let got; try { got = run(); } catch (err) { got = `threw: ${err?.message ?? err}`; }
    const ok = JSON.stringify(got) === JSON.stringify(want);
    failures += ok ? 0 : 1;
    console.log(`  ${ok ? "ok  " : "FAIL"}  colour: ${name}${ok ? "" : ` got ${JSON.stringify(got)}`}`);
  }
}

const FORMATS = [
  [fullNumber(4214678), "42,14,678"], [fullNumber(97.56, "%"), "97.56%"], [fullNumber(110493, "meters"), "1,10,493 meters"],
  [axisNumber(4214678), "42.15L"], [axisNumber(43728020), "4.4Cr"], [axisNumber(9876), "9,876"], [axisNumber(88.01, "%"), "88.01%"],
];
for (const [got, want] of FORMATS) {
  const ok = got === want;
  failures += ok ? 0 : 1;
  console.log(`  ${ok ? "ok  " : "FAIL"}  format ${JSON.stringify(want)}${ok ? "" : ` got ${JSON.stringify(got)}`}`);
}

console.log(failures ? `\n${failures} check(s) failed.` : "\nall chart checks passed.");
process.exit(failures ? 1 : 0);
