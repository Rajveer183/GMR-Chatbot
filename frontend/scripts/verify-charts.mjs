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
      "src/components/chartOptions.ts", "src/components/charts/dataviz.ts",
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
      if (layout.kind === "bar" && !layout.funnel && (chart.series?.length ?? 1) === 1 && options.colors?.[0] !== DV.series) {
        problems.push(`single-series bar must use the dashboard series hue ${DV.series}`);
      }
      if (chart.series?.length > 1 && isOrderedBands(chart.series.map((s) => s.name)) && !options.colors.every((c) => DV.bandRamp.includes(c))) {
        problems.push("ordered bands must use the band ramp");
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
