/**
 * Colours the live dashboard-parity answers get: runs the app's own name -> dashboard key ->
 * colour mapping (src/components/charts/dashboardPalette.ts) over the charts saved by
 * backend/eval/question_bank/run_dashboard_charts.py, and lists every series (every slice for
 * donuts) as  name -> dashboard key -> colour. A name that found no dashboard key falls back to
 * the palette slot for its position, as on the dashboard; those are marked "(palette)" so a
 * missed alias stands out.
 *
 * Run with:  node scripts/verify-dashboard-colours.mjs
 */
import { execFileSync } from "node:child_process";
import { pathToFileURL } from "node:url";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const results = path.join("..", "backend", "eval", "question_bank", "results", "dashboard_charts.json");
const outDir = path.join(os.tmpdir(), "gmr-dashcolours");
fs.rmSync(outDir, { recursive: true, force: true });
execFileSync(process.execPath, [
  path.join("node_modules", "typescript", "bin", "tsc"), "--ignoreConfig", "src/components/charts/dashboardPalette.ts",
  "--outDir", outDir, "--module", "esnext", "--target", "es2022", "--skipLibCheck",
], { stdio: "pipe" });
const { canonicalKey, dashboardColour, KEY_COLORS } = await import(pathToFileURL(path.join(outDir, "dashboardPalette.js")).href);
const OVERRIDE_KEYS = new Set(["supplied", "installed", "utilization rate", "total supplies", "total installed", "remaining stock",
  "supplies", "agency", "supervisor", "technician", "installed, sat pending", "sat done, revenue pending"]);

let missed = 0;
for (const r of JSON.parse(fs.readFileSync(results, "utf8"))) {
  const c = r.chart ?? {};
  if (!c.dashboard) continue;
  // As chartOptions.colours(): slices by their exact name, one unnamed series is the card's `value`.
  const slices = c.type === "donut" || c.type === "pie";
  const names = slices ? c.x : [...(c.series ?? [{ name: "value" }]).map((s) => s.name), ...(c.rates ?? []).map((s) => s.name)];
  console.log(`\n${c.dashboard} / ${c.type} | ${r.q}`);
  names.forEach((name, i) => {
    const key = canonicalKey(c.dashboard, name, slices);
    const known = key in KEY_COLORS || OVERRIDE_KEYS.has(key);
    if (!known) missed += 1;
    console.log(`   ${name} -> ${key} -> ${dashboardColour(c.dashboard, name, i, slices)}${known ? "" : "  (palette)"}`);
  });
}
console.log(`\n${missed} name(s) coloured by position (no dashboard key)`);
