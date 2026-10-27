"""The WFM dashboard's own chart for a question one of its KPI cards answers.

First priority for every chart: when the main SQL reads a KPI's gold table, the answer is
drawn the way that card draws it (Documents/WFM/HES-FRONTEND-GMR, branch
vapt-rbac-remediation: src/components/charts/SmartChart.jsx + src/utils/chartUtils.js
detectVariant). Anything else -- silver tables, the headline snapshot, a chart type the user
named -- keeps the planner's chart (llm_pipeline._chart_fallback).

The dashboard's rules, as its code applies them:
  * every bar chart is ONE stack (Recharts stackId "a"): Consumer / DT / Feeder, age bands,
    Burnt / Faulty / Others, journey stages ... one stacked column per category
  * more than 8 categories (not dates) -> horizontal bars; dates -> columns, area past 25 points
  * MI-productivity / monthly productivity stay vertical; pace-vs-stock / O&M productivity
    are horizontal whenever the axis is not dates
  * O&M open tickets by complaint type and escalations by remark are donuts (top 9 + Others)
  * a rate (Productivity, Utilization Rate, % SAT vs MI) is not a bar: its value sits above
    the bar (left as `rates`, drawn by the frontend)
  * labels: inside each segment, or only the stack total on top (MI-progress by project,
    Non-SAT ageing, MI vs SAT stages, Meter Replacement), or none (stock ageing by
    warehouse, productivity KPIs)
  * colours are fixed per series name (frontend/src/components/charts/dashboardPalette.ts)

One deliberate difference: where the dashboard stacks a total on top of its own parts
(Supplied + Installed, Total MI + Eligible MI + Total SAT at circle level, MI + SAT +
invoiced), the bar would show a sum that counts every meter twice -- those series are drawn
side by side instead, as the dashboard itself does for MI vs SAT at DISCOM / zone level.
"""
from __future__ import annotations

import re
from typing import Any

import pandas as pd

# Gold table in the main SQL -> dashboard KPI card. The first table named wins.
_TABLE_KPI: list[tuple[re.Pattern[str], str]] = [(re.compile(p, re.I), k) for p, k in [
    (r"gold_mi_installation_(?:day|month)_agg", "mi_progress"),
    (r"gold_mi_technician_productivity_(?:day|month)_agg", "mi_productivity"),
    (r"gold_mi_productivity_trend_month_agg", "monthly_productivity"),
    (r"gold_meter_defect_event_day_agg", "meter_replacement"),
    (r"gold_meter_inventory_event_(?:day|month)_agg|gold_inventory_monthly_balance", "supply_utilization"),
    (r"gold_inventory_unutilized_stock_snapshot", "stock_ageing"),
    (r"gold_mi_sat_invoice_(?:day_agg|snapshot)", "mi_vs_sat"),
    (r"gold_non_sat_ageing_snapshot", "non_sat_ageing"),
    (r"gold_meter_journey_stage_agg", "meter_journey"),
    (r"gold_meter_(?:installation|revenue)_pending_stage_snapshot", "meter_stage"),
    (r"gold_om_open_(?:ageing|ticket_detail)_snapshot", "om_active_ageing"),
    (r"gold_om_discom_escalation_snapshot", "om_escalated"),
    (r"gold_om_technician_productivity_(?:day|month)_agg", "om_productivity"),
    (r"gold_om_closed_tickets_day_agg", "om_closed"),
    (r"gold_om_hhu_reading_day_agg", "hhu_reading"),
    (r"gold_revenue_ageing_snapshot", "revenue_ageing"),
]]

# 'Show it as a pie' / 'line chart of ...': the user's own chart type is kept.
_NAMED_TYPE_RE = re.compile(
    r"\b(?:pie|donut|doughnut|funnel|heat ?map|gauge|radial|combo|dual[- ]axis)\b"
    r"|\b(?:line|bar|column|area|stacked|grouped|horizontal|vertical)\s+(?:chart|graph|plot|diagram)s?\b"
    r"|\bas (?:an? )?(?:line|bars?|columns?|area)\b", re.I)

# Seen live: 'Installs/Day (6M Avg)' was stacked on Remaining Stock in MI pace vs stock.
_RATE_RE = re.compile(r"%|\bpct\b|percent|\brate\b|productivity|per (?:team|technician|agency|day)|/\s*(?:day|team|tech)",
                      re.I)
# A total already contains the other series; stacking it on them counts every meter twice.
# Seen live: 'Escalated Tickets' (every escalation) was stacked on its Discom pending / approved parts.
_WHOLE_RE = re.compile(r"\b(?:total|suppl(?:y|ied|ies)|inventory|received|eligible)\b|^escalated tickets$", re.I)
# Geography / organisation only -- seen live: x = 'technician_remark' counted as a place, and the
# escalations-by-remark donut became bars.
_GEO_RE = re.compile(r"project|discom|zone|circle|division|cluster|warehouse|\boem\b|vendor|\bsku\b|region", re.I)
# A chart's own columns when the planner drew none (auto_spec): the dashboard card's series.
_BAND_COL_RE = r"days?_\d|age_\d|\d+_(?:\d+|plus)\b|\d+\+"
_CARD_COLUMNS = {
    "mi_progress": r"consumer|\bdtr?\b|^dt_|feeder",
    "meter_replacement": r"burnt|faulty|other",
    "hhu_reading": r"^ok|_ok\b|fail|other",
    "non_sat_ageing": _BAND_COL_RE, "revenue_ageing": _BAND_COL_RE,
    "stock_ageing": _BAND_COL_RE, "om_active_ageing": _BAND_COL_RE,
}
_NOT_A_MEASURE_RE = re.compile(r"^(?:ord|rank|sort|seq|row_?num|year|yr)$|_id$|^id$", re.I)
_NO_LABEL = {"total", "grand total", "none", "nan", "", "all"}
_PROJECTS = {"agra", "kashi", "triveni", "puvvnl", "dvvnl"}
# chartUtils.js isDate: ISO / DD-MM-YYYY dates, YYYY-MM, YYYY-Www, 'Oct 2026' / 'Oct-26', week labels.
_DATE_RE = re.compile(r"^(?:\d{4}-\d{2}-\d{2}|\d{2}-\d{2}-\d{4}|\d{2}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{2}/\d{2}/\d{2}|"
                      r"\d{4}-\d{2}|\d{4}-w\d{1,2}|[a-z]{3}[-\s]\d{2,4})$|^week|^w\d+-", re.I)

DONUT_SLICES = 9           # smartMeterUtils.js OM_TYPE_SLICES: the rest is one 'Others (N types)' slice
_TOTAL_LABELS = {"mi_progress_geo", "non_sat_ageing_stage", "mi_vs_sat_stage", "meter_replacement"}
_NO_LABELS = {"stock_ageing_geo", "mi_productivity", "monthly_productivity", "om_productivity"}


def user_named_type(question: str | None) -> bool:
    return bool(question and _NAMED_TYPE_RE.search(question))


def kpi_for(sql: str | None, names: list[str], question: str | None = None) -> str | None:
    """The dashboard card whose gold table the main query reads (first one named), refined by
    the question and the series it returns: supply asked as pace / stock availability (or with
    remaining stock, unless utilization was asked) is 'MI pace vs stock', MI/SAT with invoices is
    the invoice funnel. Seen live: 'Supply utilization rate by project' also returned remaining
    stock and was drawn as the pace-vs-stock card's horizontal bars."""
    if not sql:
        return None
    found = sorted((m.start(), kpi) for pat, kpi in _TABLE_KPI for m in [pat.search(sql)] if m)
    if not found:
        return None
    kpi = found[0][1]
    text = " ".join(names).lower()
    asked = (question or "").lower()
    if kpi == "supply_utilization" and (re.search(r"\bpace\b|stock availab", asked)
                                        or ("remaining" in text and "utili" not in asked)):
        return "mi_pace_stock"
    if kpi == "mi_vs_sat" and re.search(r"invoice|lumpsum|pmpm|collect", text):
        return "sat_invoice_funnel"
    return kpi


def _label_cols(df: Any) -> list[str]:
    return [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]


def split_column(df: Any, x: str, ys: list[str]) -> str | None:
    """Long rows (project, age_key, meters) with no `group`: the second label column that varies
    within each x value -- one series per value. Seen live: non-SAT ageing by project in age
    bands drew nine single bars 'AGRA · age_0_30', ... instead of three stacked columns."""
    for c in _label_cols(df):
        if c == x or c in ys or df[c].astype(str).str.match(r"^\d{4}-\d{2}").any():
            continue
        if df.groupby(df[x].astype(str))[c].nunique().max() > 1:
            return c
    return None


def card_measure(kpi: str, df: Any, ys: list[str]) -> list[str]:
    """A productivity card plots productivity: when the rows carry a productivity column the planner
    left out of its chart, that column is the bar. Seen live: 'MI productivity per team by project'
    stacked installations on active technicians while productivity sat only in the table."""
    if kpi not in ("mi_productivity", "monthly_productivity", "om_productivity"):
        return ys
    if any(re.search(r"productivity", str(y), re.I) for y in ys):
        return ys
    prod = [c for c in df.columns if re.search(r"productivity", str(c), re.I) and pd.api.types.is_numeric_dtype(df[c])]
    return prod[:1] or ys


def auto_spec(kpi: str, df: Any) -> dict | None:
    """The card's chart when the planner drew none: x = the first label column, y = the card's own
    series (Consumer / DT / Feeder, age bands, Burnt / Faulty / Others, OK / Failed / Other), else
    every measure. Two or more rows only -- a one-row answer stays as cards. Seen live: revenue
    not realized by age band, O&M ageing by project and HHU OK vs failed came back as tables."""
    labels = _label_cols(df)
    nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not _NOT_A_MEASURE_RE.search(str(c))]
    if not labels or not nums:
        return None
    x = labels[0]
    if (~df[x].astype(str).str.strip().str.lower().isin(_NO_LABEL)).sum() < 2:
        return None
    pattern = _CARD_COLUMNS.get(kpi)
    ys = [c for c in nums if pattern and re.search(pattern, str(c), re.I)]
    if len(ys) < 2:
        ys = nums
    group = split_column(df, x, ys) if len(ys) == 1 else None
    return {"type": "bar", "x": x, "y": ys, **({"group": group} if group else {})}


def _is_dates(cats: list[str]) -> bool:
    return bool(cats) and all(_DATE_RE.search(str(c).strip()) for c in cats)


def _is_geo(x_alias: Any, cats: list[str]) -> bool:
    return bool(_GEO_RE.search(str(x_alias or ""))) or any(str(c).strip().lower() in _PROJECTS for c in cats)


def _donut(cats: list[str], data: list[float]) -> tuple[list[str], list[float]]:
    """Largest slices first, the 10th and later merged into one 'Others (N types)' slice."""
    pairs = sorted(zip(cats, data), key=lambda p: -p[1])
    if len(pairs) <= DONUT_SLICES + 1:
        return [p[0] for p in pairs], [p[1] for p in pairs]
    head, rest = pairs[:DONUT_SLICES], pairs[DONUT_SLICES:]
    return ([p[0] for p in head] + [f"Others ({len(rest)} types)"],
            [p[1] for p in head] + [sum(p[1] for p in rest)])


def _merge_repeats(cats: list[str], series: list[dict]) -> list[str]:
    """One bar per category. Seen live: MI progress by project came back as a consumer row, a DT row,
    a feeder row and a subtotal row per project -- AGRA drawn four times. Per repeated category the
    row that holds the whole (each value at least the rest of the group's sum) is kept, else the rows
    are added."""
    if len(set(cats)) == len(cats):
        return cats
    order = list(dict.fromkeys(cats))
    groups = {c: [i for i, x in enumerate(cats) if x == c] for c in order}
    picked: dict[str, list[float]] = {}
    for c, rows in groups.items():
        cols = [[s["data"][i] for i in rows] for s in series]
        whole = next((r for r in range(len(rows))
                      if all(col[r] >= sum(col) - col[r] - 1e-6 * max(1.0, abs(sum(col))) for col in cols)), None)
        picked[c] = [col[whole] if whole is not None else sum(col) for col in cols]
    for j, s in enumerate(series):
        s["data"] = [picked[c][j] for c in order]
    return order


def _drop_sum_series(bars: list[dict]) -> list[dict]:
    """A series that is exactly the sum of the others (Meters Installed beside Consumer / DT /
    Feeder) is the total the dashboard shows in its tooltip, not a series. Seen live: the daily
    installs area drew it over its own parts, both in navy."""
    if len(bars) < 3:
        return bars
    for s in bars:
        rest = [o for o in bars if o is not s]
        if all(abs(v - sum(o["data"][i] for o in rest)) <= 0.005 * max(1.0, abs(v)) for i, v in enumerate(s["data"])):
            return rest
    return bars


_BAND_NAME_RE = re.compile(r"^(?:age|days?|band|bucket)?[_\s-]*(\d+)[_\s-]*(?:(?:to|-|_)[_\s-]*(\d+)|(plus|\+|above))?"
                           r"[_\s-]*(?:days?)?$", re.I)


def _band_order(bars: list[dict]) -> list[dict]:
    """Age bands as the dashboard shows them: youngest first, named '91-120 days' / '120+ days'.
    Seen live: the split by age_key gave the legend 'age_120_plus', 'age_91_120' -- oldest band
    at the bottom of every stack, under its raw key."""
    parsed = [_BAND_NAME_RE.match(s["name"].strip()) for s in bars]
    if len(bars) < 2 or not all(parsed):
        return bars
    for s, m in zip(bars, parsed):
        lo, hi = m.group(1), m.group(2)
        s["name"] = f"{lo}-{hi} days" if hi else f"{lo}+ days" if m.group(3) else s["name"]
    return [s for _, s in sorted(zip(parsed, bars), key=lambda p: (int(p[0].group(1)), p[0].group(2) is None))]


def shape(kpi: str, planned: str, cats: list[str], series: list[dict], x_alias: Any) -> dict:
    """The dashboard card's chart for these rows: {type, x, series, labels, rates?}.
    `series` are the built series ({name, data, type?}); rates leave the bars."""
    cats = _merge_repeats(cats, series)
    rates = [s for s in series if _RATE_RE.search(s["name"]) or s.get("type") == "line"]
    bars = [s for s in series if s not in rates] or rates
    rates = [s for s in rates if s not in bars]
    # A productivity card plots productivity itself (the dashboard's comparison chart); the
    # installs / technicians behind it stay in the table. Seen live: installs were the bars and
    # productivity only a label on top.
    prod = [s for s in rates if re.search(r"productivity", s["name"], re.I)]
    if kpi in ("mi_productivity", "monthly_productivity", "om_productivity") and prod:
        bars, rates = prod[:1], []
    bars = _band_order(_drop_sum_series(bars))
    for s in bars + rates:
        s.pop("type", None)
    n, k = len(cats), len(bars)
    dated, geo = _is_dates(cats), _is_geo(x_alias, cats)
    # Funnel stages are each a subset of the one before: side by side, never stacked.
    nested = k > 1 and (kpi == "sat_invoice_funnel" or any(_WHOLE_RE.search(s["name"]) for s in bars))

    if planned == "radial" and k == 1 and not rates and n <= 4 and all(0 <= v <= 100 for v in bars[0]["data"]):
        ctype = "radial"                                   # the dashboard's utilization gauge
    elif kpi in ("om_active_ageing", "om_escalated") and k == 1 and not dated and not geo and n >= 2 \
            and all(v >= 0 for v in bars[0]["data"]):
        cats, bars[0]["data"] = _donut(cats, bars[0]["data"])
        return {"type": "donut", "x": cats, "series": bars, "labels": "segments", "rates": []}
    elif dated:
        ctype = "area" if n > 25 else ("grouped_bar" if nested else "stacked_bar" if k > 1 else "bar")
    else:
        horizontal = n > 8
        if kpi in ("mi_productivity", "monthly_productivity"):
            horizontal = False
        elif kpi in ("mi_pace_stock", "om_productivity"):
            horizontal = True
        if nested:
            ctype = "horizontal_bar" if horizontal else "grouped_bar"
        elif k > 1:
            ctype = "stacked_horizontal_bar" if horizontal else "stacked_bar"
        else:
            ctype = "horizontal_bar" if horizontal else "bar"

    view = f"{kpi}_{'geo' if geo else 'stage'}"
    # A single productivity bar keeps its value: the dashboard drops labels to avoid crowding its
    # stacked productivity / installs / technicians bars, which a lone bar does not have.
    labels = ("none" if (kpi in _NO_LABELS and k > 1) or view in _NO_LABELS
              else "total" if (kpi in _TOTAL_LABELS or view in _TOTAL_LABELS) and k > 1
              else "segments")
    return {"type": ctype, "x": cats, "series": bars, "labels": labels, "rates": rates}
