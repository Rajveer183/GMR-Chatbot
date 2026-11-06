"""The dashboard's own chart for the Technical (HES) KPI boards: power quality, DT load, load violation.

Source: Documents/WFM/HES-FRONTEND-GMR, branch vapt-rbac-remediation -- the layout-V voltage boards
(src/components/kpi/board/voltage/*: Donut, MeanBars, DirectionCells), Layout A / C / E
(src/components/kpi/layouts/KpiLayouts.jsx) and the DT boards (bespoke/DtLoadBoard.jsx, DtLoadCharts.jsx).
Colours live in the chatbot frontend (frontend/src/components/charts/dashboardPalette.ts, HES_COLOURS);
this module only decides the chart and sends canonical colour keys:

  * point_keys  -- one key per category (donut slice / bar): a status, a loading band, a load-change band,
                   or a role colour ('mean', 'series', 'overload', 'highlight')
  * series_keys -- one key per series for grouped bars (Poor / Moderate, Dips / Surges, last / this month)

What the boards draw, by answer shape:
  * a status split            -> donut, the board's status order, 'Offline' left out (Donut / Layout A)
  * DT loading / change bands -> vertical columns in band order, each bar its band's colour (BandColumns)
  * one measure per area      -> horizontal bars worst first: grey on the voltage boards (MeanBars), blue on
                                 the % time boards (Layout A RankedBar, top 12), red for DTs above rating
                                 (RankBars, top 10); load violation = vertical columns, the highest bar red
                                 (Layout C ValueColumn)
  * two shares per area       -> grouped horizontal bars in the DirectionCells colours (the board's tinted
                                 cells: Poor / Moderate, Dips / Surges, 5-10% / over 10%)
  * a DT / feeder list        -> horizontal bars, each DT coloured by its loading or change band (RankBars)
  * last vs this month        -> grouped columns (CompareColumns)
Anything else returns None and keeps the planner's chart (still in the card's palette).
"""
from __future__ import annotations

import re
from typing import Any

import pandas as pd

# hes_gold table -> card. dt_peak_loading must not swallow dt_peak_loading_feeder.
TABLE_KPI: list[tuple[str, str]] = [
    (r"voltage_deviation_index", "pq_vdi"),
    (r"voltage_fluctuation_index", "pq_vfi"),
    (r"voltage_unbalance_index", "pq_vui"),
    (r"voltage_drop", "pq_drop"),
    (r"current_unbalance", "pq_cu"),
    (r"pct_time_out_of_band", "pq_oob"),
    (r"pct_time_current_imbalance", "pq_pci"),
    (r"low_voltage_pockets(?:_meters)?", "lv_pockets"),
    (r"dt_loading(?!_daily)", "dt_loading"),
    (r"dt_peak_loading_feeder", "dt_peak_feeder"),
    (r"dt_peak_loading(?!_feeder)", "dt_peak"),
    (r"dt_load_kva", "dt_kva"),
    (r"dt_load_rise", "dt_rise"),
    (r"load_violation|load_assessed_consumers", "load_violation"),
]
HES_KPIS = {k for _p, k in TABLE_KPI}

# The board's status order, worst first (vdiModel / vfiModel / vuiModel / dropModel / cuModel STATUS_ORDER,
# pqSpecs.js severity_order, dtLoad.js LOAD_STATUS_ORDER).
STATUS_ORDER: dict[str, list[str]] = {
    "pq_vdi": ["Poor", "Moderate", "Good"],
    "pq_vfi": ["Unstable", "Moderate", "Stable"],
    "pq_vui": ["Poor", "Moderate", "Good", "1-Phase"],
    "pq_drop": ["Severe", "Moderate", "Mild", "OK"],
    "pq_cu": ["Poor", "Elevated", "Moderate", "Good", "1-Phase", "Offline/NoLoad"],
    "pq_oob": ["ChronicViolation", "NonCompliant", "Borderline", "Compliant"],
    "pq_pci": ["NonCompliant", "Borderline", "Compliant", "1-Phase", "Offline/NoLoad"],
    "lv_pockets": ["Critical", "Poor", "Average"],
    "dt_peak": ["Overloaded >100%", "Overloaded 90-100%", "Normal", "Underloaded <20%"],
    "dt_peak_feeder": ["Overloaded >100%", "Overloaded 90-100%", "Normal", "Underloaded <20%"],
}
LOADING_BANDS = ["<20%", "20-35%", "35-55%", "55-70%", "70-85%", "85-100%", "100-120%", ">120%"]
RISE_BANDS = ["Fell >10 pp", "Fell 5-10 pp", "Stable (±5 pp)", "Rose 5-10 pp", "Rose 10-20 pp", "Rose >20 pp",
              "No previous month"]
AGEING = ["3-30 Days", "31-60 Days", "61-90 Days", "More than 90 Days"]

_GEO_COL_RE = re.compile(r"project|cluster|discom|zone|circle|division|sub_?station|role|meter_?type|category", re.I)
# a list of DTs / feeders / meters (never meter_type, a place column)
_ITEM_COL_RE = re.compile(r"meter_no|meter_number|dtr_code|dt_code|feeder_code|max_loaded_dt|^(?:meter|dt|dtr|feeder)$", re.I)
# The card's own measure when a list carries several figures (RankBars plot one).
_LIST_MEASURE = {
    "dt_peak": r"pct_peak|peak_loading|peak.*(?:pct|%)", "dt_kva": r"hours", "dt_rise": r"rise|change|points|_pp",
    "dt_peak_feeder": r"overloaded_?90|above_?90|90", "dt_loading": r"pct|%|overload_blocks", "load_violation": r"excess",
}
_COUNT_RE = re.compile(r"meters?|count|dts|consumers|^n$|number", re.I)
_BLANK = {"unmapped", "", "none", "null", "nan", "unknown"}
_TOTAL = {"total", "grand total", "all", "all circles"}


def _norm(s: Any) -> str:
    return re.sub(r"[^a-z0-9<>+]", "", str(s).lower()).replace("pp", "")


def _match(cats: list[str], names: list[str]) -> list[str] | None:
    """Canonical name per category when every category is one of `names` (spelling-insensitive)."""
    by_norm = {_norm(n): n for n in names}
    out = [by_norm.get(_norm(c)) for c in cats]
    return out if cats and all(out) else None


def loading_band(pct: float) -> str:
    for cut, band in ((20, "<20%"), (35, "20-35%"), (55, "35-55%"), (70, "55-70%"), (85, "70-85%"),
                      (100, "85-100%"), (120, "100-120%")):
        if pct < cut:
            return band
    return ">120%"


def rise_band(pp: float) -> str:
    for cut, band in ((-10, "Fell >10 pp"), (-5, "Fell 5-10 pp"), (5, "Stable (±5 pp)"), (10, "Rose 5-10 pp"),
                      (20, "Rose 10-20 pp")):
        if pp < cut:
            return band
    return "Rose >20 pp"


def _reorder(cats: list[str], keys: list[str], series: list[dict], order: list[str], drop: set[str] = frozenset()):
    rank = {k: i for i, k in enumerate(order)}
    idx = sorted((i for i, k in enumerate(keys) if k not in drop), key=lambda i: rank.get(keys[i], len(order)))
    for s in series:
        s["data"] = [s["data"][i] for i in idx]
    return [cats[i] for i in idx], [keys[i] for i in idx]


def _series_key(kpi: str, name: str) -> str | None:
    n = name.lower()
    if kpi in ("pq_vdi", "pq_vui", "pq_drop", "pq_cu", "lv_pockets"):
        for word in ("poor", "severe", "critical", "moderate", "elevated", "mild", "good", "ok", "average"):
            if re.search(rf"\b{word}\b", n.replace("_", " ")):
                return word.title() if word not in ("ok",) else "OK"
        if re.search(r"(?:gt|above|over)_?10|>\s*10", n):
            return "Poor"
        if re.search(r"5_?10|5\s*-\s*10", n):
            return "Elevated"
        if re.search(r"(?:gt|above|over)_?3|>\s*3", n):
            return "Poor"
    if kpi == "pq_vfi":
        if "dip" in n:
            return "Dips"
        if "surge" in n:
            return "Surges"
    if kpi == "dt_rise":
        if re.search(r"prev|last", n):
            return "Last month"
        if re.search(r"this|now|curr", n):
            return "This month"
    if kpi in ("dt_peak", "dt_peak_feeder"):
        if re.search(r"100", n) and re.search(r"over|above|gt", n):
            return "Overloaded >100%"
        if re.search(r"90", n):
            return "Overloaded 90-100%"
        if re.search(r"under", n):
            return "Underloaded <20%"
    if kpi == "pq_oob":
        if re.search(r"under|uv", n):
            return "Under-Voltage"
        if re.search(r"over|ov", n):
            return "Over-Voltage"
    return None


def _band_column(rows: Any, kpi: str) -> list[str] | None:
    """Per-row band keys for a DT list: from a band column when present, else from the loading / rise figure."""
    if rows is None or not isinstance(rows, pd.DataFrame) or rows.empty:
        return None
    cols = list(rows.columns)
    band_col = next((c for c in cols if re.search(r"band", str(c), re.I)), None)
    if band_col is not None:
        keys = _match([str(v) for v in rows[band_col]], RISE_BANDS if kpi == "dt_rise" else LOADING_BANDS)
        if keys:
            return keys
    if kpi == "dt_rise":
        col = next((c for c in cols if re.search(r"rise|change|points|_pp", str(c), re.I)), None)
        return [rise_band(float(v)) for v in rows[col]] if col is not None else None
    col = next((c for c in cols if re.search(r"(?:peak|max).*(?:pct|%|loading)|pct_peak|peak_loading", str(c), re.I)), None)
    return [loading_band(float(v)) for v in rows[col]] if col is not None else None


def shape(kpi: str, cats: list[str], series: list[dict], x_alias: Any, rows: Any = None) -> dict | None:
    """The board's chart for these rows, or None (keep the planner's chart). Works on copies: on None the
    caller draws the planner's chart from its own, untouched series."""
    series = [{"name": s["name"], "data": list(s["data"])} for s in series]
    # The Total row and the blank / unmapped area (rankings skip it, like the boards' MeanBars).
    keep = [i for i, c in enumerate(cats) if str(c).strip().lower() not in _TOTAL | _BLANK]
    if isinstance(rows, pd.DataFrame) and len(rows) != len(cats):
        rows = None                                   # rows no longer line up with the categories
    if len(keep) < len(cats):
        cats = [cats[i] for i in keep]
        for s in series:
            s["data"] = [s["data"][i] for i in keep]
        if rows is not None:
            rows = rows.iloc[keep]
    # One category whose series are the statuses / bands (low-voltage pockets by band under a single ageing
    # bucket): the parts are the categories -- the board's band rings, drawn as its donut.
    statuses = STATUS_ORDER.get(kpi)
    if len(cats) == 1 and len(series) >= 2 and statuses and _match([s["name"] for s in series], statuses):
        cats, series = [s["name"] for s in series], [{"name": str(cats[0]), "data": [s["data"][0] for s in series]}]
    if len(cats) < 2 or not series:
        return None
    k = len(series)
    alias = str(x_alias or "")

    # 1. A status split -> the board's donut (or, for several measures, stays with the planner).
    statuses = STATUS_ORDER.get(kpi)
    keys = _match(cats, statuses + ["Offline", "Phase missing", "No valid slot"]) if statuses else None
    if keys and k > 1:   # a split carrying several figures per status: the donut plots the meter count
        counts = [x for x in series if _COUNT_RE.search(x["name"]) and not re.search(r"%|pct|share", x["name"], re.I)]
        series = counts[:1] or series[:1]
    if keys:
        cats, keys = _reorder(cats, keys, series, statuses, drop={"Offline"})
        return {"type": "donut", "x": cats, "series": series, "labels": "segments", "rates": [], "point_keys": keys}

    # 2. Ordered bands -> columns in band order, each bar its band's colour.
    for order in (LOADING_BANDS, RISE_BANDS, AGEING):
        keys = _match(cats, order)
        if keys and k == 1:
            cats, keys = _reorder(cats, keys, series, order)
            return {"type": "bar", "x": cats, "series": series, "labels": "total", "rates": [], "point_keys": keys}

    item = bool(_ITEM_COL_RE.search(alias)) or bool(cats and re.match(r"^[A-Z]{2}\d{6,}$|^\d{8,}[A-Z]{2,}$", str(cats[0])))
    geo = bool(_GEO_COL_RE.search(alias)) and not item

    # 3. A DT / feeder list -> ranked horizontal bars.
    if item and k > 1 and kpi in _LIST_MEASURE:
        pick = [x for x in series if re.search(_LIST_MEASURE[kpi], x["name"].replace(" ", "_"), re.I)]
        series, k = pick[:1] or series[:1], 1
    if item and k == 1 and kpi in ("dt_peak", "dt_kva", "dt_rise", "dt_loading", "dt_peak_feeder"):
        band_keys = None if kpi == "dt_peak_feeder" else _band_column(rows, kpi)
        point = band_keys if band_keys and len(band_keys) == len(cats) else ["overload"] * len(cats)
        return {"type": "horizontal_bar", "x": cats, "series": series, "labels": "segments", "rates": [],
                "point_keys": point}

    if not geo:
        return None

    # Load violation by area plots flagged consumers (Layout C), even when the planner charted only the %.
    if kpi == "load_violation" and isinstance(rows, pd.DataFrame):
        col = next((c for c in rows.columns if re.search(r"violat|flag", str(c), re.I)
                    and not re.search(r"%|pct|percent|rate|share", str(c), re.I)
                    and pd.api.types.is_numeric_dtype(rows[c])), None)
        if col is not None and not any(x["name"] == str(col) for x in series):
            series = [{"name": str(col).replace("_", " ").title(), "data": [float(v) for v in rows[col]]}]

    # 4. Per area. The voltage boards plot the average (MeanBars); counts riding along stay in the table.
    if k > 1 and kpi.startswith("pq_"):
        mean = [s for s in series if re.search(r"avg|average|mean|depth|wmean", s["name"], re.I)]
        pcts = [s for s in series if re.search(r"%|pct|percent|share|time", s["name"], re.I)]
        shares = [s for s in pcts if _series_key(kpi, s["name"])]
        if mean:
            series = mean[:1]
        elif len(shares) >= 2:
            series = shares[:2]            # DirectionCells: Poor / Moderate, Dips / Surges, 5-10% / over 10%
        elif pcts:
            series = pcts[:1]              # Severe meters % by circle, % time out of band ...
        else:
            series = [s for s in series if not _COUNT_RE.search(s["name"])][:1] or series[:1]
    elif k > 1 and kpi == "load_violation":
        flagged = [s for s in series if re.search(r"violat|flag", s["name"], re.I)
                   and not re.search(r"%|pct|percent|rate|share", s["name"], re.I)]
        series = flagged[:1] or series
    elif k > 1 and kpi in ("dt_loading", "dt_peak"):
        pct = [s for s in series if re.search(r"%|pct|percent|share", s["name"], re.I)]
        series = pct[:1] or series
    k = len(series)
    if kpi == "load_violation" and k == 1:
        order = sorted(range(len(cats)), key=lambda i: -series[0]["data"][i])
        cats = [cats[i] for i in order]
        series[0]["data"] = [series[0]["data"][i] for i in order]
        return {"type": "bar", "x": cats, "series": series, "labels": "total", "rates": [],
                "point_keys": ["highlight"] + ["series"] * (len(cats) - 1)}
    if kpi == "dt_rise" and k == 2:
        skeys = [_series_key(kpi, s["name"]) for s in series]
        if all(skeys):
            return {"type": "grouped_bar", "x": cats, "series": series, "labels": "none", "rates": [],
                    "series_keys": skeys}
    if k == 1:
        role = ("mean" if kpi.startswith("pq_") and kpi not in ("pq_oob", "pq_pci")
                else (_series_key(kpi, series[0]["name"]) or "overload") if kpi in ("dt_loading", "dt_peak", "dt_peak_feeder")
                else "series")
        if kpi in ("pq_oob", "pq_pci", "dt_loading", "dt_peak"):   # ranked: worst first, top 12 / 10
            order = sorted(range(len(cats)), key=lambda i: -series[0]["data"][i])[:12 if kpi.startswith("pq_") else 10]
            cats = [cats[i] for i in order]
            series[0]["data"] = [series[0]["data"][i] for i in order]
        return {"type": "horizontal_bar", "x": cats, "series": series, "labels": "segments", "rates": [],
                "point_keys": [role] * len(cats)}
    skeys = [_series_key(kpi, s["name"]) for s in series]
    if all(skeys):
        # two series on 'horizontal_bar' = grouped horizontal bars (frontend dashboardOptions)
        return {"type": "stacked_horizontal_bar" if kpi in ("lv_pockets", "dt_peak") else "horizontal_bar",
                "x": cats, "series": series, "labels": "none", "rates": [], "series_keys": skeys}
    return None
