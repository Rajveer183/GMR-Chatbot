"""One detailed chart when the answer's rows mix dates and places.

A breakdown by day AND project (and often Consumer / DT / Feeder too) used to be drawn on one axis of
'2026-11-08 · agra' labels: thirty crowded bars. It is drawn instead as the dashboard's trend
("MI Progress Trend", HES-FRONTEND-GMR KPIDetailModal.jsx): the dates on the axis, the card's parts
stacked, and the place split carried in the hover card -- each project's figure for that day, the way
the dashboard's detail tooltip lists a slice's rows. One chart, nothing hidden, no model call and no
extra query: it only re-adds the rows already fetched. Every other chart is left exactly as it was.

It changes nothing unless every figure is exact:
  * the per-date figure is a sum over places (places are disjoint, so exact for counts); rates,
    percentages and averages never add up and are left out
  * the per-place rows add the card's parts (Consumer + DT + Feeder) -- never a total beside its parts
  * the rows must be the whole result: no top-N, not cut at the row cap
  * ROLLUP 'Total' / 'All' rows are dropped first, so nothing is counted twice
  * any other label column that varies (zone under project is fine, an unrelated split is not) -> nothing
"""
from __future__ import annotations

import re
from typing import Any

import pandas as pd

from gmr_agents.presentation import dashboard_charts

# Not a measure at all.
_SKIP_RE = re.compile(r"^(?:ord|rank|sort|seq|row_?num|year|yr|month_?num|week_?num|day_?num)$|_id$|^id$|"
                      r"(?:^|_)(?:no|code|number|serial|lat|lon|latitude|longitude)$", re.I)
# Averages / levels / rates never add up over places.
_LEVEL_RE = re.compile(r"avg|average|mean|median|ratio|share|index|score|\bpf\b|power_factor|voltage|current|"
                       r"\bload|kva|\bkw|days|hours|duration|\btime|\btat\b|per_|_per\b|"
                       # rates in column names: 'sat_pct' has no word boundary before 'pct'
                       r"pct|percent|%|(?:^|_)rate(?:$|_)|utili[sz]ation|efficiency|productivity|coverage|compliance",
                       re.I)
_PLACE_ORDER = ["project", "cluster", "discom", "zone", "circle", "division", "sub_division", "subdivision",
                "warehouse", "oem", "sku", "region"]
_PLACE_RE = re.compile(r"project|cluster|discom|zone|circle|division|warehouse|\boem\b|\bsku\b|region", re.I)
_SPLIT_RE = re.compile(r"category|meter_?type|type$|stage|status|band|bucket|remark|reason", re.I)
_TOTAL_WORDS = {"total", "grand total", "all", "overall", "none", "nan", ""}
_DETAIL_ROWS = 8      # places listed in the hover card; the rest are one 'Others (N)' row
_SPLIT_MAX = 8


def _is_date_col(s: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    vals = s.dropna().astype(str).str.strip()
    return bool(len(vals)) and vals.map(lambda v: bool(dashboard_charts._DATE_RE.search(v))).all()


def _date_text(v: Any) -> str:
    if isinstance(v, pd.Timestamp) or hasattr(v, "strftime"):
        try:
            return v.strftime("%Y-%m-%d")
        except Exception:  # noqa: BLE001
            pass
    text = str(v).strip()
    return text[:10] if re.match(r"^\d{4}-\d{2}-\d{2}[ T]", text) else text


def _place_rank(col: str) -> int:
    low = col.lower()
    return next((i for i, p in enumerate(_PLACE_ORDER) if p in low), len(_PLACE_ORDER))


def _adds_up(name: str) -> bool:
    return not dashboard_charts._RATE_RE.search(name) and not _LEVEL_RE.search(name) and not _SKIP_RE.search(name)


def _rows_complete(plan: dict, result: dict, df: pd.DataFrame) -> bool:
    sql = str(plan.get("sql") or "")
    if re.search(r"\bfetch\s+first\b|\btop\s+\d+\b|\brow_number\s*\(|\brank\s*\(", sql, re.I):
        return False
    # A LIMIT the rows did not reach cut nothing -- the executor's own row cap ('LIMIT 10000') is on every
    # query (seen live 18 Nov). A LIMIT the rows filled is a top-N: partial rows.
    if any(len(df) >= int(n) for n in re.findall(r"\blimit\s+(\d+)", sql, re.I)):
        return False
    try:
        from gmr_agents.execution.executor import _ROW_CAP
    except Exception:  # noqa: BLE001
        _ROW_CAP = 10000
    return len(df) < _ROW_CAP and int(result.get("group_count") or 0) <= len(df)


def parts_of(kpi: str | None, cols: list) -> list:
    """The columns that are the card's own parts (Consumer / DT / Feeder, OK / Failed / Other, age bands)."""
    pattern = dashboard_charts._CARD_COLUMNS.get(kpi or "")
    return [c for c in cols if pattern and re.search(pattern, str(c), re.I) and not re.search(r"total", str(c), re.I)]


def views(plan: dict, result: dict, df: pd.DataFrame, primary: dict | None) -> list[dict] | None:
    """[one detailed trend chart] in place of a main chart whose axis mixes dates and places, or None to
    keep the main chart as it is."""
    if not primary or primary.get("type") in ("radial", "heatmap", "combo", "funnel") or len(df) < 3:
        return None
    if not any(" · " in str(v) for v in primary.get("x") or []):
        return None
    if not _rows_complete(plan, result, df):
        return None
    # A measure with no value on any row (FILTER sums: no 'Failed' upload this week) comes back all NULL and
    # reads as text; it is a zero series.
    empty = [c for c in df.columns if df[c].dtype == object and df[c].isna().all()]
    if empty:
        df = df.assign(**{c: 0.0 for c in empty})
    labels = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_datetime64_any_dtype(df[c])]
    if not labels:
        return None
    data = df[~df[labels].apply(lambda c: c.astype(str).str.strip().str.lower().isin(_TOTAL_WORDS)
                                | c.astype(str).str.strip().str.lower().str.startswith("all ")).any(axis=1)]
    if len(data) < 3:
        return None
    varying = [c for c in labels if data[c].astype(str).nunique() > 1]
    dates = [c for c in varying if _is_date_col(data[c])]
    places = sorted([c for c in varying if c not in dates and _PLACE_RE.search(str(c))], key=_place_rank)
    if not dates or not places:
        return None
    d, g = dates[0], places[0]
    for c in dates[1:]:       # period_end beside period_start must follow it
        if data.groupby(data[d].astype(str))[c].nunique().max() > 1:
            return None
    for c in places[1:]:      # a finer place must sit inside one coarser place (zone inside project)
        if data.groupby(data[c].astype(str))[g].nunique().max() > 1:
            return None
    others = [c for c in varying if c not in dates and c not in places]
    split = None
    if others:
        if len(others) > 1 or not _SPLIT_RE.search(str(others[0])) or data[others[0]].nunique() > _SPLIT_MAX:
            return None
        split = others[0]

    kpi = dashboard_charts.kpi_for(str(plan.get("sql") or ""), [str(c) for c in data.columns], plan.get("question"))
    if kpi in dashboard_charts.hes_charts.HES_KPIS:
        return None
    nums = [c for c in data.columns if c not in labels and pd.api.types.is_numeric_dtype(data[c]) and _adds_up(str(c))]
    if kpi and nums:
        nums = [c for c in dashboard_charts.card_measure(kpi, data, nums) if _adds_up(str(c))]
        parts = parts_of(kpi, nums)
        if len(parts) >= 2:
            nums = parts      # a whole beside its parts would be stacked on them
    if not nums or (split and len(nums) > 1):
        return None
    # Several measures that are not one card's parts may hold each other (installed / SAT done): side by
    # side, and no per-place sum across them.
    are_parts = len(nums) == 1 or bool(split) or (kpi is not None and len(parts_of(kpi, nums)) == len(nums))
    label = lambda c: (plan.get("labels") or {}).get(c) or str(c).replace("_", " ").title()   # noqa: E731

    keys = data[d].map(_date_text)
    frame = data.assign(_k=keys)
    if split:
        tab = frame.pivot_table(index="_k", columns=split, values=nums[0], aggfunc="sum", sort=False).fillna(0)
        cols = [(str(s), tab[s]) for s in tab.columns]
    elif len(nums) == 1:
        # One measure: the places are the series (the stack shows each place, the hover adds the total).
        tab = frame.pivot_table(index="_k", columns=g, values=nums[0], aggfunc="sum", sort=False).fillna(0)
        tab = tab[list(tab.sum().sort_values(ascending=False).index)]
        cols = [(str(s), tab[s]) for s in tab.columns]
    else:
        tab = frame.groupby("_k", sort=False)[nums].sum()
        cols = [(label(c), tab[c]) for c in nums]
    order = list(tab.index)
    if all(re.match(r"^\d{4}-\d{2}", str(k)) for k in order):
        order = sorted(order)
    from gmr_agents.agents.common import _CHART_POINTS
    if len(order) < 2 or len(order) > max(int(_CHART_POINTS or 40), 40):
        return None
    series = [{"name": n, "data": [float(col[i]) for i in order]} for n, col in cols]

    # The hover card's place rows: each place's figure on that date (the card's parts added up).
    details = None
    if are_parts and (split or len(nums) > 1):
        per = frame.assign(_v=frame[nums].sum(axis=1) if not split else frame[nums[0]]) \
            .groupby(["_k", frame[g].astype(str).str.strip()])["_v"].sum()
        details = []
        for k in order:
            rows = sorted(((str(p), float(v)) for p, v in per.loc[k].items()), key=lambda r: -r[1]) if k in per.index else []
            if len(rows) > _DETAIL_ROWS + 1:
                rest = rows[_DETAIL_ROWS:]
                rows = rows[:_DETAIL_ROWS] + [(f"Others ({len(rest)})", sum(v for _p, v in rest))]
            details.append([[p, v] for p, v in rows])

    from gmr_agents.presentation import blocks   # late: blocks imports this module
    cats = [str(i) for i in order]
    planned = "area" if len(cats) > 25 else ("stacked_bar" if are_parts and len(series) > 1 else
                                             "grouped_bar" if len(series) > 1 else "bar")
    out: dict[str, Any] = {"id": primary.get("id") or "primary_chart", "title": primary.get("title")}
    dash = dashboard_charts.shape(kpi, planned, list(cats), [dict(s) for s in series], d, None) if kpi else None
    if dash is not None:
        out.update({"type": dash["type"], "x": dash["x"], "y": dash["series"][0]["data"],
                    **({"series": dash["series"]} if len(dash["series"]) > 1 else {}),
                    "dashboard": kpi, "labels": dash["labels"],
                    **({"rates": dash["rates"]} if dash["rates"] else {})})
        if dash["x"] != cats:
            details = None    # the card reordered its axis: rows would no longer line up
    else:
        ctype, _why = blocks._chart_fallback(planned, cats, series, 0)
        out.update({"type": ctype, "x": cats, "y": series[0]["data"],
                    **({"series": series} if len(series) > 1 else {}), **({"dashboard": kpi} if kpi else {})})
    if details and any(details):
        out["details"], out["details_label"] = details, label(g)
    if primary.get("unit"):
        out["unit"] = primary["unit"]
    return [out]
