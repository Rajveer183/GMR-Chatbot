"""Chart Maker: KPI cards, chart and tables built by code from the result rows.

Moved from gmr_agents/llm_pipeline.py unchanged.
"""

from __future__ import annotations
import re
from datetime import datetime
from typing import Any
import pandas as pd
from shared.formatting import format_number
from gmr_agents.catalog import business_rules
from gmr_agents.presentation import dashboard_charts
import logging
from gmr_agents.agents.common import _CHART_POINTS, _IST, _TABLE_ROWS

log = logging.getLogger("gmr_agents.llm_pipeline")  # same logger name as before the split


# --------------------------------------------------------------------------- #
# Presentation -- exactly what the planner asked for
# --------------------------------------------------------------------------- #

def _label(plan: dict, col: str) -> str:
    return plan["labels"].get(col) or col.replace("_", " ").title()


def _number(v: Any) -> float | None:
    n = pd.to_numeric(v, errors="coerce")
    return None if pd.isna(n) else float(n)


def _cell(v: Any, raw: bool = False) -> Any:
    """Display text of one table cell; `raw` (a year or an ID column) keeps whole numbers
    without separators -- consumer 4065288000, not 4,065,288,000."""
    if v is None or (not isinstance(v, (list, dict)) and pd.isna(v)):
        return ""
    if isinstance(v, bool):
        return "Yes" if v else "No"
    if isinstance(v, (int, float)) or type(v).__name__ == "Decimal":
        if raw and float(v).is_integer():
            return str(int(v))
        return format_number(float(v))
    return _tidy_ts(str(v))


_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2})(?::\d{2}(?:\.\d+)?)?"
                    r"(?:\s*(UTC|Z|[+-]\d{2}:?\d{2}))?$")


def _tidy_ts(text: str) -> str:
    """'2025-06-16 00:00:00.000000 UTC' -> '2025-06-16'; a zoned time is shown in IST, as on the
    dashboard: '2026-10-05 03:13:19.59+00:00' -> '2026-10-05 08:43' (seen live: 'export today's
    headline stats' showed refreshed_at as 03:13 -- the UTC clock -- while the dashboard said 08:43)."""
    m = _TS_RE.match(text.strip())
    if not m:
        return text
    if m.group(2) == "00:00":
        return m.group(1)
    zone = m.group(3)
    if zone:
        offset = "+00:00" if zone in ("UTC", "Z") else (zone if ":" in zone else f"{zone[:3]}:{zone[3:]}")
        try:
            local = datetime.fromisoformat(f"{m.group(1)}T{m.group(2)}{offset}").astimezone(_IST)
            return f"{local:%Y-%m-%d %H:%M}"
        except ValueError:
            pass
    return f"{m.group(1)} {m.group(2)}"


# Years and identifiers: consumer_number, sat_number, meter_serial, feeder_code, imei, mobile ...
_RAW_COL_RE = re.compile(r"(?:^|_)(?:year|yr|fy|id|no|num|number|serial|code|imei|sim|mobile|phone|pincode|ticket)$", re.I)


def raw_cols(columns: Any) -> set[int]:
    """Positions of the columns whose whole numbers are shown as-is (no thousands separators)."""
    return {i for i, c in enumerate(columns) if _RAW_COL_RE.search(str(c).strip().lower().replace(" ", "_"))}


_MULTI_SERIES = {"grouped_bar", "stacked_bar", "stacked_horizontal_bar", "percent_stacked_bar", "heatmap"}

_SINGLE_SERIES = {"pie", "donut", "funnel", "radial"}

_STACKED = {"stacked_bar", "stacked_horizontal_bar", "percent_stacked_bar"}

_WHOLE_SERIES_RE = re.compile(r"\b(total|suppl(y|ied|ies)|inventory|received)\b", re.I)


def _chart_fallback(ctype: str, cats: list[str], series: list[dict], y2_count: int) -> tuple[str, str | None]:
    """The planner picked ctype (few-shot guide in charts.yaml). Keep it when the
    data can show it well; otherwise step to the closest type that can, and say why.
    Mirrors the WFM dashboard's own rules (more than 8 categories -> horizontal bars)."""
    n, k = len(cats), len(series)
    values = [v for s in series for v in s["data"]]
    if ctype in _SINGLE_SERIES and k > 1 and ctype != "radial":
        return ("grouped_bar" if n <= 8 else "horizontal_bar"), "several series"
    if ctype in ("pie", "donut") and (n < 2 or n > 8 or any(v < 0 for v in values)):
        return ("bar" if n <= 8 else "horizontal_bar"), "slices must be 2-8 non-negative parts"
    if ctype == "radial" and (n > 4 or any(v < 0 or v > 100 for v in values)):
        return "bar", "radial shows up to 4 percentages (0-100)"
    if ctype in ("line", "area") and n < 3:
        return ("grouped_bar" if k > 1 else "bar"), "a trend needs 3+ points"
    if ctype == "combo" and (y2_count == 0 or y2_count == k):
        return ("grouped_bar" if k > 1 else "bar"), "combo needs bars and a line"
    if ctype in _MULTI_SERIES and k < 2:
        return ("bar" if n <= 8 else "horizontal_bar"), "only one series"
    # A total (supplied, inventory, total MI) already contains the other series, so stacking
    # adds it twice: zone-wise supplied + installed was drawn as an '11,37,105 meters' bar.
    if ctype in _STACKED and any(_WHOLE_SERIES_RE.search(str(s.get("name", ""))) for s in series):
        return ("grouped_bar" if n <= 12 else "line"), "a total is not a part to stack"
    if ctype == "heatmap" and n < 2:
        return "bar", "heatmap needs 2+ columns"
    long_names = max((len(c) for c in cats), default=0) > 14
    if ctype == "bar" and (n > 15 or (n > 8 and long_names)):
        return "horizontal_bar", "many or long category names"
    if ctype == "grouped_bar" and n > 12:
        return "line" if k > 1 else "horizontal_bar", "too many groups"
    if ctype == "stacked_bar" and (n > 15 or (n > 8 and long_names)):
        return "stacked_horizontal_bar", "many or long category names"
    return ctype, None


def _distinct_labels(df: pd.DataFrame, x: str, plotted: list[str]) -> list[str]:
    """x labels, with the other label columns appended when x alone repeats. Seen live: stock
    cover by project AND category drew three bars all named 'AGRA' -- now 'AGRA · CONSUMER',
    'AGRA · DT', 'AGRA · All' on the axis and in the hover tooltip."""
    labels = [str(v) for v in df[x]]
    if len(set(labels)) == len(labels):
        return labels
    candidates = [c for c in df.columns if c != x and c not in plotted and not pd.api.types.is_numeric_dtype(df[c])
                  and not pd.api.types.is_datetime64_any_dtype(df[c])
                  and not df[c].astype(str).str.match(r"^\d{4}-\d{2}(-\d{2})?").any()   # dates are values, not names
                  and df.groupby(df[x].astype(str))[c].nunique().max() > 1]
    extra: list[str] = []
    for c in candidates:   # only as many columns as it takes to tell the bars apart
        extra.append(c)
        if not df[[x, *extra]].astype(str).duplicated().any():
            break
    if not extra:
        return labels
    return [" · ".join([lab] + [_cell(df.iloc[i][c]) for c in extra if _cell(df.iloc[i][c]) != ""])
            for i, lab in enumerate(labels)]


def _is_subtotal(v: Any) -> bool:
    s = str(v).strip().lower()
    return s in ("total", "grand total", "all") or s.startswith("all ")


def _one_rollup_level(data: pd.DataFrame, x: Any, group: Any) -> tuple[pd.DataFrame, Any, Any]:
    """A multi-level ROLLUP mixes subtotal rows ('AGRA · All batches') with their detail rows
    ('AGRA · SAT-12'); a chart of both counts every meter twice. Keep one level: the detail rows
    only when the planner splits the outer x by the inner column (group), otherwise the outer
    level's subtotals, one bar per outer value (seen live: 'SAT-passed not invoiced' drew
    SAT-11/12/13 bars beside 'All batches' -- now one bar per project; batches stay in the table)."""
    labels = [c for c in data.columns if not pd.api.types.is_numeric_dtype(data[c])]
    data = data[~data[labels].apply(lambda col: col.astype(str).str.strip().str.lower()
                                    .isin({"total", "grand total"})).any(axis=1)] if labels else data
    if data.empty:
        return data, x, group
    sub = {c: data[c].map(_is_subtotal) for c in labels}
    inner = [c for c in labels if sub[c].any() and not sub[c].all()]
    if not inner:
        return data, x, group
    subs = pd.concat([sub[c] for c in inner], axis=1)
    outer = [c for c in labels if c not in inner]
    if x not in inner and group in inner:
        return data[~subs.any(axis=1)], x, group
    if not outer:
        return data, x, group
    return data[subs.all(axis=1)], (x if x in outer else outer[0]), None


def _build_chart(plan: dict, df: pd.DataFrame) -> dict | None:
    """The chart the planner chose (display.chart), shaped for the frontend:
      * long rows + "group" -> one series per group value (stacked/grouped/heatmap/multi-line)
      * one wide row (several y, no usable x) -> the y columns become the categories
      * the ROLLUP 'Total' row is left out (it would dwarf the bars)"""
    chart = plan["display"].get("chart") if isinstance(plan["display"].get("chart"), dict) else {}
    ctype = str(chart.get("type") or "none")
    # A dashboard card's rows always get the card's chart, even when the planner drew none.
    card = dashboard_charts.kpi_for(plan.get("sql"), [str(c) for c in df.columns], plan.get("question"))
    on_card = bool(card) and not dashboard_charts.user_named_type(plan.get("question"))
    named = dashboard_charts.exact_named_type(plan.get("question"))
    if named and ctype not in ("none", named) and named in business_rules.chart_types():
        ctype = named   # still checked by _chart_fallback below (a pie needs 2-8 parts, ...)
    if on_card and (ctype not in business_rules.chart_types() or ctype == "none"):
        chart = dashboard_charts.auto_spec(card, df) or chart
        ctype = str(chart.get("type") or "none")
    if ctype not in business_rules.chart_types() or ctype == "none":
        return None
    x, group = chart.get("x"), chart.get("group")
    ys = [y for y in (chart.get("y") or []) if y in df.columns]
    if on_card:
        measure = dashboard_charts.card_measure(card, df, ys)
        if measure != ys:
            ys, chart = measure, {**chart, "y2": []}
    if on_card and not group and len(ys) == 1 and x in df.columns:
        group = dashboard_charts.split_column(df, x, ys)
    y2 = [y for y in (chart.get("y2") or []) if y in df.columns and y not in ys]
    if not ys and not y2:
        return None
    # Categories are labels, never a measure. Seen live 27 Oct: 'consumer supply utilization type wise'
    # drew x = 98.84 / 91.12 / 32.18 (the utilization %), 'sat progress for feeder meters' x = 7874.0.
    if ctype != "radial" and x in df.columns and (pd.api.types.is_numeric_dtype(df[x]) or x in ys + y2):
        labels = [c for c in df.columns if c != group and not pd.api.types.is_numeric_dtype(df[c])]
        x = labels[0] if labels else None
    data = df
    if x in df.columns:
        data = df[~df[x].astype(str).str.strip().str.lower().isin({"total", "grand total"})]
    if data.empty:
        data = df
    elif x in data.columns:
        leveled, lx, lgroup = _one_rollup_level(data, x, group)
        if not leveled.empty:
            data, x, group = leveled, lx, lgroup

    # A gauge is named by its measure: when x is missing, numeric or one of the plotted
    # columns (planner wrote x = the % column), its categories are the column labels.
    radial_by_column = ctype == "radial" and len(data) == 1 and (
        x not in data.columns or x in ys + y2 or _number(data.iloc[0][x]) is not None)
    if radial_by_column or x not in data.columns or (
            len(data) == 1 and len(ys + y2) > 1 and ctype in _SINGLE_SERIES | {"bar", "horizontal_bar"}):
        row = data.iloc[0]   # one wide row: its columns are the categories
        cats = [_label(plan, y) for y in ys + y2]
        series = [{"name": plan.get("table_title") or "Value", "data": [_number(row[y]) or 0.0 for y in ys + y2]}]
        y2 = []
    # A numeric group (a stage's sort order) is not a split -- seen live: journey stages drew one
    # series per 'ord' value, '1' to '7', each holding a single bar.
    elif group in data.columns and group != x and ys and not pd.api.types.is_numeric_dtype(data[group]):
        pivot = data.pivot_table(index=x, columns=group, values=ys[0], aggfunc="sum", sort=False).fillna(0)
        pivot = pivot.head(_CHART_POINTS)
        cats = [str(c) for c in pivot.index]
        series = [{"name": str(g), "data": [_number(v) or 0.0 for v in pivot[g]]} for g in pivot.columns]
    else:
        cdf = data.head(_CHART_POINTS)
        cats = _distinct_labels(cdf, x, ys + y2)
        series = [{"name": _label(plan, y), "data": [_number(v) or 0.0 for v in cdf[y]]} for y in ys]
        series += [{"name": _label(plan, y), "data": [_number(v) or 0.0 for v in cdf[y]], "type": "line"} for y in y2]
    if not cats or not series:
        return None

    chosen = ctype
    # First priority: a question one of the dashboard's KPI cards answers is drawn the way
    # that card draws it (dashboard_charts.py); anything else keeps the planner's chart.
    kpi = dashboard_charts.kpi_for(plan.get("sql"), [s["name"] for s in series], plan.get("question"))
    if kpi and not dashboard_charts.user_named_type(plan.get("question")):
        dash = dashboard_charts.shape(kpi, ctype, cats, series, x)
        out = {"id": "primary_chart", "title": plan.get("table_title") or f"{dash['series'][0]['name']} by {_label(plan, x) if x else ''}",
               "type": dash["type"], "x": dash["x"], "y": dash["series"][0]["data"],
               **({"series": dash["series"]} if len(dash["series"]) > 1 else {}),
               "dashboard": kpi, "labels": dash["labels"],
               **({"rates": dash["rates"]} if dash["rates"] else {}),
               **({"unit": str(chart["unit"])} if chart.get("unit") else {})}
        if dash["type"] != chosen:
            out["chosen_type"], out["fallback_reason"] = chosen, "dashboard chart"
        return out
    ctype, why = _chart_fallback(ctype, cats, series, sum(1 for s in series if s.get("type") == "line"))
    if ctype != "combo":
        for s in series:
            s.pop("type", None)
    elif not any(s.get("type") for s in series[:1]):
        series[0]["type"] = "bar"
    names = ", ".join(s["name"] for s in series[:3])
    out = {"id": "primary_chart", "title": plan.get("table_title") or f"{names} by {_label(plan, x) if x else ''}",
           "type": ctype, "x": cats, "y": series[0]["data"],
           **({"series": series} if len(series) > 1 or ctype == "combo" else {}),
           **({"unit": str(chart["unit"])} if chart.get("unit") else {}),
           **({"unit2": str(chart["unit2"])} if chart.get("unit2") else {}),
           **({"dashboard": kpi} if kpi else {})}   # the user's chart type, in the card's colours
    if why:
        out["chosen_type"], out["fallback_reason"] = chosen, why
    return out


def _blocks(plan: dict, result: dict) -> dict:
    df: pd.DataFrame = result.get("result")
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {"kpis": [], "charts": [], "tables": []}
    display = plan["display"]
    kpis = []
    for alias in [m["alias"] for m in plan["metrics"]]:
        if alias in df.columns:
            value = _number(df.iloc[0][alias])
            kpis.append({"label": _label(plan, alias), "value": value if value is not None else 0.0,
                         "formatted_value": format_number(value) if value is not None else _cell(df.iloc[0][alias]),
                         "unit": None})
    # A one-row answer of a few figures (a home-page module headline) reads as cards, like the
    # dashboard's tiles, even when the planner named none.
    if not kpis and len(df) == 1 and len(df.columns) <= 6:
        for alias in df.columns:
            value = _number(df.iloc[0][alias])
            kpis.append({"label": _label(plan, alias), "value": value if value is not None else 0.0,
                         "formatted_value": format_number(value) if value is not None else _cell(df.iloc[0][alias]),
                         "unit": None})
    # A breakdown with a 'Total' row leads with that row's figures as cards -- the dashboard's
    # headline tiles over its charts (the figures are the query's own, nothing is derived).
    if not kpis and len(df) > 1:
        labels = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
        total = df[df[labels].apply(lambda col: col.astype(str).str.strip().str.lower()
                                    .isin({"total", "grand total"})).any(axis=1)] if labels else df.iloc[0:0]
        if len(total) == 1:
            row = total.iloc[0]
            for alias in [c for c in df.columns if c not in labels and not _RAW_COL_RE.search(str(c))][:4]:
                value = _number(row[alias])
                if value is not None:
                    kpis.append({"label": f"Overall {_label(plan, alias)}" if "total" not in _label(plan, alias).lower()
                                 else _label(plan, alias), "value": value, "formatted_value": format_number(value),
                                 "unit": None})
    # A chart the rows cannot make is left out, never the answer: seen live 27 Oct, 'How can we
    # shrink the Review bucket fastest?' failed in pivot_table and the user got an error.
    try:
        built = _build_chart(plan, df)
    except Exception:  # noqa: BLE001
        log.exception("chart: could not build from %s", list(df.columns))
        built = None
    charts = [built] if built else []
    tables = []
    # A one-row answer with more columns than cards (the Never / Non-comm card has ten figures) keeps
    # its table too, or the figures past the fourth card are never shown. A breakdown always keeps it:
    # the summary gives only the total and points to it (seen live 27 Oct: 'consumers, dtr and feeder
    # installed monthwise' and 'sat progress circle wise' came back as a chart with no table).
    if (display.get("table") or (not kpis and not charts) or len(df) > 1
            or (len(df) == 1 and len(df.columns) > len(kpis))):
        shown = df.head(_TABLE_ROWS)
        cols = [_label(plan, c) for c in shown.columns]
        raw = raw_cols(shown.columns)
        rows = [{cols[i]: _cell(v, i in raw) for i, v in enumerate(r)} for r in shown.itertuples(index=False)]
        total = max(len(df), int(result.get("group_count") or 0))
        tables.append({"id": "primary_table", "title": plan.get("table_title") or "Result", "heading": None,
                       "columns": cols, "rows": rows, "page": 1, "page_size": _TABLE_ROWS, "total_rows": total,
                       "has_next": total > len(shown), "has_previous": False})
    return {"kpis": kpis, "charts": charts, "tables": tables}
