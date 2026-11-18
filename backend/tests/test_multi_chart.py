"""One detailed chart for rows that mix dates and places (presentation/multi_chart.py): exact, or untouched."""
import pandas as pd

from gmr_agents.presentation import blocks

DATES = ["2026-11-08", "2026-11-09", "2026-11-10"]
PROJECTS = ["agra", "kashi", "triveni"]
MI_SQL = "SELECT period_start, project, ... FROM gold_mi_installation_day_agg GROUP BY 1, 2"
MIXED = {"type": "stacked_bar", "x": "install_date", "y": ["consumer", "dt", "feeder"]}


def _rows(**cols):
    data = {"install_date": [d for d in DATES for _ in PROJECTS], "project": PROJECTS * len(DATES)}
    data.update(cols)
    return pd.DataFrame(data)


def _charts(df, chart, sql=MI_SQL, question="q"):
    plan = {"sql": sql, "question": question, "display": {"chart": chart, "table": True},
            "metrics": [], "labels": {}, "table_title": "t"}
    return blocks._blocks(plan, {"result": df})["charts"]


def _total(chart):
    return sum(sum(s["data"]) for s in chart.get("series") or [{"data": chart["y"]}])


def test_mixed_axis_becomes_one_trend_with_place_rows_in_the_hover():
    df = _rows(consumer=[float(i * 10) for i in range(9)], dt=[1.0] * 9, feeder=[0.0] * 8 + [1.0])
    charts = _charts(df, MIXED)
    assert len(charts) == 1
    chart = charts[0]
    assert chart["id"] == "primary_chart" and chart["x"] == DATES and chart["dashboard"] == "mi_progress"
    assert [s["name"] for s in chart["series"]] == ["Consumer", "Dt", "Feeder"]
    assert _total(chart) == float(df[["consumer", "dt", "feeder"]].to_numpy().sum())
    assert chart["details_label"] == "Project"
    # each day's place rows add up to that day's stack
    for j, rows in enumerate(chart["details"]):
        assert sum(v for _p, v in rows) == sum(s["data"][j] for s in chart["series"])
    assert chart["details"][0][0] == ["triveni", 21.0]


def test_single_measure_mixed_axis_stacks_the_places():
    df = _rows(meters_installed=[float(i) for i in range(9)])
    charts = _charts(df, {"type": "bar", "x": "install_date", "y": ["meters_installed"]})
    assert len(charts) == 1 and charts[0]["x"] == DATES
    assert sorted(s["name"] for s in charts[0]["series"]) == PROJECTS
    assert _total(charts[0]) == 36.0 and "details" not in charts[0]


def test_clean_axes_are_left_alone():
    df = _rows(meters_installed=[float(i) for i in range(9)])
    for chart in ({"type": "bar", "x": "project", "y": ["meters_installed"], "group": "install_date"},
                  {"type": "bar", "x": "install_date", "y": ["meters_installed"], "group": "project"}):
        charts = _charts(df, chart)
        assert len(charts) == 1 and not any(" · " in str(x) for x in charts[0]["x"]) and "details" not in charts[0]


def test_top_n_keeps_the_main_chart():
    df = _rows(consumer=[float(i) for i in range(9)], dt=[1.0] * 9, feeder=[0.0] * 9)
    chart = _charts(df, MIXED, sql=MI_SQL + " ORDER BY 3 DESC LIMIT 9")[0]
    assert " · " in chart["x"][0]


def test_executor_row_cap_is_not_a_top_n():
    df = _rows(consumer=[float(i) for i in range(9)], dt=[1.0] * 9, feeder=[0.0] * 9)
    assert _charts(df, MIXED, sql=MI_SQL + " LIMIT 10000")[0]["x"] == DATES


def test_rates_are_never_summed_over_places():
    df = _rows(sat_pct=[90.0] * 9, avg_days=[5.0] * 9)
    for chart in _charts(df, {"type": "bar", "x": "install_date", "y": ["sat_pct"]}):
        assert all(v <= 100 for s in chart.get("series") or [{"data": chart["y"]}] for v in s["data"])


def test_rollup_rows_do_not_double_count():
    df = _rows(consumer=[1.0] * 9, dt=[0.0] * 9, feeder=[0.0] * 9)
    total = pd.DataFrame({"install_date": ["Total"], "project": ["All"], "consumer": [9.0], "dt": [0.0], "feeder": [0.0]})
    chart = _charts(pd.concat([df, total], ignore_index=True), MIXED)[0]
    assert chart["x"] == DATES and _total(chart) == 9.0


def test_card_parts_only_and_all_null_part_is_zero():
    """HHU: tickets = OK + Failed + Other; a week with no failures sends Failed as all NULL (object dtype)."""
    sql = "SELECT period_start, project, ... FROM gold_om_hhu_reading_day_agg GROUP BY 1, 2"
    df = _rows(tickets_with_hhu_upload=[5.0] * 9, upload_ok=[5.0] * 9, upload_failed=[None] * 9,
               upload_other=[None] * 9, upload_attempts=[6.0] * 9)
    chart = _charts(df, {"type": "stacked_bar", "x": "install_date", "y": ["upload_ok", "upload_failed", "upload_other"]},
                    sql=sql)[0]
    assert chart["x"] == DATES and _total(chart) == 45.0
    assert [s["name"] for s in chart["series"]] == ["Upload Ok", "Upload Failed", "Upload Other"]


def test_no_date_column_keeps_one_chart():
    df = pd.DataFrame({"project": PROJECTS, "meters_installed": [3.0, 2.0, 1.0]})
    assert len(_charts(df, {"type": "bar", "x": "project", "y": ["meters_installed"]})) == 1
