"""Dashboard KPI charts (dashboard_charts.py): a question a WFM dashboard card answers is drawn
the way that card draws it; anything else keeps the planner's chart."""

import pandas as pd

from gmr_agents import dashboard_charts as dc
from gmr_agents.llm_pipeline import _build_chart

MI_BY_PROJECT = ("SELECT project, SUM(meter_count) FILTER (WHERE progress_category = 'CONSUMER') AS consumer, "
                 "SUM(meter_count) FILTER (WHERE progress_category = 'DT') AS dtr, "
                 "SUM(meter_count) FILTER (WHERE progress_category = 'FEEDER') AS feeder "
                 "FROM public.gold_mi_installation_month_agg WHERE NOT is_replacement GROUP BY project")


def plan(sql, chart, question="", labels=None):
    return {"sql": sql, "question": question, "labels": labels or {}, "table_title": "t",
            "display": {"chart": chart}, "metrics": []}


def s(name, data):
    return {"name": name, "data": list(data)}


def test_kpi_from_gold_table():
    assert dc.kpi_for(MI_BY_PROJECT, ["Consumer"]) == "mi_progress"
    assert dc.kpi_for("SELECT 1 FROM public.gold_non_sat_ageing_snapshot", []) == "non_sat_ageing"
    assert dc.kpi_for("SELECT 1 FROM public.gold_meter_inventory_event_day_agg", ["Supplied", "Installed"]) == "supply_utilization"
    assert dc.kpi_for("SELECT 1 FROM public.gold_meter_inventory_event_day_agg", ["Remaining Stock"]) == "mi_pace_stock"
    # live: the utilization question also returned remaining stock -- still the utilization card
    assert dc.kpi_for("SELECT 1 FROM public.gold_meter_inventory_event_day_agg", ["Supplied", "Remaining Stock"],
                      "Supply utilization rate by project") == "supply_utilization"
    assert dc.kpi_for("SELECT 1 FROM public.gold_meter_inventory_event_day_agg", ["Supplied"],
                      "MI pace vs stock availability by zone") == "mi_pace_stock"
    assert dc.kpi_for("SELECT 1 FROM public.gold_mi_sat_invoice_snapshot", ["Total MI", "Total SAT"]) == "mi_vs_sat"
    assert dc.kpi_for("SELECT 1 FROM public.gold_mi_sat_invoice_snapshot", ["Lumpsum Invoiced"]) == "sat_invoice_funnel"


def test_other_tables_keep_the_planners_chart():
    assert dc.kpi_for("SELECT 1 FROM gmr_wfm_silver.unified_installations", ["x"]) is None
    assert dc.kpi_for("SELECT 1 FROM public.gold_dashboard_stats_snapshot", ["x"]) is None
    assert dc.kpi_for(None, []) is None


def test_first_table_named_wins():
    sql = ("SELECT 1 FROM public.gold_non_sat_ageing_snapshot n "
           "JOIN public.gold_mi_installation_month_agg m ON TRUE")
    assert dc.kpi_for(sql, []) == "non_sat_ageing"


def test_parts_stack_like_the_dashboard():
    # the planner's grouped bars become the dashboard's one stack per project, total on top
    out = dc.shape("mi_progress", "grouped_bar", ["AGRA", "KASHI", "TRIVENI"],
                   [s("Consumer", [3, 2, 1]), s("Dtr", [1, 1, 1]), s("Feeder", [0, 1, 0])], "project")
    assert out["type"] == "stacked_bar" and out["labels"] == "total"


def test_more_than_eight_categories_go_horizontal():
    cats = [f"DIVISION {i}" for i in range(12)]
    assert dc.shape("non_sat_ageing", "bar", cats, [s("120+ days", range(12))], "division")["type"] == "horizontal_bar"
    out = dc.shape("non_sat_ageing", "stacked_bar", cats, [s("91-120", range(12)), s("120+", range(12))], "division")
    assert out["type"] == "stacked_horizontal_bar"


def test_dates_are_columns_then_area_past_25_points():
    months = [f"2026-{m:02d}" for m in range(1, 11)]
    assert dc.shape("meter_replacement", "line", months, [s("Burnt", range(10))], "month")["type"] == "bar"
    days = [f"2026-09-{d:02d}" for d in range(1, 31)]
    assert dc.shape("mi_progress", "bar", days, [s("Meters", range(30))], "day")["type"] == "area"


def test_rates_leave_the_bars():
    out = dc.shape("supply_utilization", "combo", ["AGRA", "KASHI"],
                   [s("Installed", [500, 400]), {**s("Utilization Rate", [97.2, 96.9]), "type": "line"}], "project")
    assert [x["name"] for x in out["series"]] == ["Installed"]
    assert [x["name"] for x in out["rates"]] == ["Utilization Rate"] and "type" not in out["rates"][0]


def test_a_productivity_card_plots_productivity():
    # live: installs were the bars and productivity only a label on top
    out = dc.shape("mi_productivity", "combo", ["AGRA", "KASHI"],
                   [s("Installations", [500, 400]), {**s("Productivity", [3.2, 2.9]), "type": "line"}], "project")
    assert [x["name"] for x in out["series"]] == ["Productivity"] and out["rates"] == []
    assert out["type"] == "bar" and out["labels"] == "segments"   # a lone bar keeps its value


def test_age_bands_youngest_first_with_readable_names():
    # live: legend 'age_120_plus', 'age_91_120' with the oldest band at the bottom
    out = dc.shape("non_sat_ageing", "stacked_bar", ["AGRA", "KASHI"],
                   [s("age_120_plus", [92, 71]), s("age_91_120", [4, 3])], "project")
    assert [x["name"] for x in out["series"]] == ["91-120 days", "120+ days"]
    assert [x["data"] for x in out["series"]] == [[4, 3], [92, 71]]
    named = dc.shape("stock_ageing", "stacked_bar", ["W1", "W2"],
                     [s("days_0_30", [1, 2]), s("31-60 Days", [1, 2]), s("90+ days", [1, 2])], "warehouse")
    assert [x["name"] for x in named["series"]] == ["0-30 days", "31-60 days", "90+ days"]


def test_productivity_column_is_the_bar_even_when_the_planner_left_it_out():
    # live: installations stacked on active technicians; productivity only in the table
    sql = "SELECT project, productivity_per_tech_day, installations, active_technicians FROM public.gold_mi_technician_productivity_day_agg"
    df = pd.DataFrame({"project": ["KASHI", "AGRA"], "productivity_per_tech_day": [3.36, 3.21],
                       "installations": [44207, 34092], "active_technicians": [1204, 767]})
    out = _build_chart(plan(sql, {"type": "stacked_bar", "x": "project", "y": ["installations", "active_technicians"]}), df)
    assert out["type"] == "bar" and out["y"] == [3.36, 3.21] and "series" not in out


def test_repeated_categories_keep_their_subtotal_row():
    # live: MI progress by project drew AGRA four times (consumer, DT, feeder and subtotal rows)
    cats = ["AGRA", "AGRA", "AGRA", "AGRA", "KASHI", "KASHI"]
    out = dc.shape("mi_progress", "stacked_bar", cats,
                   [s("Consumer", [16, 0, 0, 16, 13, 13]), s("DT", [0, 4, 0, 4, 0, 0]), s("Feeder", [0, 0, 1, 1, 0, 0])],
                   "project")
    assert out["x"] == ["AGRA", "KASHI"] and [x["data"] for x in out["series"]] == [[16, 13], [4, 0], [1, 0]]


def test_a_series_that_is_the_sum_of_the_others_is_dropped():
    # live: daily installs drew 'Meters Installed' over Consumer / DT / Feeder
    days = [f"2026-09-{d:02d}" for d in range(1, 31)]
    out = dc.shape("mi_progress", "area", days, [s("Consumer", [5] * 30), s("DT", [2] * 30), s("Feeder", [1] * 30),
                                                 s("Meters Installed", [8] * 30)], "day")
    assert [x["name"] for x in out["series"]] == ["Consumer", "DT", "Feeder"] and out["type"] == "area"


def test_total_row_leads_as_cards():
    from gmr_agents.llm_pipeline import _blocks
    sql = "SELECT project, productivity, technicians FROM public.gold_mi_technician_productivity_day_agg"
    df = pd.DataFrame({"project": ["KASHI", "AGRA", "Total"], "productivity": [3.36, 3.22, 3.19],
                       "technicians": [1202, 763, 2808]})
    p = {**plan(sql, {"type": "bar", "x": "project", "y": ["productivity"]}), "metrics": [],
         "labels": {"productivity": "Productivity", "technicians": "Active Technicians"}}
    out = _blocks(p, {"result": df})
    assert [(k["label"], k["value"]) for k in out["kpis"]] == [("Overall Productivity", 3.19),
                                                              ("Overall Active Technicians", 2808.0)]


def test_a_total_is_never_stacked_on_its_parts():
    out = dc.shape("mi_vs_sat", "stacked_bar", ["AGRA", "KASHI"],
                   [s("Total MI", [10, 9]), s("Eligible MI", [9, 8]), s("Total SAT", [8, 7]), s("SAT % vs MI", [80, 77])],
                   "project")
    assert out["type"] == "grouped_bar" and [r["name"] for r in out["rates"]] == ["SAT % vs MI"]


def test_complaint_types_are_a_donut_of_ten_slices():
    cats = [f"type {i}" for i in range(14)]
    out = dc.shape("om_active_ageing", "horizontal_bar", cats, [s("Open tickets", range(14, 0, -1))], "complaint_type")
    assert out["type"] == "donut" and len(out["x"]) == 10 and out["x"][-1] == "Others (5 types)"
    assert sum(out["series"][0]["data"]) == sum(range(14, 0, -1))
    # by project it is the dashboard's comparison chart, not a donut
    assert dc.shape("om_active_ageing", "bar", ["AGRA", "KASHI"], [s("Open", [5, 4])], "project")["type"] == "bar"


def test_named_chart_type_wins():
    assert dc.user_named_type("Give me a pie chart of defects by type")
    assert dc.user_named_type("show monthly installs as a line")
    assert not dc.user_named_type("Monthly burnt meters over the last 12 months")


def test_build_chart_tags_the_dashboard_card():
    df = pd.DataFrame({"project": ["AGRA", "KASHI"], "consumer": [10, 8], "dtr": [2, 1], "feeder": [1, 0]})
    out = _build_chart(plan(MI_BY_PROJECT, {"type": "grouped_bar", "x": "project", "y": ["consumer", "dtr", "feeder"]}), df)
    assert out["dashboard"] == "mi_progress" and out["type"] == "stacked_bar" and out["labels"] == "total"
    assert out["chosen_type"] == "grouped_bar"


def test_build_chart_keeps_a_named_type_in_the_cards_colours():
    df = pd.DataFrame({"project": ["AGRA", "KASHI"], "consumer": [10, 8], "dtr": [2, 1], "feeder": [1, 0]})
    out = _build_chart(plan(MI_BY_PROJECT, {"type": "grouped_bar", "x": "project", "y": ["consumer", "dtr", "feeder"]},
                            question="Show MI by project as a grouped bar chart"), df)
    assert out["type"] == "grouped_bar" and out["dashboard"] == "mi_progress" and "labels" not in out


def test_card_rows_get_the_cards_chart_when_the_planner_drew_none():
    # live: 'O&M active ticket ageing by project in age bands' came back as a table only
    sql = "SELECT project, SUM(ticket_count) AS total_open_tickets FROM public.gold_om_open_ageing_snapshot"
    df = pd.DataFrame({"project": ["Total", "AGRA", "KASHI"], "total_open_tickets": [30, 20, 10], "sat_pending": [3, 2, 1],
                       "days_0_7": [9, 6, 3], "days_8_15": [12, 8, 4], "days_60_plus": [9, 6, 3]})
    out = _build_chart(plan(sql, {"type": "none"}), df)
    assert out["dashboard"] == "om_active_ageing" and out["type"] == "stacked_bar"
    assert [s["name"] for s in out["series"]] == ["Days 0 7", "Days 8 15", "Days 60 Plus"] and out["x"] == ["AGRA", "KASHI"]
    # a one-row answer stays as cards
    assert _build_chart(plan(sql, {"type": "none"}), df.head(1)) is None


def test_long_rows_split_by_their_second_label():
    # live: non-SAT ageing by project in age bands drew nine single bars
    sql = "SELECT project, age_key, SUM(meter_count) AS non_sat_meters FROM public.gold_non_sat_ageing_snapshot"
    df = pd.DataFrame({"project": ["AGRA", "AGRA", "KASHI", "KASHI"], "age_key": ["age_91_120", "age_120_plus"] * 2,
                       "non_sat_meters": [5, 100, 2, 80]})
    out = _build_chart(plan(sql, {"type": "horizontal_bar", "x": "project", "y": ["non_sat_meters"]}), df)
    assert out["type"] == "stacked_bar" and out["x"] == ["AGRA", "KASHI"]
    assert [s["name"] for s in out["series"]] == ["91-120 days", "120+ days"]


def test_a_numeric_group_is_not_a_split():
    # live: journey stages grouped by their sort order drew series '1'..'7'
    sql = "SELECT ord, stage, avg_days FROM public.gold_meter_journey_stage_agg"
    df = pd.DataFrame({"ord": [1, 2, 3], "stage": ["DI to Agency", "Agency to Supervisor", "Supervisor to Technician"],
                       "avg_days": [4.0, 2.5, 1.2]})
    out = _build_chart(plan(sql, {"type": "bar", "x": "stage", "y": ["avg_days"], "group": "ord"}), df)
    assert out["type"] == "bar" and "series" not in out and out["x"] == list(df["stage"])


def test_live_rule_fixes():
    # 'Installs/Day' is a rate; the remark donut; funnel stages side by side
    out = dc.shape("mi_pace_stock", "stacked_bar", [f"Z{i}" for i in range(11)],
                   [s("Remaining Stock", range(11)), s("Installs/Day (6M Avg)", range(11))], "zone")
    assert out["type"] == "horizontal_bar" and [r["name"] for r in out["rates"]] == ["Installs/Day (6M Avg)"]
    assert dc.shape("om_escalated", "horizontal_bar", [f"r{i}" for i in range(15)], [s("Tickets", range(15, 0, -1))],
                    "technician_remark")["type"] == "donut"
    out = dc.shape("sat_invoice_funnel", "stacked_bar", ["AGRA", "KASHI"],
                   [s("Installed meters", [9, 8]), s("SAT meters", [8, 7]), s("Lumpsum invoiced meters", [7, 6])], "project")
    assert out["type"] == "grouped_bar"


def test_build_chart_without_a_dashboard_table_is_unchanged():
    df = pd.DataFrame({"zone": ["A", "B"], "meters": [10, 8]})
    out = _build_chart(plan("SELECT zone, COUNT(*) AS meters FROM gmr_wfm_silver.unified_installations GROUP BY zone",
                            {"type": "bar", "x": "zone", "y": ["meters"]}), df)
    assert out["type"] == "bar" and "dashboard" not in out
