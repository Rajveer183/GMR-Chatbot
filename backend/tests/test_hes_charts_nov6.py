"""Technical (HES) boards draw the dashboard's own chart (6 Nov 2026): HES-FRONTEND-GMR voltage boards
(Donut / MeanBars / DirectionCells), Layout A / C and the DT boards (BandColumns / RankBars / CompareColumns)."""

import pandas as pd

from gmr_agents.presentation import dashboard_charts as D

DM = ("(SELECT MAX(month_date) FROM (SELECT month_date FROM hes_gold.dt_load_profile WHERE month_date >= "
      "DATE '2026-06-01' GROUP BY month_date HAVING COUNT(DISTINCT data_date) >= 10) c)")


def _s(name, data):
    return [{"name": name, "data": list(data)}]


def test_tables_map_to_their_board():
    assert D.kpi_for(f"SELECT 1 FROM hes_gold.voltage_deviation_index WHERE data_date = {DM}", []) == "pq_vdi"
    assert D.kpi_for(f"SELECT 1 FROM hes_gold.dt_peak_loading_feeder WHERE data_date = {DM}", []) == "dt_peak_feeder"
    assert D.kpi_for(f"SELECT 1 FROM hes_gold.dt_peak_loading WHERE data_date = {DM}", []) == "dt_peak"
    assert D.kpi_for(f"WITH m AS ({DM}) SELECT 1 FROM hes_gold.load_violation", []) == "load_violation"


def test_status_split_is_the_boards_donut_in_status_order():
    out = D.shape("pq_drop", "bar", ["OK", "Severe", "Offline", "Mild", "Moderate"], _s("meters", [5, 9, 1, 2, 3]), "status")
    assert out["type"] == "donut"
    assert out["point_keys"] == ["Severe", "Moderate", "Mild", "OK"]          # Offline dropped
    assert out["series"][0]["data"] == [9, 3, 2, 5]


def test_loading_bands_are_columns_in_band_order():
    cats = [">120%", "<20%", "20-35%", "100-120%"]
    out = D.shape("dt_peak", "bar", cats, _s("dts", [4, 30, 20, 3]), "peak_band")
    assert (out["type"], out["x"], out["point_keys"]) == ("bar", ["<20%", "20-35%", "100-120%", ">120%"],
                                                           ["<20%", "20-35%", "100-120%", ">120%"])


def test_voltage_area_breakdown_is_grey_mean_bars():
    series = _s("Avg Vdi Pct", [12.3, 11.0]) + _s("Meters", [100, 200])
    out = D.shape("pq_vdi", "bar", ["EDC II JAUNPUR", "EDC GHAZIPUR"], series, "circle")
    assert out["type"] == "horizontal_bar" and out["point_keys"] == ["mean", "mean"]
    assert [s["name"] for s in out["series"]] == ["Avg Vdi Pct"]


def test_load_violation_by_area_highlights_the_highest():
    out = D.shape("load_violation", "bar", ["A", "B", "C"], _s("Flagged Consumers", [5, 9, 7]), "circle")
    assert out["type"] == "bar" and out["x"] == ["B", "C", "A"]
    assert out["point_keys"] == ["highlight", "series", "series"]


def test_dt_list_bars_take_each_dts_band():
    rows = pd.DataFrame({"meter_no": ["AL1", "AL2"], "pct_peak_loading": [150.0, 90.0]})
    out = D.shape("dt_peak", "horizontal_bar", ["AL1", "AL2"], _s("Pct Peak Loading", [150, 90]), "meter_no", rows)
    assert out["point_keys"] == [">120%", "85-100%"]


def test_unknown_shape_keeps_the_planners_chart():
    series = _s("a", [1, 2])
    assert D.shape("pq_vdi", "line", ["2026-06-01", "2026-07-01"], series, "data_month") is None
    assert series[0]["data"] == [1, 2]                                         # untouched for the fallback
