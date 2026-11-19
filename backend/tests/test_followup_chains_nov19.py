"""19 Nov follow-up chain test (chains 3-13): offline checks for each fix."""
import pandas as pd

from gmr_agents.agents.router import triage
from gmr_agents.agents.sql_writer import planner
from gmr_agents.checks import entity_check
from gmr_agents.execution import hierarchy
from gmr_agents.presentation import analyst, blocks, dashboard_charts, drivers, stats


# --- ROLLUP over a column and its own label: per-group blank echo rows, padded TO_CHAR month names ---------------
def test_echo_subtotals_dropped_and_labels_trimmed():
    sql = "SELECT ... GROUP BY ROLLUP(EXTRACT(MONTH FROM p), TO_CHAR(p,'Month'))"
    df = pd.DataFrame({"month_name": ["Total", None, "May      ", None, "June     "],
                       "a": [300, 100, 100, 200, 200], "b": [3, 1, 1, 2, 2]})
    assert list(hierarchy.relabel(sql, df)["month_name"]) == ["Total", "May", "June"]


def test_lone_blank_total_still_named_total():
    df = pd.DataFrame({"zone": ["A", "B", "C", None], "n": [1, 2, 3, 6]})
    assert list(hierarchy.relabel("x GROUP BY ROLLUP(zone)", df)["zone"]) == ["A", "B", "C", "Total"]


def test_single_blank_row_equal_to_neighbour_kept():
    df = pd.DataFrame({"zone": ["A", "B", None, "C"], "n": [5, 0, 5, 5]})
    assert len(hierarchy.relabel("x GROUP BY ROLLUP(zone)", df)) == 4


# --- periods compared side by side, never stacked ----------------------------------------------------------------
def test_period_series():
    assert dashboard_charts.period_series(["2025 Installs", "2026 Installs"])
    assert dashboard_charts.period_series(["Last Week", "Week Before"])
    assert dashboard_charts.period_series(["This Month", "Last Month"])
    assert not dashboard_charts.period_series(["Consumer 2026", "DT 2026"])   # parts of one period
    assert not dashboard_charts.period_series(["Consumer", "DT"])


def test_stacked_year_comparison_becomes_grouped():
    series = [{"name": "2025 Installs", "data": [1, 2]}, {"name": "2026 Installs", "data": [1, 2]}]
    assert blocks._chart_fallback("stacked_bar", ["May", "June"], series, 0)[0] == "grouped_bar"
    parts = [{"name": "Consumer", "data": [1, 2]}, {"name": "DT", "data": [1, 2]}]
    assert blocks._chart_fallback("stacked_bar", ["May", "June"], parts, 0)[0] == "stacked_bar"


# --- why-questions over rows labelled 'Last Week' / 'Week Before' ------------------------------------------------
def test_labelled_periods_pair():
    long = pd.DataFrame({"period": ["Last Week"] * 3 + ["Week Before"] * 3, "zone": list("ABC") * 2,
                         "installations": [10, 20, 30, 12, 18, 25]})
    got = drivers.pair("why did installations drop last week", long)
    assert got is not None
    both, measure, group, p0, p1, _ = got
    assert (measure, group, p0, p1) == ("installations", "zone", "Week Before", "Last Week")
    assert both["prev"].sum() == 55 and both["curr"].sum() == 60


def test_labelled_periods_ignores_plain_categories():
    other = pd.DataFrame({"cat": ["X", "Y"] * 3, "zone": list("AABBCC"), "n": [1, 2, 3, 4, 5, 6]})
    assert drivers._labelled_periods(other) is other


# --- follow-ups -------------------------------------------------------------------------------------------------
def test_contribution_words_are_not_a_measure():
    assert triage._content_words("Which feeder contributed most?") == set()


def test_dangling_followup_without_history_clarifies():
    for q in ("And for Kashi?", "What about yesterday?", "Same for Triveni."):
        t = {"kind": "data", "question": q, "topics": ["x"], "tables": ["t"]}
        assert triage.clarify_vague({"question": q, "steps": [], "trace_id": "t", "trace": []}, t, []), q
        assert t["kind"] == "chitchat"
    t = {"kind": "data", "question": "And for Kashi?", "topics": ["x"], "tables": ["t"]}
    assert not triage.clarify_vague({"question": "And for Kashi?", "steps": [], "trace_id": "t", "trace": []}, t, [{"question": "SAT %"}])


# --- SQL ------------------------------------------------------------------------------------------------------
def test_total_first_limit_keeps_the_rows_asked_for():
    sql = ("SELECT CASE WHEN GROUPING(zone) = 1 THEN 'Total' ELSE zone END AS zone, SUM(n) AS n FROM t "
           "GROUP BY ROLLUP (zone) ORDER BY GROUPING(zone) DESC, n DESC LIMIT 1")
    assert planner._clean_sql(sql).endswith("LIMIT 2")
    plain = "SELECT zone, SUM(n) FROM t GROUP BY zone ORDER BY 2 DESC LIMIT 1"
    assert planner._clean_sql(plain).endswith("LIMIT 1")
    total_last = "SELECT zone, SUM(n) FROM t GROUP BY ROLLUP (zone) ORDER BY GROUPING(zone), 2 DESC LIMIT 5"
    assert planner._clean_sql(total_last).endswith("LIMIT 5")


def test_per_meter_months_condition_on_zone_groups_flagged():
    bad = ("SELECT zone, COUNT(DISTINCT meter_no) AS m FROM hes_gold.lpf2_low_power_factor_meters WHERE ageing_days >= 3 "
           "GROUP BY zone HAVING COUNT(DISTINCT data_date) >= 3")
    assert any("per meter" in p for p in entity_check.check("which zone has most low pf meters for 3 months", [bad]))
    default_month = ("SELECT zone, COUNT(*) FROM hes_gold.lpf2_low_power_factor_meters WHERE data_date = (SELECT "
                     "MAX(month_date) FROM (SELECT month_date FROM hes_gold.dt_load_profile WHERE month_date >= DATE "
                     "'2026-06-01' GROUP BY month_date HAVING COUNT(DISTINCT data_date) >= 10) c) GROUP BY zone")
    assert not any("per meter" in p for p in entity_check.check("low pf meters by zone", [default_month]))


# --- stats: the query's own projected / partial rows are not history ---------------------------------------------
def test_projected_rows_left_out_of_forecast():
    df = pd.DataFrame({"month": ["2026-04-01", "2026-05-01", "2026-06-01", "2026-07-01", "2026-08-01", "2026-09-01"],
                       "low_pf_meters": [100.0, 110.0, 120.0, 130.0, 60.0, 500.0],
                       "basis": ["actual"] * 4 + ["actual (partial, 2-3 days)", "projected (trend)"]})
    text = stats.analyse("forecast low pf meters next month", df, today=pd.Timestamp("2026-11-19"))
    assert "500" not in text and "4 months" in text or "4 month" in text


# --- the date window note knows any *_date column ---------------------------------------------------------------
def test_date_window_column_names(monkeypatch):
    sent = []

    def fake(sql):
        sent.append(sql)
        return pd.DataFrame({"lo": [pd.Timestamp("2026-10-19")], "hi": [pd.Timestamp("2026-11-18")]})

    monkeypatch.setattr(analyst.executor, "_run_postgres", fake)
    sql = ("SELECT SUM(closed_tickets) FROM public.gold_om_closed_tickets_day_agg WHERE closed_date >= CURRENT_DATE - 31 "
           "AND closed_date <= CURRENT_DATE - 1")
    note = analyst._date_window({"sql": sql, "data_engine": "postgres"}, None)
    assert "19 Oct 2026 .. 18 Nov 2026" in note and "ALL TIME" not in note


def test_double_year_shift_flagged():
    bad = ("WITH c AS (SELECT period_start AS month, SUM(meter_count) AS n FROM public.gold_mi_installation_month_agg "
           "GROUP BY 1), p AS (SELECT (period_start + INTERVAL '1 year')::date AS month, SUM(meter_count) AS n_ly FROM "
           "public.gold_mi_installation_month_agg GROUP BY 1) SELECT c.month, c.n, COALESCE(p.n_ly, 0) FROM c LEFT JOIN p "
           "ON p.month = (c.month + INTERVAL '1 year')::date")
    assert any("shifted by a year twice" in p for p in entity_check.check("installs vs last year", [bad]))
    good = bad.replace("ON p.month = (c.month + INTERVAL '1 year')::date", "ON p.month = c.month")
    assert not any("shifted by a year twice" in p for p in entity_check.check("installs vs last year", [good]))
