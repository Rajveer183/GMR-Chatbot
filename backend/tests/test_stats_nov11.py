"""presentation/stats.py: analysis figures computed by code for analytical questions (11 Nov 2026)."""
import numpy as np
import pandas as pd

from gmr_agents.presentation import stats


def test_kpi_names_do_not_switch_analysis_on():
    # KPI / place words that look like analysis words: no STATS block, no planner hint (no extra tokens).
    for q in ["Voltage deviation index by circle", "list distribution transformer meters", "Varanasi cluster installs",
              "which zone is lagging on SAT", "consumers by overload risk bucket", "how many more SAT to reach 80%",
              "median low PF % by circle", "How many meters were installed yesterday?"]:
        assert stats.wanted(q) == set(), q
        assert stats.analyse(q, pd.DataFrame({"zone": ["a", "b", "c"], "n": [1, 2, 3]})) == ""


def test_analysis_words_detected():
    assert "corr" in stats.wanted("Is there a correlation between DT loading and power factor?")
    assert "pca" in stats.wanted("Run PCA on zone-level KPIs")
    assert {"sim", "forecast"} <= stats.wanted("Run a Monte Carlo simulation of next month's installations")
    assert "tiers" in stats.wanted("Cluster circles by feeder loss")
    assert "pareto" in stats.wanted("Which 20% of zones cause 80% of the backlog?")
    assert "change" in stats.wanted("Detect regime shifts in monthly installations")


def test_methods_not_implemented_are_declared_not_run():
    out = stats.analyse("Show SHAP values and a Granger causality test", pd.DataFrame({"x": ["a"], "n": [1]}))
    assert "NOT RUN" in out and "SHAP" in out and "Granger" in out


def test_time_series_correlation_and_partial_day():
    d = pd.date_range("2026-09-01", "2026-11-11")
    y = 3000 + np.arange(len(d)) * 10.0
    tech = 1000 + np.arange(len(d)) * 2.0
    df = pd.DataFrame({"install_date": d.date.astype(str), "meters": y, "techs": tech})
    out = stats.analyse("Is there a correlation between technicians and installs? forecast next month",
                        df, today=pd.Timestamp("2026-11-11"))
    assert "2026-11-11 is still in progress" in out
    assert "Pearson r 1 " in out or "Pearson r 1.0" in out or "Pearson r 1 (" in out
    assert "next 30 days total" in out
    # Projection / simulation only for the headline measure.
    assert out.count("projection (straight-line") == 1


def test_breakdown_excludes_total_row_and_finds_outlier():
    df = pd.DataFrame({"zone": ["Total"] + [f"Z{i}" for i in range(10)],
                       "defect_rate_pct": [5.0] + [1.0, 1.1, 0.9, 1.05, 0.95, 1.0, 1.02, 0.98, 1.01, 4.0]})
    out = stats.analyse("Which zones have unusually high defect rates?", df)
    assert "across 10 items" in out
    assert "Z9 4" in out


def test_pca_and_monte_carlo_are_deterministic():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"zone": [f"Z{i}" for i in range(12)], "a": rng.normal(10, 2, 12),
                       "b": rng.normal(5, 1, 12), "c": rng.normal(1, .2, 12)})
    q = "Run PCA on zone KPIs"
    assert "PCA over 3 standardised measures" in stats.analyse(q, df)
    ts = pd.DataFrame({"month": [f"2026-0{m}-01" for m in range(1, 10)], "installs": [5, 7, 6, 8, 9, 8, 10, 11, 12]})
    q2 = "Monte Carlo simulation of installations next month"
    assert stats.analyse(q2, ts, today=pd.Timestamp("2026-11-11")) == stats.analyse(q2, ts, today=pd.Timestamp("2026-11-11"))


def test_whatif_change_applied_by_code():
    assert stats.scenario("If technician productivity rises by 20%, how many more meters per month?") == 0.2
    assert stats.scenario("Simulate installations if 10% of technicians are absent") == -0.1
    assert stats.scenario("Stress test: what if supply drops by 50% next month?") == -0.5
    assert stats.scenario("SAT % by December if SAT pace doubles") == 1.0
    assert stats.scenario("When will Triveni reach 90% SAT?") is None
    out = stats.analyse("If productivity rises by 20%, how many more meters per month?",
                        pd.DataFrame({"installs_last_30_days": [100244]}))
    assert "100,244 -> 120,293 (+20,049)" in out


def test_added_technicians_use_productivity():
    assert stats.technicians_added("What if we add 50 technicians in Kashi, how many more installations per day?") == 50
    assert stats.technicians_added("if 30 technicians leave") == -30
    assert stats.technicians_added("how many technicians do we need") is None
    out = stats.analyse("What if we add 50 technicians in Kashi?",
                        pd.DataFrame({"installs": [40000], "tech_days": [12000], "productivity": [3.33]}))
    assert "+166.5 installs per day" in out and "+4,995 per 30 days" in out


def test_cross_system_correlation_lines_up_places():
    wfm = pd.DataFrame({"zone": ["agra", "aligarh", "etah", "mathura", "firozabad"], "non_comm_tickets": [50, 20, 35, 10, 40]})
    mdm = pd.DataFrame({"zone": ["AGRA", "ALIGARH", "ETAH", "MATHURA", "FIROZABAD"], "silent_meters": [500, 210, 330, 90, 420]})
    out = stats.analyse("Do zones with more non-comm tickets show more meters stopped communicating in MDM?", wfm, extras=[mdm])
    assert "across 5 zone values found in both" in out and "very strong positive" in out


def test_variance_split_two_labels():
    rows = [(z, m, base + bump) for z, base in (("A", 100), ("B", 200), ("C", 300)) for m, bump in (("2026-08", 0), ("2026-09", 10), ("2026-10", 20))]
    df = pd.DataFrame(rows, columns=["zone", "month", "installs"])
    out = stats.analyse("How much of the variation in installations comes from differences between zones vs between months?", df)
    assert "differences between zone explain 99" in out and "differences between month explain" in out


def test_stats_figures_pass_the_number_check():
    from gmr_agents.checks import number_check
    block = stats.analyse("Run a Monte Carlo simulation of next month installations",
                          pd.DataFrame({"d": pd.date_range("2026-08-01", periods=60).date.astype(str),
                                        "n": [3000 + (i % 7) * 100 for i in range(60)]}), today=pd.Timestamp("2026-11-11"))
    import re
    p10 = re.search(r"P10 ([\d,.]+)", block).group(1)
    allowed = number_check.allowed_values([], block)
    assert not number_check.unsupported(f"The 30-day P10 is {p10.strip(' ,.')} meters.", allowed)


def test_projection_horizon_follows_question():
    assert stats.horizon("Project installations for the next 6 months", "month") == 6
    assert stats.horizon("forecast the next 3 months", "day") == 90
    assert stats.horizon("next month", "month") is None
    ts = pd.DataFrame({"month": [f"2026-{m:02d}-01" for m in range(1, 11)], "n": [70000 + (m % 3) * 9000 for m in range(1, 11)]})
    out = stats.analyse("Project installations for the next 6 months if the trend continues", ts, today=pd.Timestamp("2026-11-11"))
    assert "next 6 months total" in out


def test_multi_horizon_forecast_gives_each_total():
    assert stats.horizons("Forecast installations for 1 week, 1 month and 3 months ahead") == [("1 week", 7), ("1 month", 30), ("3 months", 90)]
    ts = pd.DataFrame({"install_date": pd.date_range("2026-08-13", "2026-11-10").date.astype(str), "n": [2990, 3010] * 45})
    out = stats.analyse("Forecast installations for 1 week, 1 month and 3 months ahead", ts, today=pd.Timestamp("2026-11-11"))
    assert "next 1 week about 21,000" in out and "next 3 months about 270,000" in out
