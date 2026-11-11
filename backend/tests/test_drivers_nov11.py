"""presentation/drivers.py: where a change came from, per group, computed by code."""
import pandas as pd

from gmr_agents.presentation import drivers

TODAY = pd.Timestamp("2026-11-11")


def long_rows():
    return pd.DataFrame({
        "month": ["2026-09-01"] * 4 + ["2026-10-01"] * 4,
        "zone": ["Varanasi I", "Varanasi II", "Prayagraj I", "Aligarh"] * 2,
        "meters_installed": [40000, 30000, 20000, 10000, 25000, 29000, 21000, 9000],
    })


def test_only_why_questions():
    assert drivers.breakdown("installations by zone last two months", long_rows(), TODAY) == ""
    assert drivers.asked("Why did installations drop in October?")
    assert drivers.asked("installation kyun kam hua october mein")
    assert drivers.asked("What drove the fall in SAT completion?")
    assert not drivers.asked("Which zone has the highest installations?")


def test_long_rows_rank_drivers_by_change_not_size():
    out = drivers.breakdown("Why did installations drop in October?", long_rows(), TODAY)
    assert "Sep 2026 vs Oct 2026" in out
    assert "total: 100,000 -> 84,000, change -16,000 (-16.0%)" in out
    lines = out.splitlines()
    first = next(l for l in lines if l.startswith("- driver:"))
    assert first.startswith("- driver: Varanasi I") and "94% of the total change" in first
    assert any(l.startswith("- moved against the trend: Prayagraj I") for l in lines)


def test_running_month_left_out():
    df = long_rows()
    df = pd.concat([df, pd.DataFrame({"month": ["2026-11-01"] * 4, "zone": df.zone[:4],
                                      "meters_installed": [3000, 2000, 1000, 500]})])
    out = drivers.breakdown("why did installations fall", df, TODAY)
    assert "Sep 2026 vs Oct 2026" in out and "Nov 2026 is still in progress" in out


def test_wide_previous_current_columns():
    df = pd.DataFrame({"discom": ["PUVVNL", "DVVNL", "MVVNL", "Total"],
                       "prev_month_sat": [5000, 3000, 2000, 10000], "this_month_sat": [5200, 1500, 2100, 8800]})
    out = drivers.breakdown("Why did SAT completions go down?", df, TODAY)
    assert "total: 10,000 -> 8,800, change -1,200" in out   # Total row not counted twice
    assert "- driver: DVVNL 3,000 -> 1,500 (-1,500, 125% of the total change)" in out


def test_last_week_vs_week_before_columns():
    df = pd.DataFrame({"category": ["CONSUMER", "FEEDER"], "last_week_closed": [358, 0],
                       "week_before_closed": [1495, 2], "change": [-1137, -2], "pct_change": [-76.05, -100.0]})
    out = drivers.breakdown("What drove the change in tickets closed last week vs the week before?", df, TODAY)
    assert "week_before_closed vs last_week_closed" in out
    assert "- driver: CONSUMER 1,495 -> 358 (-1,137, 100% of the total change)" in out


def test_wide_dated_columns():
    df = pd.DataFrame({"oem": ["A", "B", "C"], "sep_2026": [100, 200, 300], "oct_2026": [90, 260, 300]})
    out = drivers.breakdown("what caused the increase in defects", df, TODAY)
    assert "sep_2026 vs oct_2026" in out and "- driver: B" in out


def test_rates_not_shared_out():
    df = pd.DataFrame({"month": ["2026-09-01"] * 3 + ["2026-10-01"] * 3, "zone": ["A", "B", "C"] * 2,
                       "sat_pct": [80.0, 70.0, 60.0, 79.0, 55.0, 61.0]})
    out = drivers.breakdown("why did sat % fall", df, TODAY)
    assert "is a rate" in out and "of the total change" not in out
    assert "- biggest move: B 70 -> 55 (-15.00 points)" in out


def test_measure_named_in_question_is_used():
    df = long_rows().assign(sat_done=[1, 2, 3, 4, 9, 9, 9, 9])
    out = drivers.breakdown("why did sat done rise", df, TODAY)
    assert "compared: sat_done" in out


def test_group_missing_in_one_period():
    df = pd.DataFrame({"month": ["2026-09-01", "2026-09-01", "2026-10-01", "2026-10-01"],
                       "agency": ["X", "Y", "X", "Z"], "installs": [100, 50, 80, 10]})
    out = drivers.breakdown("why did installs drop", df, TODAY)
    assert "only in Oct 2026: Z" in out and "only in Sep 2026 (no figure in Oct 2026): Y" in out


def test_nothing_to_compare():
    assert drivers.breakdown("why is it low", pd.DataFrame({"zone": ["A", "B"], "n": [1, 2]}), TODAY) == ""
    assert drivers.breakdown("why is it low", None, TODAY) == ""


def test_identical_periods_are_not_no_change():
    df = pd.DataFrame({"project": ["agra", "kashi"], "prev_month_sat": [100, 200], "latest_month_sat": [100, 200]})
    out = drivers.breakdown("why did sat fall last month", df, TODAY)
    assert "no history" in out and "Do NOT say 'no change'" in out


def test_totals_only_says_no_split():
    df = pd.DataFrame({"month": ["2026-09-01", "2026-10-01"], "closed_tickets": [17475, 13065]})
    out = drivers.breakdown("Why did O&M ticket closures fall in October versus September?", df, TODAY)
    assert "no split by place or category" in out
