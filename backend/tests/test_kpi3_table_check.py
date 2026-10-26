"""KPI 2 vs KPI 3 table choice (entity_check): the question's words decide the productivity table."""

from gmr_agents.validation.entity_check import check

CUBE = ("SELECT label, SUM(total_installations) FROM public.gold_mi_productivity_trend_month_agg "
        "WHERE aggregate_level = 'circle' AND project = 'ALL' AND filter_level = 'all' AND category = 'ALL' GROUP BY label")
TECH = ("SELECT period_start, SUM(installation_count) FROM public.gold_mi_technician_productivity_month_agg "
        "WHERE category = 'ALL' GROUP BY period_start")


def kpi3_problems(question, sql):
    return [p for p in check(question, [sql]) if "Monthly Productivity trend card" in p]


def test_kpi2_question_on_the_trend_cube_is_rejected():
    # live 26 Oct: answered from the cube, 3.23 instead of the KPI 2 card's 2.91
    assert kpi3_problems("Productivity circle wise in Kashi in September 2026", CUBE)


def test_kpi3_question_on_the_technician_table_is_rejected():
    assert kpi3_problems("Monthly productivity trend project wise from July 2026 to September 2026", TECH)
    assert kpi3_problems("Show the KPI 3 card", TECH)


def test_matching_tables_pass():
    assert not kpi3_problems("Monthly productivity trend for Kashi", CUBE)
    assert not kpi3_problems("Average active technicians per month in 2026", CUBE)
    assert not kpi3_problems("MI productivity in Triveni month wise", TECH)
    # the card is monthly only: a daily / weekly trend is KPI 2's
    assert not kpi3_problems("daily productivity trend", TECH)


def test_sum_active_days_is_allowed_on_the_cube_only():
    sum_days = "SELECT SUM(active_days) FROM {} WHERE category = 'ALL'"
    on_cube = sum_days.format("public.gold_mi_productivity_trend_month_agg")
    on_tech = sum_days.format("public.gold_mi_technician_productivity_day_agg")
    assert not [p for p in check("monthly productivity trend active days", [on_cube]) if "SUM(active_days)" in p]
    assert [p for p in check("active days in Kashi", [on_tech]) if "SUM(active_days)" in p]
