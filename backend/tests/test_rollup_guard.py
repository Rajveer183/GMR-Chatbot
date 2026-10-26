"""MI productivity roll-up level check (sql_guard._check_rollup_level), with SQL the planner wrote live."""

import sqlglot

from gmr_agents.validation.sql_guard import _check_rollup_level

T = "public.gold_mi_technician_productivity_day_agg"
M = "public.gold_mi_technician_productivity_month_agg"


def check(sql):
    return _check_rollup_level(sqlglot.parse_one(sql, dialect="postgres"))


def test_double_count_patterns_are_rejected():
    # category <> 'ALL' alone (the old rule) -- 2x installs
    assert "twice" in check(f"SELECT SUM(installation_count) FROM {T} WHERE category <> 'ALL' AND project = 'kashi'")
    # one category without its roll-up row (k09, 23 Oct)
    assert "twice" in check(f"SELECT SUM(installation_count) FROM {T} WHERE category = 'DT' AND LOWER(project) = 'kashi'")
    # no category filter (k19) -- 3x
    assert "three times" in check(f"SELECT SUM(installation_count) FROM {M} WHERE project = 'triveni'")


def test_two_levels_summed_together_is_rejected():
    # k06 (23 Oct): both levels summed per zone
    assert "twice" in check(f"SELECT zone, SUM(installation_count) FROM {M} WHERE (category = 'ALL' OR meter_type = 'ALL') "
                            "GROUP BY ROLLUP (zone)")
    # one row per level is fine
    assert check(f"SELECT category, SUM(installation_count) FROM {M} WHERE (category = 'ALL' OR meter_type = 'ALL') "
                 "GROUP BY category") is None


def test_all_with_meter_type_all_is_rejected():
    # k06 / k15: category 'ALL' rows have meter_type NULL -> zero rows
    assert "no rows" in check(f"SELECT zone, SUM(installation_count) FROM {M} WHERE category = 'ALL' "
                              "AND meter_type = 'ALL' GROUP BY zone")


def test_correct_levels_pass():
    assert check(f"SELECT SUM(installation_count) FROM {T} WHERE category = 'ALL' AND LOWER(zone) = 'agra'") is None
    assert check(f"SELECT SUM(installation_count) FROM {T} WHERE category = 'FEEDER' AND meter_type = 'ALL'") is None
    assert check(f"SELECT SUM(installation_count) FROM {M} WHERE category <> 'ALL' "
                 "AND meter_type = '1PH-Consumer_meter'") is None
    assert check(f"SELECT meter_type, SUM(installation_count) FROM {M} WHERE category = 'CONSUMER' "
                 "AND meter_type IS NOT NULL GROUP BY meter_type") is None
    # two levels on purpose, each measure FILTERed to its level
    assert check(f"SELECT SUM(installation_count) FILTER (WHERE category = 'ALL') FROM {T} "
                 "WHERE (category = 'ALL' OR meter_type = 'ALL')") is None
    # distinct counts only: duplicates do not change them
    assert check(f"SELECT COUNT(DISTINCT technician_name) FROM {T} WHERE category <> 'ALL'") is None
    # inner select picks the level, outer sums
    assert check(f"SELECT SUM(n) FROM (SELECT installation_count AS n FROM {M} WHERE category = 'ALL') t") is None


def test_other_tables_untouched():
    # O&M productivity has no meter_type roll-ups: category <> 'ALL' is right there
    assert check("SELECT SUM(complaint_count) FROM public.gold_om_technician_productivity_day_agg "
                 "WHERE category <> 'ALL'") is None


def test_impossible_category_meter_type_pair_is_rejected():
    assert "no rows" in check(f"SELECT SUM(installation_count) FROM {T} WHERE category = 'DT' AND meter_type = '3PH-Consumer_meter'")
    assert "no rows" in check(f"SELECT SUM(installation_count) FROM {T} WHERE category = 'FEEDER' AND meter_type = '1PH-Consumer_meter'")
    assert check(f"SELECT SUM(installation_count) FROM {T} WHERE category = 'DT' AND meter_type = 'LTCT-Consumer_meter'") is None
    assert check(f"SELECT SUM(installation_count) FROM {T} WHERE category <> 'ALL' AND meter_type = '3PH-Consumer_meter'") is None


def test_trend_cube_needs_a_project_filter():
    cube = "public.gold_mi_productivity_trend_month_agg"
    assert "twice" in check(f"SELECT label, SUM(total_installations) FROM {cube} WHERE aggregate_level = 'discom' AND filter_level = 'all' AND category = 'ALL' GROUP BY label")
    assert check(f"SELECT label, SUM(total_installations) FROM {cube} WHERE project = 'ALL' AND aggregate_level = 'discom' AND filter_level = 'all' AND category = 'ALL' GROUP BY label") is None


def test_trend_cube_needs_both_scope_filters():
    cube = "public.gold_mi_productivity_trend_month_agg"
    # circle wise inside a zone without filter_level (26 Oct draft): repeated per filter scope
    assert "filter_level" in check(f"SELECT label, SUM(total_installations) FROM {cube} WHERE project = 'ALL' "
                                   "AND aggregate_level = 'circle' AND filter_label = 'AZAMGARH' AND category = 'ALL' GROUP BY label")
    assert "aggregate_level" in check(f"SELECT SUM(total_installations) FROM {cube} WHERE project = 'ALL' "
                                      "AND filter_level = 'all' AND category = 'ALL'")
    assert check(f"SELECT label, SUM(total_installations) FROM {cube} WHERE project = 'ALL' AND aggregate_level = 'circle' "
                 "AND filter_level = 'zone' AND filter_label = 'AZAMGARH' AND category = 'ALL' GROUP BY label") is None
    assert check(f"SELECT category, SUM(total_installations) FROM {cube} WHERE aggregate_level = 'all' AND category <> 'ALL' "
                 "AND meter_type = 'ALL' AND LOWER(project) = 'kashi' AND filter_level = 'all' GROUP BY category") is None


def test_summing_per_category_rollups_is_rejected():
    assert "two technician-days" in check(f"SELECT SUM(installation_count) FROM {T} WHERE category <> 'ALL' AND meter_type = 'ALL' AND LOWER(division) = 'x'")
    assert check(f"SELECT category, SUM(installation_count) FROM {T} WHERE category <> 'ALL' AND meter_type = 'ALL' GROUP BY category") is None
