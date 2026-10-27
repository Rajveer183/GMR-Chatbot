"""entity_check rules added 27 Oct 2026 for mistakes seen live while matching the MI / supply / stock / SAT cards."""

from unittest import mock

from gmr_agents.sql import hierarchy
from gmr_agents.validation.entity_check import check

SAT = "FROM public.gold_mi_sat_invoice_snapshot WHERE LOWER(project) = 'kashi'"


def has(problems, words):
    return any(words in p for p in problems)


def test_eligible_sat_pct_numerator_is_all_sat():
    wrong = ("SELECT ROUND(COALESCE(SUM(meter_count) FILTER (WHERE sat_stage IS NOT NULL AND is_sat_eligible), 0) * 100.0 "
             f"/ NULLIF(SUM(meter_count) FILTER (WHERE is_sat_eligible), 0), 2) {SAT}")
    right = ("SELECT ROUND(COALESCE(SUM(meter_count) FILTER (WHERE sat_stage IS NOT NULL), 0) * 100.0 "
             f"/ NULLIF(SUM(meter_count) FILTER (WHERE is_sat_eligible), 0), 2) {SAT}")
    assert has(check("eligible sat % in kashi", [wrong]), "% SAT vs eligible MI")
    assert not has(check("eligible sat % in kashi", [right]), "% SAT vs eligible MI")


def test_stock_ageing_table_and_stage():
    assert has(check("consumer stock ageing type wise",
                     ["SELECT meter_type, SUM(meter_count) FROM public.gold_non_sat_ageing_snapshot WHERE category = 'CONSUMER' "
                      "GROUP BY meter_type"]), "gold_inventory_unutilized_stock_snapshot")
    no_stage = ("SELECT SUM(stock_count) FROM public.gold_inventory_unutilized_stock_snapshot "
                "WHERE LOWER(project) = 'triveni' AND CURRENT_DATE - inventory_date > 90")
    assert has(check("90+ days stock still in the warehouse in triveni", [no_stage]), "current_stage = 'warehouse'")
    assert has(check("how many meters are lying unused with technicians", [no_stage]), "current_stage = 'technician'")


def test_one_meter_is_looked_up_and_invoice_dates_read():
    assert has(check("is meter CA8081038 eligible for sat",
                     ["SELECT project, eligibility_cutoff_date FROM public.sat_eligibility_config"]), "CA8081038")
    no_dates = "SELECT project FROM gmr_wfm_silver.unified_installations WHERE new_meter_serial_number = 'HP9000628'"
    assert has(check("is meter HP9000628 invoiced", [no_dates]), "invoice_date")


def test_sat_pending_count_uses_the_non_sat_card():
    flags_only = ("SELECT SUM(meter_count) FILTER (WHERE sat_stage IS NULL) AS a, "
                  f"SUM(meter_count) FILTER (WHERE sat_stage IS NULL AND is_sat_eligible) AS b {SAT}")
    card = ("SELECT SUM(meter_count) FILTER (WHERE sat_stage IS NULL), (SELECT SUM(meter_count) FROM "
            f"public.gold_non_sat_ageing_snapshot WHERE LOWER(project) = 'kashi') {SAT}")
    assert has(check("how many meters are sat pending in kashi", [flags_only]), "Non-SAT card")
    assert not has(check("how many meters are sat pending in kashi", [card]), "Non-SAT card")
    # a ranking splits the snapshot's own flags
    assert not has(check("Which project is worst on pending SAT?", [flags_only + " GROUP BY project"]), "Non-SAT card")


def test_oem_named_in_the_users_words():
    sql = "SELECT SUM(meter_count) FROM public.gold_mi_installation_month_agg WHERE LOWER(project) = 'kashi'"
    assert has(check("How many SEPL meters installed in Kashi in August 2026", [sql],
                     asked="how many seipl meters installed in kashi in august 2026"), "OEM 'seipl'")


def test_unknown_place_and_kpi3_scope():
    with mock.patch.object(hierarchy, "place_names", return_value={"edd chunar": "division", "agra": "zone"}):
        p = check("total active days in chunar division last month",
                  ["SELECT COUNT(DISTINCT period_start) FROM public.gold_mi_technician_productivity_day_agg "
                   "WHERE category = 'ALL' AND LOWER(division) = 'chunar'"])
        assert has(p, "'edd chunar'")
        assert not has(check("x", ["SELECT 1 FROM t WHERE LOWER(division) = 'edd chunar'"]), "no stored")
    cube = ("SELECT month_start FROM public.gold_mi_productivity_trend_month_agg WHERE aggregate_level = 'all' "
            "AND category = 'ALL' AND LOWER(project) = 'agra' AND filter_level = 'zone' GROUP BY month_start")
    assert has(check("Month-wise productivity in Agra zone from Jan 2026 to Jun 2026", [cube]), "filter_label")


def test_category_leak_on_the_defect_rate_union():
    leak = ("SELECT SUM(u.f) FROM (SELECT defect_rate_numerator_count AS f, 0 AS i FROM public.gold_meter_defect_event_day_agg "
            "WHERE UPPER(oem) = 'SEIPL' UNION ALL SELECT 0, meter_count FROM public.gold_mi_installation_month_agg "
            "WHERE UPPER(oem) = 'SEIPL' AND progress_category = 'DT') u")
    assert has(check("defect rate for seipl meters in varanasi ii zone", [leak]), "category 'DT'")
    assert not has(check("defect rate for DT seipl meters", [leak]), "category 'DT'")


def test_stage_installs_and_sat_pending_breakdown():
    no_stage = "SELECT SUM(stock_count) FROM public.gold_inventory_unutilized_stock_snapshot WHERE CURRENT_DATE - inventory_date > 90"
    # the rewrite can drop the stage words: the user's own wording still counts
    assert has(check("How many unutilized meters are there?", [no_stage], asked="how many meters are lying unused with technicians"),
               "current_stage = 'technician'")
    staged = ("SELECT warehouse, SUM(stock_count) FROM public.gold_inventory_unutilized_stock_snapshot "
              "WHERE LOWER(project) = 'kashi' AND current_stage = 'warehouse' GROUP BY warehouse")
    assert has(check("stock ageing warehouse wise in kashi", [staged]), "names no stock stage")
    inv = ("SELECT SUM(installed_count) FROM public.gold_meter_inventory_event_day_agg WHERE category = 'DT' "
           "AND LOWER(division) = 'edd chunar'")
    assert has(check("how many dtr installed in chunar division in june 2026", [inv]), "MI progress")
    assert not has(check("supply utilization rate for dtr meters in agra", [inv]), "MI progress")
    flags = ("SELECT category, SUM(meter_count) FILTER (WHERE sat_stage IS NULL), SUM(meter_count) FILTER (WHERE sat_stage IS NULL "
             "AND is_sat_eligible) FROM public.gold_mi_sat_invoice_snapshot WHERE category = 'FEEDER' GROUP BY ROLLUP (category)")
    assert has(check("feeder meters without sat", [flags]), "Non-SAT card")


def test_defect_union_place_on_both_legs_and_upper_labels():
    gap = ("SELECT SUM(u.f) FROM (SELECT defect_rate_numerator_count AS f, 0 AS i FROM public.gold_meter_defect_event_day_agg "
           "WHERE LOWER(zone) = 'varanasi ii' AND UPPER(oem) = 'SEIPL' UNION ALL SELECT 0, meter_count FROM "
           "public.gold_mi_installation_month_agg WHERE UPPER(oem) IN (SELECT DISTINCT UPPER(oem) FROM "
           "public.gold_meter_defect_event_day_agg WHERE LOWER(zone) = 'varanasi ii')) u")
    assert has(check("defect rate for seipl meters in varanasi ii zone", [gap]), "zone = 'varanasi ii'")
    ok = gap.replace("WHERE UPPER(oem) IN", "WHERE LOWER(zone) = 'varanasi ii' AND UPPER(oem) IN")
    assert not has(check("defect rate for seipl meters in varanasi ii zone", [ok]), "installed-meters leg")
    cube = ("SELECT month_start FROM public.gold_mi_productivity_trend_month_agg WHERE aggregate_level = 'all' AND "
            "category = 'ALL' AND project = 'ALL' AND filter_level = 'zone' AND filter_label = '{}' GROUP BY month_start")
    assert has(check("Month-wise productivity in Agra zone", [cube.format("agra")]), "UPPER case")
    assert not has(check("Month-wise productivity in Agra zone", [cube.format("AGRA")]), "UPPER case")
