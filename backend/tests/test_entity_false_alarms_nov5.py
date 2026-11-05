"""entity_check false alarms fixed 5 Nov 2026: verified, correct SQL was sent back for a repair round
(~6.8k tokens, ~6 s each). Each test keeps the real mistake the rule was written for flagged."""

from gmr_agents.checks.entity_check import check

SAT10 = ("SELECT CASE WHEN GROUPING(category) = 1 THEN 'Total' WHEN category = 'DT' THEN 'DTR' ELSE category END "
         "AS category, SUM(meter_count) AS meters FROM public.gold_mi_sat_invoice_snapshot WHERE sat_stage = 10 "
         "GROUP BY ROLLUP (category)")


def has(problems, words):
    return any(words in p for p in problems)


def test_category_split_grouped_by_category_is_not_a_missing_filter():
    assert not has(check("Split SAT 10 by DT, Consumer and Feeder.", [SAT10]), "never filters/selects category")
    assert not has(check("Of the 120+ day non-SAT meters, how many are DT or Feeder meters?",
                         ["SELECT category, SUM(meter_count) FROM public.gold_non_sat_ageing_snapshot "
                          "WHERE age_key = 'age_120_plus' GROUP BY ROLLUP (category)"]), "never filters/selects category")


def test_one_category_asked_still_needs_its_filter():
    no_filter = "SELECT SUM(meter_count) FROM public.gold_mi_sat_invoice_snapshot WHERE sat_stage = 10"
    assert has(check("How many consumer meters are in SAT 10?", [no_filter]), "category 'CONSUMER'")
    # grouped, but only one category asked: still the one category
    assert has(check("How many consumer meters are in SAT 10?",
                     ["SELECT SUM(meter_count) FROM public.gold_mi_sat_invoice_snapshot WHERE sat_stage = 10 "
                      "GROUP BY sat_stage"]), "category 'CONSUMER'")


def test_sub_division_wise_is_not_division_wise():
    sql = ("SELECT CASE WHEN GROUPING(sub_division) = 1 THEN 'Total' ELSE UPPER(sub_division) END AS sub_division, "
           "SUM(meter_count) FROM public.gold_mi_installation_day_agg WHERE LOWER(division) = 'edd chunar' "
           "GROUP BY ROLLUP (sub_division)")
    assert not has(check("How many meters installed sub division wise in EDD Chunar division in August 2026", [sql]),
                   "division-wise breakdown")
    # a real division-wise ask without the GROUP BY is still caught
    one_row = "SELECT SUM(meter_count) FROM public.gold_mi_installation_day_agg WHERE LOWER(circle) = 'edc agra'"
    assert has(check("installs division wise in EDC Agra circle", [one_row]), "division-wise breakdown")


def test_share_of_a_named_project_from_a_project_breakdown():
    sql = ("SELECT CASE WHEN GROUPING(project) = 1 THEN 'Total' ELSE UPPER(project) END AS project, "
           "SUM(meter_count) AS installed FROM public.gold_mi_installation_month_agg GROUP BY ROLLUP (project)")
    assert not has(check("What share of all installed meters is from Triveni?", [sql]), "names project 'triveni'")
    # not a share, not grouped: the named project must still be filtered
    assert has(check("How many meters installed in Triveni?",
                     ["SELECT SUM(meter_count) FROM public.gold_mi_installation_month_agg"]), "names project 'triveni'")
