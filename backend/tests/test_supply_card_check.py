"""Supply utilization / MI pace vs stock cards (entity_check): the card's four figures, and warehouse names."""

from gmr_agents.validation.entity_check import check

RATE_ONLY = ("SELECT ROUND(SUM(installed_count) * 100.0 / NULLIF(SUM(inventory_count), 0), 2) AS utilization_pct "
             "FROM public.gold_meter_inventory_event_month_agg WHERE UPPER(oem) = 'AEW' AND period_start <= CURRENT_DATE")
CARD = ("SELECT SUM(inventory_count) AS total_supplies, SUM(installed_count) AS total_installed, "
        "ROUND(SUM(installed_count) * 100.0 / NULLIF(SUM(inventory_count), 0), 2) AS utilization_rate_pct, "
        "SUM(inventory_count) - SUM(stock_consumed_count) AS remaining_stock "
        "FROM public.gold_meter_inventory_event_day_agg WHERE UPPER(oem) = 'AEW' AND period_start <= CURRENT_DATE")
PROJECT_AGEING = ("SELECT SUM(stock_count) FROM public.gold_inventory_unutilized_stock_snapshot "
                  "WHERE LOWER(project) = 'agra'")
WAREHOUSE_AGEING = ("SELECT SUM(stock_count) FROM public.gold_inventory_unutilized_stock_snapshot "
                    "WHERE LOWER(warehouse) = 'agra_hw'")


def card_problems(question, sql):
    return [p for p in check(question, [sql]) if "four figures" in p]


def warehouse_problems(question, sql):
    return [p for p in check(question, [sql]) if "names a warehouse" in p]


def test_rate_alone_is_rejected():
    # live 26 Oct: 'supply utilization rate for OEM AEW' came back as 97.84% alone
    assert card_problems("Supply utilization rate for OEM AEW", RATE_ONLY)
    assert card_problems("MI pace vs stock availability", RATE_ONLY)


def test_card_figures_pass():
    assert not card_problems("Supply utilization rate for OEM AEW", CARD)


def test_trend_questions_are_not_held_to_the_card():
    assert not card_problems("Supply utilization monthwise from Jan 2026 to Aug 2026", RATE_ONLY)


def test_warehouse_name_read_as_project_is_rejected():
    # live 26 Oct: 'Agra HW' answered for project agra (49,307 instead of 37,139)
    assert warehouse_problems("Un-utilized stock ageing for Agra HW", PROJECT_AGEING)
    assert warehouse_problems("Stock ageing for Aligarh Warehouse", PROJECT_AGEING)
    assert not warehouse_problems("Un-utilized stock ageing for Agra HW", WAREHOUSE_AGEING)


def test_generic_warehouse_wording_is_not_a_name():
    assert not warehouse_problems("Stock ageing for each warehouse in agra", PROJECT_AGEING)
    assert not warehouse_problems("Stock ageing for agra project", PROJECT_AGEING)
