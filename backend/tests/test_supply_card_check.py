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


DROPPED = ("SELECT zone, SUM(meter_count) FROM public.gold_mi_sat_invoice_snapshot WHERE LOWER(project) = 'triveni' "
           "AND meter_type IS NOT NULL AND meter_type != 'OTHER-Consumer_meter' GROUP BY zone")
STAGE = "SELECT project, SUM(meter_count) FROM public.gold_meter_revenue_pending_stage_snapshot GROUP BY project"
AGEING = "SELECT project, SUM(outstanding_count) FROM public.gold_revenue_ageing_snapshot GROUP BY project"


def test_unasked_meter_type_exclusion_is_rejected():
    # live 26 Oct: 'invoice funnel zone wise in Triveni' dropped OTHER / NULL meter types (44 meters)
    assert [p for p in check("Invoice funnel zone wise in Triveni", [DROPPED]) if "drops meter types" in p]
    assert not [p for p in check("Invoice funnel zone wise in Triveni excluding other meters", [DROPPED])
                if "drops meter types" in p]


def test_revenue_ageing_reads_the_ageing_snapshot():
    # live 26 Oct: answered from the current-stage table (1,019,165 instead of 157,979)
    assert [p for p in check("Revenue not realized ageing project wise", [STAGE]) if "ageing reads" in p]
    assert not [p for p in check("Revenue not realized ageing project wise", [AGEING]) if "ageing reads" in p]
    assert not [p for p in check("Revenue not realized meters project wise", [STAGE]) if "ageing reads" in p]


ALL_PROJECTS = ("SELECT SUM(lumpsum_within_timeline) FROM (SELECT r.*, ROW_NUMBER() OVER (PARTITION BY project "
                "ORDER BY created_at DESC) AS rn FROM public.revenue_receivables r) l WHERE rn = 1")
PUVVNL = ALL_PROJECTS + " AND LOWER(project) IN ('kashi', 'triveni')"


def test_discom_on_receivables_becomes_its_projects():
    # live 26 Oct: 'within timeline for PUVVNL' summed Agra (DVVNL) too
    assert [p for p in check("Lumpsum within timeline for PUVVNL", [ALL_PROJECTS]) if "no discom column" in p]
    assert not [p for p in check("Lumpsum within timeline for PUVVNL", [PUVVNL])
                if "no discom column" in p or "does not mention" in p]


def test_om_ticket_sat_status_is_not_a_pending_sat_count():
    sql = "SELECT zone, SUM(ticket_count) FROM public.gold_om_open_ageing_snapshot WHERE sat_status = 'SAT PENDING' GROUP BY zone"
    assert not [p for p in check("Open O&M tickets of SAT pending meters zone wise", [sql]) if "without / pending SAT" in p]
    assert [p for p in check("How many meters are pending SAT?", ["SELECT SUM(meter_count) FROM public.gold_non_sat_ageing_snapshot"])
            if "without / pending SAT" in p]
