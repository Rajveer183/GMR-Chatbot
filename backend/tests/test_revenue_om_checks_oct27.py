"""entity_check rules added 27 Oct 2026 for mistakes seen live on the journey, current-stage, receivables,
revenue-ageing and O&M productivity / closed cards."""

from gmr_agents.validation.entity_check import check

J = "FROM public.gold_meter_journey_stage_agg"
RCV = ("FROM (SELECT r.*, ROW_NUMBER() OVER (PARTITION BY project ORDER BY created_at DESC, id DESC) AS rn "
       "FROM public.revenue_receivables r) l WHERE rn = 1")
OM = "FROM public.gold_om_technician_productivity_day_agg"


def has(problems, words):
    return any(words in p for p in problems)


def test_journey_keeps_replacements_and_uses_installation_year():
    tile = f"SELECT SUM(total_journey_days_sum) / NULLIF(SUM(revenue_realized_meter_count), 0) {J} WHERE NOT is_replacement"
    assert has(check("Meter journey avg time category wise", [tile]), "counts replacements")
    assert not has(check("Meter journey avg time excluding replacements", [tile]), "counts replacements")
    assert not has(check("Meter journey headline", [tile]), "counts replacements")
    supply = f"SELECT SUM(meter_count) {J} WHERE supply_month >= DATE '2026-01-01' AND supply_month < DATE '2027-01-01'"
    assert has(check("Total journey avg time by category for 2026", [supply]), "installation_year")
    year = f"SELECT SUM(meter_count) {J} WHERE installation_year = 2026"
    assert not has(check("Total journey avg time by category for 2026", [year]), "installation_year")


def test_journey_stage_denominators():
    wrong = (f"SELECT ROUND(SUM(sat_to_revenue_realized_days_sum) / NULLIF(SUM(meter_count), 0), 2) {J} "
             "WHERE LOWER(project) = 'agra' GROUP BY circle")
    right = wrong.replace("SUM(meter_count)", "SUM(revenue_realized_meter_count)")
    q = "Which circle has the slowest SAT to revenue time in agra?"
    assert has(check(q, [wrong]), "revenue_realized_meter_count")
    assert not has(check(q, [right]), "revenue_realized_meter_count")


def test_several_meters_and_gold_tables_for_one_meter():
    one = ("SELECT * FROM gmr_wfm_silver.unified_installations WHERE new_meter_serial_number IN ('AL1151970','AL3188371') "
           "ORDER BY updated_at DESC LIMIT 1")
    assert has(check("Journey of meters AL1151970 and AL3188371", [one]), "one row per meter")
    gold = ("SELECT current_stage FROM public.gold_meter_installation_pending_stage_snapshot "
            "WHERE meter_serial_number = 'SC10196043'")
    assert has(check("Where is meter SC10196043 stuck?", [gold]), "silver tables")


def test_current_stage_not_installed_and_stage_wise():
    blank = ("SELECT COUNT(*) FROM gmr_wfm_silver.unified_installations "
             "WHERE TRIM(COALESCE(new_meter_serial_number,'')) = ''")
    assert has(check("How many meters are not yet installed?", [blank]), "gold_meter_installation_pending_stage_snapshot")
    two = ("SELECT current_stage, SUM(meter_count) FROM (SELECT current_stage, meter_count FROM "
           "public.gold_meter_installation_pending_stage_snapshot UNION ALL SELECT current_stage, meter_count FROM "
           "public.gold_meter_revenue_pending_stage_snapshot) n WHERE current_stage IN "
           "('installation_done_sat_pending', 'sat_done_revenue_pending') GROUP BY ROLLUP (current_stage)")
    assert has(check("Meters current stage for revenue not realized stage wise", [two]), "all six stages")


def test_group_by_lower_beside_the_column():
    sets = ("SELECT CASE WHEN GROUPING(LOWER(project)) = 1 THEN 'Total' ELSE UPPER(project) END, SUM(meter_count) "
            "FROM public.gold_meter_installation_pending_stage_snapshot WHERE current_stage = 'agency' "
            "GROUP BY GROUPING SETS ((LOWER(project)), (project))")
    assert has(check("Which project has the most meters pending with agency?", [sets]), "group by the column itself")
    ok = ("SELECT CASE WHEN GROUPING(project) = 1 THEN 'Total' ELSE UPPER(project) END, SUM(meter_count) "
          "FROM public.gold_meter_installation_pending_stage_snapshot GROUP BY ROLLUP (project)")
    assert not has(check("Which project has the most meters pending with agency?", [ok]), "group by the column itself")


def test_receivables_crore_outstanding_and_money_tables():
    scaled = f"SELECT ROUND(SUM(lumpsum_invoiced + pmpm_invoiced + cable_invoiced) / 100000000.0, 2) {RCV}"
    assert has(check("Total invoiced amount in crore", [scaled]), "already in crore")
    overdue = f"SELECT SUM(pmpm_invoiced - pmpm_realised - pmpm_within_timeline) {RCV}"
    assert has(check("How much PMPM revenue is outstanding?", [overdue]), "outstanding = invoiced - realised")
    assert not has(check("How much PMPM revenue is overdue?", [overdue]), "outstanding = invoiced - realised")
    meters = "SELECT project, SUM(outstanding_count) FROM public.gold_revenue_ageing_snapshot GROUP BY project"
    assert has(check("Which project has the highest outstanding amount?", [meters]), "receivables topic")
    assert not has(check("Which zone has the most revenue not realized meters?", [meters]), "receivables topic")


def test_revenue_outstanding_reads_the_revenue_ageing_table():
    nonsat = ("SELECT SUM(meter_count) FROM public.gold_non_sat_ageing_snapshot WHERE LOWER(project) = 'triveni' "
              "AND age_key IN ('age_91_120', 'age_120_plus')")
    assert has(check("How many meters are outstanding older than 90 days in triveni?", [nonsat]),
               "gold_revenue_ageing_snapshot")
    assert not has(check("non-SAT meters older than 90 days in triveni", [nonsat]), "gold_revenue_ageing_snapshot")


def test_om_productivity_rollup_rows_month_bucket_and_ticket_rows():
    q = "Daily O&M productivity of technician NITIN PAL in September 2026"
    assert has(check(q, [f"SELECT period_start, SUM(complaint_count) {OM} WHERE technician_name ILIKE '%nitin pal%' "
                        "AND period_start BETWEEN DATE '2026-09-01' AND DATE '2026-09-30' GROUP BY period_start"]),
               "roll-up rows")
    by_day = (f"SELECT period_start, SUM(complaint_count) {OM} WHERE category = 'DT' AND period_start BETWEEN "
              "DATE '2026-01-01' AND DATE '2026-10-27' GROUP BY ROLLUP (period_start)")
    assert has(check("O&M productivity month wise for DT complaints in 2026", [by_day]), "date_trunc('month'")
    assert not has(check("O&M productivity month wise for DT complaints in 2026",
                         [by_day.replace("ROLLUP (period_start)", "ROLLUP (date_trunc('month', period_start))")]),
                   "date_trunc('month'")
    closed = ("SELECT * FROM gmr_wfm_silver.unified_non_qc_complaints WHERE technician_name = 'ARYENDRA KUMAR VISION ETAH' "
              "AND CAST(closed_date AS DATE) BETWEEN DATE '2026-08-25' AND DATE '2026-09-24'")
    q = "List the tickets closed by technician ARYENDRA KUMAR VISION ETAH from Aug 25 2026 to Sep 24 2026"
    assert has(check(q, [closed]), "work_completed_at")


def test_extra_query_keeps_the_place_and_last_month():
    main = ("SELECT SUM(closed_tickets) FROM public.gold_om_closed_tickets_day_agg WHERE LOWER(project) = 'triveni' "
            "AND closed_date >= DATE '2026-09-01' AND closed_date < DATE '2026-10-01'")
    extra = (f"SELECT SUM(complaint_count) {OM} WHERE category <> 'ALL' AND period_start >= DATE '2026-09-01' "
             "AND period_start < DATE '2026-10-01'")
    assert has(check("List the closures in triveni in September 2026", [main, extra]), "same project filter")
    import datetime as dt
    first = dt.date.today().replace(day=1).isoformat()
    now = f"SELECT SUM(complaint_count) {OM} WHERE category = 'ALL' AND period_start BETWEEN DATE '{first}' AND CURRENT_DATE"
    assert has(check("Which technician closed the most complaints in agra last month?", [now]), "previous calendar month")


def test_revenue_not_realized_meters_is_the_current_stage_card():
    ageing = "SELECT project, SUM(outstanding_count) FROM public.gold_revenue_ageing_snapshot GROUP BY project"
    assert has(check("Revenue not realized meters project wise", [ageing]), "current-stage card")
    assert not has(check("Revenue not realized ageing project wise", [ageing]), "current-stage card")
    assert not has(check("How many meters have revenue not realized for more than 60 days?", [ageing]),
                   "current-stage card")
