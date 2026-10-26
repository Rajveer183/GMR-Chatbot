"""O&M card checks (entity_check._om_problems): mistakes seen live on 26 Oct 2026 against the dashboard SQL."""

from gmr_agents.validation.entity_check import check


def om(question, sql, needle):
    return [p for p in check(question, [sql]) if needle in p]


def test_om_technician_days_are_distinct_pairs():
    bad = ("SELECT SUM(complaint_count) / SUM(technician_day_count) FROM public.gold_om_technician_productivity_day_agg "
           "WHERE category <> 'ALL'")
    good = ("SELECT SUM(complaint_count) / COUNT(DISTINCT (technician_name, period_start)) "
            "FROM public.gold_om_technician_productivity_day_agg WHERE category = 'ALL'")
    assert om("O&M productivity per team", bad, "COUNT(DISTINCT (technician_name, period_start))")
    assert not om("O&M productivity per team", good, "COUNT(DISTINCT (technician_name, period_start))")
    month = "SELECT SUM(complaint_count) FROM public.gold_om_technician_productivity_month_agg"
    assert om("O&M productivity for consumer meters in Kashi", month, "not the month table")


def test_escalations_count_tickets():
    bad = "SELECT zone, SUM(escalation_count) FROM public.gold_om_discom_escalation_snapshot GROUP BY zone"
    assert om("Pending discom escalations zone wise", bad, "SUM(ticket_count)")


def test_non_comm_filters_its_two_types():
    unfiltered = "SELECT SUM(ticket_count) FROM public.gold_om_open_ticket_detail_snapshot WHERE category = 'DT'"
    both = ("SELECT complaint_status, SUM(ticket_count) FROM public.gold_om_open_ticket_detail_snapshot "
            "WHERE complaint_type IN ('NEVER COMM', 'METER NOT IN COMMUNICATION') GROUP BY complaint_status")
    assert om("Unassigned non-comm tickets for DT meters", unfiltered, "holds every open ticket")
    assert om("Never comm tickets status wise", both, "NEVER COMM tickets only")
    assert not om("Non-comm tickets status wise", both, "tickets only")


def test_open_ageing_carries_sat_split():
    bands = "SELECT age_key, SUM(ticket_count) FROM public.gold_om_open_ageing_snapshot GROUP BY age_key"
    assert om("Open O&M ticket ageing for complaint source 1912", bands, "SAT Done and SAT Pending")
    assert not om("Open O&M tickets complaint type wise",
                  "SELECT complaint_type, SUM(ticket_count) FROM public.gold_om_open_ageing_snapshot GROUP BY 1",
                  "SAT Done and SAT Pending")


def test_month_words_are_not_read_inside_other_words():
    from gmr_agents.validation.query_encoder import encode
    assert not encode("Escalations by discom decision").get("months")  # live 26 Oct: filtered December 2026
    assert not encode("marching ahead with augmented teams").get("months")
    assert encode("installs in Dec 2025")["months"] == ["december"]
    assert encode("installs in September 2026")["months"] == ["september"]
    assert encode("from 1 Aug 2026 to 30 sept 2026")["dates"] == ["1 Aug", "30 sept"]


def test_type_narrowing_reads_the_users_own_words():
    both = ("SELECT age_key, SUM(ticket_count) FROM public.gold_om_open_ticket_detail_snapshot "
            "WHERE complaint_type IN ('NEVER COMM', 'METER NOT IN COMMUNICATION') GROUP BY age_key")
    widened = "Show the ageing of open 'Never Comm' and 'Meter Not in Communication' tickets in Agra."
    assert [p for p in check(widened, [both], asked="Never comm tickets ageing in Agra") if "NEVER COMM tickets only" in p]


def test_non_comm_count_carries_the_split():
    total = ("SELECT SUM(ticket_count) FROM public.gold_om_open_ticket_detail_snapshot "
             "WHERE complaint_type IN ('NEVER COMM', 'METER NOT IN COMMUNICATION') AND LOWER(zone) = 'prayagraj ii'")
    assert [p for p in check("How many communication failure tickets are open in Prayagraj II zone?", [total])
            if "also gives its split" in p]


def test_hhu_window_ends_yesterday():
    to_year_end = ("SELECT circle, SUM(ticket_count) FROM public.gold_om_hhu_reading_day_agg "
                   "WHERE period_start >= DATE '2026-01-01' AND period_start <= DATE '2099-12-31' GROUP BY circle")
    good = ("SELECT circle, SUM(ticket_count) FROM public.gold_om_hhu_reading_day_agg "
            "WHERE period_start BETWEEN DATE '2026-01-01' AND CURRENT_DATE - 1 GROUP BY circle")
    assert [p for p in check("HHU readings circle wise in 2026", [to_year_end]) if "end yesterday" in p]
    assert not [p for p in check("HHU readings circle wise in 2026", [good]) if "end yesterday" in p]


def test_place_productivity_is_the_team_figure():
    per_tech = ("SELECT technician_name, SUM(complaint_count) / COUNT(DISTINCT (technician_name, period_start)) "
                "FROM public.gold_om_technician_productivity_day_agg WHERE category = 'ALL' GROUP BY technician_name")
    assert [p for p in check("O&M tickets closed per technician per day in Triveni", [per_tech]) if "team figure" in p]
    assert not [p for p in check("Top performing O&M technician", [per_tech]) if "team figure" in p]


def test_complaint_meter_sat_status_is_counts():
    rows = ("SELECT meter_serial_number, sat_status FROM public.gold_om_meter_complaints_agg "
            "ORDER BY total_complaints DESC LIMIT 50")
    assert [p for p in check("Complaint meters SAT status for Triveni in September 2026", [rows]) if "tab's tiles" in p]
    assert not [p for p in check("Top 50 complaint meters with SAT status", [rows]) if "tab's tiles" in p]
