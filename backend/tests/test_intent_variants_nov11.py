"""11 Nov 2026 retest (wording variants / filters / follow-ups / ambiguity / multilingual): the generic guards added.
Each case is a real question + SQL from that run."""
from gmr_agents.agents.router import triage as T
from gmr_agents.checks import entity_check as E
from gmr_agents.resolvers import language as L

INSTALL_WHERE = "has_meter_serial AND progress_category IS NOT NULL AND NOT is_replacement"


def _clarified(q, prior=None):
    state = {"question": q, "trace": [], "trace_id": "t"}
    tri = {"kind": "data", "question": q, "followup": False, "topics": ["x"], "tables": ["y"]}
    return T.clarify_vague(state, tri, prior or []), tri


# ---- clarification ------------------------------------------------------------------------------------------- #

def test_vague_first_messages_ask_back():
    for q in ["How many are flagged in Kashi?", "What is the current status?", "Show me the pending ones.",
              "Which meters need attention?", "Give me the worst-performing areas.", "How many are remaining?",
              "What changed this month?"]:
        done, tri = _clarified(q)
        assert done, q
        assert tri["kind"] == "chitchat" and tri["reply"].startswith("Could you tell me which figure"), q


def test_particular_place_asks_which():
    done, tri = _clarified("Give me the installation count for a particular Circle.")
    assert done and tri["reply"].startswith("Which circle")


def test_measured_or_followup_questions_are_not_clarified():
    for q in ["Show Low PF flagged meters in Kashi.", "Kashi mein kitne meters pending hain?", "Who is the top performer?",
              "Give me a one-line status of the program", "What has changed since last week on the headline KPIs?",
              "How many meters are installed in Kashi?", "Which Division has the highest pending SAT?",
              "How far are we from the pending-verification target?"]:
        assert not _clarified(q)[0], q
    # with an earlier turn the message is a follow-up -- triage resolves it
    assert not _clarified("Show me the pending ones.", prior=[{"question": "SAT in Kashi"}])[0]


# ---- invented period ------------------------------------------------------------------------------------------ #

def _period(q, resolved, followup=False):
    state = {"question": q, "trace": [], "trace_id": "t"}
    tri = {"kind": "data", "question": resolved, "followup": followup}
    return T.drop_invented_period(state, tri, []), tri["question"]


def test_invented_yesterday_is_dropped():
    assert _period("Kitne meter install hue?", "How many meters were installed yesterday?") == \
        (True, "How many meters were installed?")


def test_user_periods_are_kept():
    assert not _period("Kal kitne meter lage?", "How many meters were installed yesterday?")[0]
    assert not _period("pichhle 7 din me kitne lage", "How many meters were installed in the last 7 days?")[0]
    assert not _period("कल कितने मीटर लगे?", "How many meters were installed yesterday?")[0]
    assert not _period("DTs by load change band", "Show DTs by load change band (DT load rise since last month)")[0]


# ---- SQL checks ----------------------------------------------------------------------------------------------- #

def test_installs_without_period_are_all_time():
    week = (f"SELECT SUM(meter_count) FROM public.gold_mi_installation_day_agg WHERE {INSTALL_WHERE} "
            "AND LOWER(project) = 'kashi' AND period_start BETWEEN CURRENT_DATE - 7 AND CURRENT_DATE - 1")
    assert any("ALL TIME" in p for p in E.check("How many meters were installed in Kashi project", [week],
                                                 asked="Kashi mein installed meters batao"))
    recent = (f"SELECT period_start, SUM(meter_count) FROM public.gold_mi_installation_day_agg WHERE project = 'kashi' "
              f"AND {INSTALL_WHERE} GROUP BY period_start ORDER BY period_start DESC LIMIT 30")
    assert any("ALL TIME" in p for p in E.check("Kashi installation figures", [recent],
                                                 asked="Give me the Kashi installation figures."))
    # a period asked -> fine
    assert not any("ALL TIME" in p for p in E.check("How many meters were installed in Kashi in the last 7 days",
                                                     [week], asked="Kashi mein pichhle 7 din me kitne lage"))


def test_fanout_join_is_rejected():
    x10 = ("SELECT d.division, SUM(d.meter_count) AS total_mi, SUM(s.meter_count) FILTER (WHERE s.sat_stage IS NOT NULL) "
           "AS total_sat FROM public.gold_mi_installation_month_agg d JOIN public.gold_mi_sat_invoice_snapshot s "
           "ON d.zone = s.zone AND d.division = s.division GROUP BY d.division")
    assert any("JOIN" in p for p in E._fanout_problems(x10))
    amb4 = ("SELECT nsa.project, SUM(nsa.meter_count), COALESCE(SUM(agg.open_complaints), 0) FROM "
            "public.gold_non_sat_ageing_snapshot nsa LEFT JOIN (SELECT project, category, SUM(open_complaints) AS "
            "open_complaints FROM public.gold_om_meter_complaints_agg GROUP BY project, category) agg "
            "ON nsa.project = agg.project AND nsa.category = agg.category GROUP BY nsa.project")
    assert E._fanout_problems(amb4)
    ok = ("SELECT a.p, a.x, b.y FROM (SELECT project p, SUM(meter_count) x FROM public.gold_a GROUP BY 1) a "
          "JOIN (SELECT project p, SUM(n) y FROM public.gold_b GROUP BY 1) b ON a.p = b.p")
    assert not E._fanout_problems(ok)
    bands = ("SELECT b.label, COALESCE(SUM(n.meter_count), 0) FROM (VALUES (1, 'age_0_30', '0-30 days')) AS "
             "b(ord, age_key, label) LEFT JOIN public.gold_non_sat_ageing_snapshot n ON n.age_key = b.age_key "
             "GROUP BY b.ord, b.label")
    assert not E._fanout_problems(bands)


def test_bare_varanasi_is_the_two_zones():
    g1 = f"SELECT SUM(meter_count) FROM public.gold_mi_installation_day_agg WHERE {INSTALL_WHERE} AND LOWER(project) = 'kashi'"
    assert E._varanasi_problems("How many meters are installed in Varanasi?", g1.lower())
    assert not E._varanasi_problems("How many meters are installed in Varanasi project?", g1.lower())
    zones = g1 + " AND LOWER(zone) IN ('varanasi i', 'varanasi ii')"
    assert not E._varanasi_problems("How many meters are installed in Varanasi?", zones.lower())


def test_two_projects_get_a_row_each():
    f5 = ("SELECT SUM(meter_count) AS total_mi, (SELECT SUM(meter_count) FROM public.gold_non_sat_ageing_snapshot WHERE "
          "LOWER(project) IN ('kashi', 'triveni')) AS eligible_pending_sat, SUM(meter_count) FILTER (WHERE sat_stage IS "
          "NULL) AS not_sat FROM public.gold_mi_sat_invoice_snapshot WHERE LOWER(project) IN ('kashi', 'triveni')")
    assert any("one row per project" in p for p in E.check("Show pending SAT counts for Kashi and Triveni", [f5]))
    grouped = f5 + " GROUP BY project"
    assert not any("one row per project" in p for p in E.check("Show pending SAT counts for Kashi and Triveni", [grouped]))


def test_ranking_pending_sat_by_division_is_allowed():
    a4 = ("SELECT division, SUM(meter_count) AS eligible_pending_sat FROM public.gold_non_sat_ageing_snapshot "
          "GROUP BY division ORDER BY eligible_pending_sat DESC LIMIT 10")
    assert not any("total AND its split" in p for p in E.check("Show the top 10 divisions with the highest number of "
                                                                "installed meters pending SAT", [a4]))


def test_bare_count_needs_a_total():
    x4 = ("SELECT data_date AS data_month, zone, category, SUM(unique_meter_count) AS flagged_meters FROM "
          "hes_gold.lpf2_low_power_factor WHERE severity NOT IN ('Normal', 'NoData') GROUP BY data_date, zone, category "
          "ORDER BY flagged_meters DESC")
    assert E._bare_count_problems("Low PF flagged count", x4)
    assert not E._bare_count_problems("Low PF flagged count zone wise", x4)
    assert not E._bare_count_problems("Low PF flagged count", x4.replace("GROUP BY data_date, zone, category",
                                                                          "GROUP BY ROLLUP (zone, category)"))


# ---- language ------------------------------------------------------------------------------------------------- #

def test_english_assumption_lead_in_hindi_is_translated():
    mixed = "Assuming you mean installed meters (all time): 43,40,272 मीटर इंस्टॉल हुए हैं (all time, all projects)."
    assert L.written_in(mixed, "hi") != "hi"


def test_vague_explain_is_clarified_but_definitions_are_not():
    state = {"question": "What is the current status?", "trace": [], "trace_id": "t"}
    tri = {"kind": "explain", "question": "What is the current status?", "followup": False, "topics": [], "tables": []}
    assert T.clarify_vague(state, tri, []) and tri["kind"] == "chitchat"
    state = {"question": "What is SAT status?", "trace": [], "trace_id": "t"}
    tri = {"kind": "explain", "question": "What is SAT status?", "followup": False, "topics": [], "tables": []}
    assert not T.clarify_vague(state, tri, [])


def test_union_legs_need_the_same_place_filter():
    f5 = ("SELECT u.project, SUM(u.not_sat), SUM(u.pend) FROM (SELECT LOWER(project) AS project, CASE WHEN sat_stage IS "
          "NULL THEN meter_count ELSE 0 END AS not_sat, 0 AS pend FROM public.gold_mi_sat_invoice_snapshot UNION ALL "
          "SELECT LOWER(project), 0, meter_count FROM public.gold_non_sat_ageing_snapshot WHERE LOWER(project) IN "
          "('kashi', 'triveni')) u GROUP BY u.project")
    assert E._union_filter_gaps(f5)
    both = f5.replace("FROM public.gold_mi_sat_invoice_snapshot UNION",
                      "FROM public.gold_mi_sat_invoice_snapshot WHERE LOWER(project) IN ('kashi', 'triveni') UNION")
    assert not E._union_filter_gaps(both)


def test_compare_both_keeps_places_from_earlier_turns():
    prior = [{"question": "How many meters are installed in Kashi?", "summary": "1,383,993 meters in Kashi.",
              "context": {"domain": "wfm", "topics": ["installation"]}},
             {"question": "What about Triveni?", "resolved_question": "How many meters are installed in Triveni?",
              "summary": "1,297,481 meters in Triveni.", "context": {"domain": "wfm", "topics": ["installation"]}}]
    for msg in ["Compare both.", "Now show their pending SAT counts.", "Dono ka SAT batao"]:
        state = {"question": msg, "trace": [], "trace_id": "t"}
        tri = {"kind": "data", "question": "Compare installed meters in Kashi and Triveni", "followup": True,
               "domain": "wfm", "topics": ["installation"], "named": ""}
        T.fix_scope(state, tri, prior)
        assert "Kashi" in tri["question"] and "Triveni" in tri["question"], msg
