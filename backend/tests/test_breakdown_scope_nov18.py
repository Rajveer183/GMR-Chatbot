"""18 Nov UI chat: 'complaints of all kashi cluster' -> 'all clusters' -> 'in detail' -> 'meter wise breakdown'.
Grain words (meter / DT / consumer / category), 'all clusters' = all projects, and 'in detail' drill-down."""

from gmr_agents.agents.router.triage import _detail_drilldown
from gmr_agents.agents.sql_writer.agent import _repeats_previous_turn
from gmr_agents.agents.sql_writer.planner import _new_grouping
from gmr_agents.checks import entity_check
from gmr_agents.resolvers import scope
from gmr_agents.resolvers.query_encoder import encode

PROJECT_ROLLUP = ("SELECT CASE WHEN GROUPING(project) = 1 THEN 'Total' ELSE UPPER(project) END AS project, "
                  "SUM(total_complaints) AS total_complaints FROM public.gold_om_meter_complaints_agg "
                  "GROUP BY ROLLUP (project) ORDER BY GROUPING(project) DESC, total_complaints DESC")
COMPLAINT_COLS = {"meter_serial_number", "project", "discom", "zone", "circle", "division", "sub_division",
                  "category", "total_complaints", "open_complaints", "closed_complaints", "sat_status"}


def test_encoder_grains():
    assert encode("give me meter wise breakdown")["breakdown_by"] == ["meter"]
    assert encode("show me complaints per meter in agra")["breakdown_by"] == ["meter"]
    assert encode("low pf by dt")["breakdown_by"] == ["dt"]
    assert "project" in encode("also give me total complaints for all clusters")["breakdown_by"]
    # GMR meaning: consumer + DT + feeder together = the meter category, not per consumer / per DT
    assert encode("split sat 10 by dt, consumer and feeder")["breakdown_by"] == ["category"]
    # attributes are not the entity
    assert "meter" not in (encode("break that down by meter type per day").get("breakdown_by") or [])
    assert encode("in detail\nas much detail as you can provide").get("detail") is True
    assert not encode("give me the complaints of all kashi cluster").get("breakdown_by")


def test_all_clusters_resets_carried_project():
    assert scope._RESET_RE.search("also give me total complaints for all clusters")
    assert scope._RESET_RE.search("cluster wise complaints")
    assert scope.compared_rank("which cluster has most complaints") == 0
    out, changes = scope.apply("also give me total complaints for all clusters",
                               "Total complaints for all clusters in the Kashi project",
                               {"question": "complaints of all kashi cluster"}, True, {})
    assert "kashi" not in out.lower() and changes.get("stripped") == ["kashi"]


def test_meter_wise_sql_must_group_by_meter():
    problems = entity_check.check("Show total complaints for all projects meter-wise", [PROJECT_ROLLUP],
                                  columns=COMPLAINT_COLS, asked="give me meter wise breakdown")
    assert any("meter-wise breakdown" in p for p in problems)
    ok = ("SELECT meter_serial_number, total_complaints FROM public.gold_om_meter_complaints_agg "
          "ORDER BY total_complaints DESC LIMIT 50")
    assert not any("meter-wise" in p for p in entity_check.check(
        "Show complaints meter-wise", [ok], columns=COMPLAINT_COLS, asked="give me meter wise breakdown"))


def test_all_clusters_sql_must_not_filter_one_project():
    sql = PROJECT_ROLLUP.replace("GROUP BY", "WHERE LOWER(project) = 'kashi' GROUP BY")
    problems = entity_check.check("Total complaints for all clusters in the Kashi project", [sql],
                                  columns=COMPLAINT_COLS, asked="also give me total complaints for all clusters")
    assert any("all projects" in p for p in problems)


def test_detail_followup_drills_one_level_down():
    assert "zone wise" in _detail_drilldown({"context": {"sql": PROJECT_ROLLUP}})
    zone = PROJECT_ROLLUP.replace("ROLLUP (project)", "ROLLUP (zone)")
    assert "circle wise" in _detail_drilldown({"context": {"sql": zone}})


def test_repeat_of_previous_sql_is_rejected_for_detail_and_new_grain():
    prior = [{"context": {"sql": PROJECT_ROLLUP + "\nLIMIT 10000"}}]
    assert "MORE DETAIL" in _repeats_previous_turn({"prior_turns": prior, "question": "in detail"}, PROJECT_ROLLUP)
    assert "meter" in _repeats_previous_turn({"prior_turns": prior, "question": "meter wise breakdown"},
                                             PROJECT_ROLLUP)
    assert _repeats_previous_turn({"prior_turns": prior, "question": "and in agra?"}, PROJECT_ROLLUP) is None


def test_planner_flags_new_grouping():
    assert _new_grouping("Show total complaints meter-wise", PROJECT_ROLLUP) == "meter-wise"
    assert _new_grouping("Show total complaints project-wise", PROJECT_ROLLUP) == ""


def test_detail_wording_names_no_category_and_no_share():
    text = _detail_drilldown({"context": {"sql": PROJECT_ROLLUP}})
    assert "consumer" not in text.lower() and "share" not in text.lower()
    assert not encode(text).get("categories")


def test_window_share_over_rollup_is_rejected():
    sql = ("SELECT zone, SUM(meter_count) AS n, ROUND(SUM(meter_count) * 100.0 / NULLIF(SUM(SUM(meter_count)) OVER (), 0), 2) "
           "AS share_pct FROM public.gold_non_sat_ageing_snapshot GROUP BY ROLLUP (zone)")
    assert any("OVER ()" in p for p in entity_check.check("eligible pending SAT zone-wise", [sql]))
