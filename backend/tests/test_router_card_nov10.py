"""10 Nov 2026 full retest: the code router fuzzy-matched 'Voltage unbalance by circle' (VUI card) to the verified
example 'voltage imbalance by circle' (3-sigma imbalance card) and answered from the wrong table."""

from gmr_agents.agents.router import triage


def test_unbalance_does_not_route_to_imbalance_example():
    assert not triage._same_card("Voltage unbalance by circle",
                                 ["Voltage imbalance circle-wise", "voltage imbalance by circle"])


def test_same_card_wordings_still_route():
    assert triage._same_card("voltage imbalance by circle", ["Voltage imbalance circle-wise", "voltage imbalance by circle"])
    assert triage._same_card("average VUI", ["What is the average voltage unbalance index?", "average VUI"])


def test_non_technical_questions_unaffected():
    assert triage._same_card("How many meters were installed yesterday?", ["How many meters were installed yesterday?"])


def test_concept_difference_is_explain():
    import pytest
    for q in ["What is the difference between % DT loading and % DT peak loading?",
              'What is the difference between "Total MI" and "Eligible MI for SAT"?',
              "What is the difference between index based and AI based feeder?"]:
        assert triage._CONCEPT_DIFF_RE.search(q), q
    for q in ["What is the difference between Agra and Kashi in SAT?", "difference between 1,277,180 and 1,106,912",
              "What is the difference between installed and supplied this month?",
              "what is the difference between AEW and HPL defect rate"]:
        assert not triage._CONCEPT_DIFF_RE.search(q), q


def test_pq_meter_type_must_filter_role():
    from gmr_agents.checks import entity_check as E
    sql = ("SELECT COUNT_IF(status = 'Poor') FROM hes_gold.voltage_unbalance_index "
           "WHERE data_date = DATE '2026-07-01' AND blocks_3ph > 0")
    assert any("role = 'dtr'" in p for p in E._technical_problems("How many DT meters have poor voltage unbalance?", [sql]))
    assert not E._technical_problems("How many meters have poor voltage unbalance?", [sql])
    assert not E._technical_problems("voltage deviation by DT", [sql.replace("COUNT_IF", "dtr_name, COUNT_IF")])
    ok = sql + " AND role = 'dtr'"
    assert not E._technical_problems("How many DT meters have poor voltage unbalance?", [ok])


def test_comparison_of_two_cards_gets_both_tables():
    q = "Which circles have both average VDI above 9% and load violation above 9%?"
    tables, extra = triage.multi_card_tables(q, triage.technical_topic(q))
    assert {"voltage_deviation_index", "load_violation", "load_assessed_consumers"} <= set(tables)
    assert extra == ["mdm_power_quality"]
    q = ("Per project, compare DTs overloaded above 100% on peak loading with DTs that peaked above rating "
         "on % DT loading")
    tables, _ = triage.multi_card_tables(q, triage.technical_topic(q))
    assert {"dt_loading", "dt_peak_loading"} <= set(tables) and tables[-1] == "dt_load_profile"


def test_dt_lists_exclude_implausible_dts():
    from gmr_agents.checks import entity_check as E
    sql = ("SELECT meter_no, dtr_code, SUM(overload_slots) / 2.0 AS hours FROM hes_gold.dt_load_kva "
           "WHERE month_date = DATE '2026-07-01' GROUP BY 1, 2 ORDER BY hours DESC LIMIT 10")
    q = "Which transformers run above their rating for the most hours?"
    assert any("implausible" in p for p in E._technical_problems(q, [sql]))
    ok = sql.replace("ORDER BY", "HAVING MAX(peak_pct_loading) < 200 ORDER BY")
    assert not any("implausible" in p for p in E._technical_problems(q, [ok]))


def test_explain_plus_current_value_is_data():
    assert triage._EXPLAIN_AND_VALUE_RE.search("How is the defect rate worked out, and what is it right now?")
    assert not triage._EXPLAIN_AND_VALUE_RE.search("How is SAT % calculated?")
    assert not triage._EXPLAIN_AND_VALUE_RE.search("what does non-SAT ageing mean")


def test_domain_follows_topics_when_model_contradicts_them():
    d = {"kind": "data", "domain": "mdm", "followup": True, "tables": [],
         "question": "How many DT meters installed, how many SAT done, and how many defects found OEM-wise in Agra?",
         "topics": ["installation", "sat_progress", "defect"]}
    r = triage._parse_triage(d, {"question": "only DT"}, "test")
    assert r["domain"] == "wfm" and r["topics"][:1] == ["installation"]


def test_pending_sat_never_adds_eligible_to_installed_not_sat():
    from gmr_agents.checks import entity_check as E
    bad = ("SELECT project, SUM(meter_count) AS pending FROM (SELECT LOWER(project) AS project, meter_count "
           "FROM public.gold_mi_sat_invoice_snapshot WHERE sat_stage IS NULL UNION ALL SELECT LOWER(project), meter_count "
           "FROM public.gold_non_sat_ageing_snapshot) sub GROUP BY project ORDER BY pending DESC")
    assert any("never UNION" in p for p in E.check("Which project has the highest pending SAT meters?", [bad]))


def test_share_columns_above_100_are_implausible():
    import pandas as pd
    from gmr_agents.agents.sql_writer.planner import _impossible_shares
    bad = pd.DataFrame({"project": ["kashi"], "sat_pct": [1140615.93], "defect_rate_pct": [14261.06]})
    assert len(_impossible_shares(bad)) == 2
    ok = pd.DataFrame({"sat_pct_vs_mi": [88.05], "change_pct": [347.2], "peak_pct_loading": [180.0],
                       "pct_vs_target": [120.0]})
    assert _impossible_shares(ok) == []


def test_flat_default_month_is_rewritten_by_code():
    from gmr_agents.agents.sql_writer.planner import _clean_sql
    from gmr_agents.checks import entity_check as E
    flat = ("SELECT COUNT(*) FROM hes_gold.voltage_drop WHERE data_date = (SELECT MAX(month_date) FROM "
            "hes_gold.dt_load_profile WHERE month_date >= DATE '2026-06-01' GROUP BY month_date "
            "HAVING COUNT(DISTINCT data_date) >= 10)")
    fixed = _clean_sql(flat)
    assert E.TECHNICAL_DEFAULT_MONTH in fixed
    cte = "WITH m AS (SELECT MAX(month_date) AS mon FROM hes_gold.dt_load_profile GROUP BY month_date HAVING COUNT(DISTINCT data_date) >= 10) SELECT 1"
    assert "SELECT MAX(month_date) AS mon FROM (SELECT month_date" in _clean_sql(cte)


def test_without_sat_wording_with_contractions():
    from gmr_agents.checks import entity_check as E
    assert E._WITHOUT_SAT_RE.search("How many meters are in the ground but haven't cleared SAT yet?")
    assert not E._WITHOUT_SAT_RE.search("How many meters have cleared SAT?")


def test_explain_evidence_keeps_chosen_topic_and_named_terms():
    from shared import config
    from gmr_agents.checks import grounding
    ids = [e.chunk_id for e in grounding.collect_explain_evidence(
        "Explain Confirmed, Reassign, Review and No data.", ["mdm_autoindexing"], k=config.GROUNDING_TOP_K)]
    assert "rule:mdm_autoindexing" in ids and {"term:confirmed", "term:reassign", "term:review"} <= set(ids)


def test_measure_retriage_carries_its_domain(monkeypatch):
    first = {"kind": "data", "question": "only DT meters", "domain": "mdm", "topics": ["mdm_dt_load"],
             "tables": ["dt_loading"], "followup": True}
    again = {"kind": "data", "question": "How many DT meters installed, how many SAT done, how many defects OEM-wise in Agra",
             "domain": "wfm", "topics": ["installation", "sat_progress", "defect"], "tables": [], "followup": True}
    monkeypatch.setattr(triage, "_triage", lambda *a, **k: dict(again))
    prior = [{"question": "How many meters installed, how many SAT done, how many defects found OEM wise in Agra?"}]
    triage.fix_measure({"question": "only DT", "trace_id": "t", "trace": []}, first, prior)
    assert first["domain"] == "wfm" and first["topics"][0] == "installation"


def test_wfm_question_not_left_on_meter_master_topics():
    q = "Non SAT meters zone wise in Kashi"
    r = triage._parse_triage({"kind": "data", "domain": "mdm", "question": q,
                              "topics": ["mdm_meter_master", "mdm_meter_readings"]}, {"question": q}, "test")
    assert r["domain"] == "wfm" and "non_sat_ageing" in r["topics"]


def test_zero_rows_with_wrong_case_is_not_genuine():
    from gmr_agents.agents.sql_writer import planner as P
    from gmr_agents.retrieval import knowledge as K
    if not K.known_value("edc agra", exact=True):
        import pytest
        pytest.skip("geography values not loaded")
    sql = "SELECT 1 FROM public.gold_mi_installation_day_agg WHERE circle = 'EDC Agra'"
    assert not P._zero_is_genuine(sql)
    assert P._zero_is_genuine(sql.replace("circle = 'EDC Agra'", "LOWER(circle) = 'edc agra'"))


def test_router_never_sends_wfm_wording_to_mdm_example():
    import re
    src = open(triage.__file__, encoding="utf-8").read()
    assert "HES meters zone-wise in Kashi" in src  # the guard documents the 11 Nov case
    assert triage._WFM.search("Non SAT meters zone wise in Kashi")
    assert not triage._WFM.search("HES meters zone-wise in Kashi")
