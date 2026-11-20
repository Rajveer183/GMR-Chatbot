"""19 Nov follow-up chain test, second set (N1-N12): offline checks for each fix."""
import pandas as pd

from gmr_agents.agents.router import triage
from gmr_agents.checks import number_check
from gmr_agents.execution import hierarchy
from gmr_agents.presentation import analyst
from gmr_agents.resolvers import scope


def _state(q):
    return {"question": q, "steps": [], "trace_id": "t", "trace": []}


# --- grain / change words are not a measure: the latest turn's measure is carried --------------------------------
def test_grain_and_change_words_are_not_measures():
    for q in ("Which day was the best?", "Which zone improved the most compared to last month?", "Is it getting better?"):
        assert triage._content_words(q) == set(), q
    assert "installations" in triage._content_words("Which day had the most installations?")


# --- an out-of-scope turn is not what a follow-up continues --------------------------------------------------------
def test_latest_data_turn_skips_refused_turn():
    turns = [{"question": "Pending SAT for Triveni", "context": {"domain": "wfm"}},
             {"question": "What's the weather in Varanasi today?", "context": {"language": "en"}}]
    assert triage.latest_data_turn(turns)["question"] == "Pending SAT for Triveni"
    assert triage.latest_data_turn(turns[1:])["question"].startswith("What's the weather")  # nothing better: as before


# --- 'back to project level' drops the drilled-into project ----------------------------------------------------------
def test_project_level_resets_places():
    assert scope._RESET_RE.search("Go back to project level, but for last month")
    assert scope._RESET_RE.search("back to all projects")
    assert not scope._RESET_RE.search("Break Kashi down by zone")


# --- a filter follow-up called chitchat is re-triaged with the latest question as the hint --------------------------
def test_filter_followup_rescued_from_chitchat(monkeypatch):
    seen = {}

    def fake(state, prior):
        seen["q"] = state["question"]
        return {"kind": "data", "question": "Meter stock by warehouse, single phase meters only", "domain": "wfm",
                "topics": ["stock_ageing"], "tables": [], "examples": [], "assumption": ""}
    monkeypatch.setattr(triage, "_triage", fake)
    t = {"kind": "chitchat", "question": "Only single phase meters", "reply": "Please specify ...", "followup": False}
    prior = [{"question": "Meter stock by warehouse", "context": {"domain": "wfm"}}]
    assert triage.rescue_filter_followup(_state("Only single phase meters"), t, prior)
    assert t["kind"] == "data" and t["reply"] == "" and t["followup"] and "Meter stock by warehouse" in seen["q"]
    # a real chitchat reply ('thanks') or no history is left alone
    t2 = {"kind": "chitchat", "question": "Thanks", "reply": "You're welcome"}
    assert not triage.rescue_filter_followup(_state("Thanks"), t2, prior)
    assert not triage.rescue_filter_followup(_state("Only single phase meters"), dict(t, kind="chitchat"), [])


# --- 'the worst one' = the code the previous answer named ------------------------------------------------------------
def test_the_worst_one_carries_the_named_code():
    t = {"kind": "data", "question": "Daily trend of energy loss for the worst feeder in Kashi in July 2026"}
    prior = [{"question": "Only for Kashi", "context": {"domain": "mdm"},
              "summary": "In July 2026 the highest energy loss was on feeder 41644120174YPR with 1,623,545 kWh lost."}]
    assert triage.carry_named_code(_state("Daily trend for the worst one"), t, prior)
    assert "41644120174YPR" in t["question"]
    t2 = {"kind": "data", "question": "Show the pending SAT counts for Kashi month-wise"}
    assert not triage.carry_named_code(_state("Just the first one"), t2,
                                       [{"question": "x", "summary": "Kashi 1,234 and Triveni 2,345", "context": {"domain": "wfm"}}])


# --- 'just the first one' keeps the first of the places the latest question named -------------------------------------
def test_ordinal_place_narrows_a_list():
    hierarchy.wait_first_load()
    cases = [("Just the first one, month-wise.", "Show the pending SAT counts for Kashi and Triveni month-wise",
              "Show the pending SAT counts for Kashi and Triveni", "Show the pending SAT counts for Kashi month-wise"),
             ("Only the second one", "SAT % for Agra, Kashi and Triveni", "SAT % for Agra, Kashi and Triveni",
              "SAT % for Kashi")]
    for q, res, prev, want in cases:
        t = {"kind": "data", "question": res}
        assert triage.pick_ordinal_place(_state(q), t, [{"question": "x", "resolved_question": prev,
                                                         "context": {"domain": "wfm"}}])
        assert t["question"] == want


# --- 'compare with the week before' keeps both periods ---------------------------------------------------------------
def test_compare_with_keeps_both_periods():
    t = {"kind": "data", "question": "How many meters were installed day wise from 5 Nov 2026 to 11 Nov 2026",
         "followup": False}
    prior = [{"question": "Daily installations last week", "context": {"domain": "wfm"},
              "resolved_question": "How many meters installed day wise in the last 7 days"}]
    assert triage.keep_comparison(_state("Compare with the week before"), t, prior)
    assert "compared with" in t["question"] and "last 7 days" in t["question"]
    t2 = {"kind": "data", "question": "Compare Kashi with Triveni installations"}
    assert not triage.keep_comparison(_state("Compare it with Triveni"), t2, prior)  # comparison already kept


# --- bare result counts get thousands separators; years, ids and non-result numbers do not --------------------------
def test_group_digits():
    df = pd.DataFrame({"project": ["Total", "KASHI"], "installations": [41104, 15087],
                       "consumer_number": [1234567890, 1234567891]})
    out = number_check.group_digits("KASHI 15087; Total 41104 in 2026; consumer 1234567890; 99999 other; 2026-11-18", [df])
    assert out == "KASHI 15,087; Total 41,104 in 2026; consumer 1234567890; 99999 other; 2026-11-18"
    # a sentence comma or full stop after the number (live retest: 'KASHI 15087, AGRA 13341' was left bare)
    assert number_check.group_digits("KASHI 15087, AGRA 41104.", [df]) == "KASHI 15,087, AGRA 41,104."


# --- two FILTER periods under a WHERE that keeps only one of them ------------------------------------------------------
def test_dead_filter_period():
    from gmr_agents.checks import entity_check
    sql = ("SELECT period_start, SUM(meter_count) FILTER (WHERE period_start BETWEEN CURRENT_DATE - 7 AND CURRENT_DATE - 1) "
           "AS last_7, SUM(meter_count) FILTER (WHERE period_start BETWEEN CURRENT_DATE - 14 AND CURRENT_DATE - 8) AS prev "
           "FROM public.gold_mi_installation_day_agg WHERE has_meter_serial AND period_start BETWEEN CURRENT_DATE - 14 "
           "AND CURRENT_DATE - 8 GROUP BY period_start")
    assert entity_check._dead_filter_problems([sql])
    wide = sql.replace("AND CURRENT_DATE - 8 GROUP", "AND CURRENT_DATE - 1 GROUP")
    assert entity_check._dead_filter_problems([wide]) == []


# --- backlog clear date by code ---------------------------------------------------------------------------------------
def test_clear_date():
    df = pd.DataFrame({"pending_meters": [37917], "days_required": [13]})
    note = analyst.clear_date("If we install 3000 a day, when will it be cleared?", [df], pd.Timestamp("2026-11-19"))
    assert "02 Dec 2026" in note
    assert analyst.clear_date("How long does SAT take?", [pd.DataFrame({"avg_days": [40]})],
                               pd.Timestamp("2026-11-19")) == ""


# --- a current-month row covers the month only up to today ------------------------------------------------------------
def test_month_window_capped_at_today(monkeypatch):
    sent = {}

    def fake(sql):
        sent["sql"] = sql
        return pd.DataFrame({"lo": [pd.Timestamp("2026-11-01")], "hi": [pd.Timestamp("2026-11-19")]})
    monkeypatch.setattr(analyst.executor, "_run_postgres", fake)
    plan = {"data_engine": "postgres",
            "sql": "SELECT SUM(meter_count) FROM public.gold_mi_installation_month_agg "
                   "WHERE period_start >= date_trunc('month', CURRENT_DATE)::date AND period_start <= CURRENT_DATE"}
    assert "19 Nov 2026" in analyst.date_window(plan, None)
    assert "LEAST(" in sent["sql"] and "CURRENT_DATE)" in sent["sql"]
