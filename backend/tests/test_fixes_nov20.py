"""20 Nov fixes: 'last N months' rolling to yesterday with partial months in brackets, and the open items from the
19 Nov follow-up runs. Offline: the database is stubbed."""
import pandas as pd

from gmr_agents.agents.narrator import writer
from gmr_agents.agents.router import triage
from gmr_agents.agents.sql_writer import planner
from gmr_agents.checks import entity_check, wording_check
from gmr_agents.execution import hierarchy
from gmr_agents.presentation import analyst, blocks
from gmr_agents.resolvers import scope


def _state(q):
    return {"question": q, "steps": [], "trace_id": "t", "trace": []}


_ROLLING = ("SELECT date_trunc('month', period_start)::date AS month, SUM(meter_count) AS meters_installed "
            "FROM public.gold_mi_installation_day_agg WHERE period_start BETWEEN "
            "(CURRENT_DATE - 1 - INTERVAL '4 months')::date AND CURRENT_DATE - 1 GROUP BY 1 ORDER BY 1")


def _months(*vals):
    return pd.DataFrame({"month": [f"2026-{m:02d}-01" for m in vals], "meters_installed": [10.0 * m for m in vals]})


def _stub_window(monkeypatch, lo, hi):
    analyst._BOUNDS_CACHE.clear()
    monkeypatch.setattr(analyst.executor, "_run_postgres",
                        lambda sql: pd.DataFrame({"lo": [pd.Timestamp(lo)], "hi": [pd.Timestamp(hi)]}))


# --- 'last 4 months' asked 20 Nov = 19 Jul .. 19 Nov: July and November are partial, said in brackets ----------------
def test_rolling_window_partial_months(monkeypatch):
    _stub_window(monkeypatch, "2026-07-19", "2026-11-19")
    plan = {"data_engine": "postgres", "sql": _ROLLING}
    df = _months(7, 8, 9, 10, 11)
    spans = analyst.month_spans(plan, df)
    assert spans == {"2026-07": "19-31 Jul", "2026-11": "1-19 Nov"}
    col, partial, line = analyst._partial_periods(df, spans)
    assert col == "month" and partial >= {"2026-07-01", "2026-11-01"} and "2026-08-01" not in partial
    assert "Jul 2026 covers only 19-31 Jul" in line and "Nov 2026 covers only 1-19 Nov" in line
    assert blocks.table_cell("2026-07-01", month=True, spans=spans) == "Jul 2026 (19-31 Jul)"
    assert blocks.table_cell("2026-08-01", month=True, spans=spans) == "Aug 2026"
    # the chart axis gets the same note (display only: x stays ISO)
    out = blocks._blocks({**plan, "metrics": [], "display": {"table": True}, "labels": {}, "chart": {"type": "bar", "x": "month",
                          "y": ["meters_installed"]}}, {"result": df})
    assert out["tables"][0]["rows"][0][next(iter(out["tables"][0]["rows"][0]))] == "Jul 2026 (19-31 Jul)"
    for c in out["charts"]:
        assert c["x"][0] == "2026-07-01" and c.get("x_notes", {}).get("2026-07-01") == "19-31 Jul"


def test_whole_months_have_no_brackets(monkeypatch):
    _stub_window(monkeypatch, "2026-08-01", "2026-10-31")
    assert analyst.month_spans({"data_engine": "postgres", "sql": _ROLLING}, _months(8, 9, 10)) == {}
    # a running balance is a month-END figure: its first month is not partial
    _stub_window(monkeypatch, "2026-07-19", "2026-11-19")
    bal = {"data_engine": "postgres", "sql": _ROLLING.replace("SUM(meter_count)", "SUM(SUM(meter_count)) OVER (ORDER BY 1)")}
    assert analyst.month_spans(bal, _months(7, 8, 9, 10, 11)) == {"2026-11": "1-19 Nov"}
    # no month column: nothing
    assert analyst.month_spans({"data_engine": "postgres", "sql": _ROLLING}, pd.DataFrame({"p": ["a", "b"], "v": [1, 2]})) == {}


def test_partial_periods_unchanged_without_spans():
    df = _months(8, 9, 10)
    assert "every month in RESULT is complete" in analyst._partial_periods(df)[2]


# --- whole-month SQL for 'last N months' goes back to the writer ------------------------------------------------------
def test_last_n_months_must_roll():
    f = entity_check._last_n_months_problems
    whole = ("select period_start from public.gold_mi_installation_month_agg "
             "where period_start >= (date_trunc('month', current_date) - interval '5 months')::date")
    assert f("installation trend last 6 months", whole)
    assert f("burnt meters over the last 3 months", "select 1 from public.gold_meter_defect_event_day_agg where "
             "period_start >= date_trunc('month', current_date - interval '3 months')")
    assert not f("installation trend last 6 months", _ROLLING.lower())
    assert not f("installs in the last 3 full months", whole)
    assert not f("installs this month", whole)
    assert not f("remaining stock last 3 months", "select sum(x) over (partition by p) from "
                 "public.gold_meter_inventory_event_month_agg")
    # a months spine built with date_trunc (not a filter) is fine
    assert not f("installs vs sat last 6 months", "select generate_series(date_trunc('month', current_date - 1 - "
                 "interval '6 months'), date_trunc('month', current_date - 1), interval '1 month') from "
                 "public.gold_mi_installation_day_agg where period_start between (current_date - 1 - interval "
                 "'6 months')::date and current_date - 1")


# --- an answer that is only a caveat is sent back for the figures -----------------------------------------------------
def test_caveat_only_summary_is_caught():
    df = pd.DataFrame({"feeder_code": ["42623421057BIG", "42601210211ABC"], "project": ["kashi", "kashi"],
                       "loss_pct": [100.0, 100.0], "input_kwh": [1141392.0, 682320.0]})
    assert wording_check.quotes_nothing("Feeder input carries the feeder meter's multiplying factor while consumers use "
                                        "1, and HES delivery is partial-month, so absolute percentages are not "
                                        "commercial losses, though rankings are usable.", df)
    assert not wording_check.quotes_nothing("Feeder 42623421057BIG tops the list.", df)
    assert not wording_check.quotes_nothing("Top input was 1,141,392 kWh.", df)
    assert not wording_check.quotes_nothing("There are 5 feeders.", pd.DataFrame({"n": [5]}))
    assert not wording_check.quotes_nothing("anything", pd.DataFrame())


# --- loss rankings get the card's tie-break by code -------------------------------------------------------------------
def test_loss_tie_break():
    f = planner._loss_tie_break
    assert f("SELECT net_input FROM hes_gold.feeder_loss ORDER BY loss_pct DESC NULLS LAST LIMIT 10").endswith(
        "ORDER BY loss_pct DESC NULLS LAST, net_input DESC LIMIT 10")
    kept = "SELECT net_input FROM hes_gold.feeder_loss ORDER BY loss_pct DESC NULLS LAST, net_input DESC LIMIT 1"
    assert f(kept) == kept
    grouped = ("SELECT zone, SUM(loss_kwh) * 100 / SUM(net_input) AS loss_pct FROM hes_gold.feeder_loss "
               "GROUP BY zone ORDER BY loss_pct DESC LIMIT 5")
    assert f(grouped) == grouped
    other = "SELECT x FROM public.gold_mi_installation_day_agg ORDER BY loss_pct DESC"
    assert f(other) == other


# --- 'the worst one' falls back to the codes of the latest table when the summary named none ---------------------------
def test_worst_one_from_table_codes():
    t = {"kind": "data", "question": "Daily trend for the feeder with the highest loss in July in Kashi"}
    prior = [{"question": "Feeders with the highest loss in July in Kashi", "context": {"domain": "mdm",
              "codes": ["42623421057BIG", "42601210211ABC"]}, "summary": "Feeder input carries the multiplying factor."}]
    assert triage.carry_named_code(_state("Daily trend for the worst one"), t, prior)
    assert "42623421057BIG" in t["question"]


# --- a bare 'Varanasi' is carried as Varanasi (zones I + II), not as the Kashi project -----------------------------------
def test_varanasi_carried_as_varanasi():
    hierarchy.wait_first_load()
    names = hierarchy.place_names()
    latest = {"question": "Which Varanasi zone is worst?",
              "resolved_question": "Which Varanasi zone has the most meters with low power factor for 3 or more months?",
              "summary": "Varanasi II is the worst zone with 88,124 meters, against 63,156 in Varanasi I."}
    got, _ = scope.apply("Forecast next month.", "Forecast low power factor meters for next month", latest, True, names)
    assert got.endswith("in Varanasi") and "Kashi" not in got
    kashi = {"question": "installs in kashi", "resolved_question": "Meters installed in Kashi", "summary": "x"}
    assert scope.apply("Forecast next month.", "Forecast installs next month", kashi, True, names)[0].endswith("in Kashi")


# --- 'Why is it lower?' after 'Compare it with Kashi' keeps the first subject -------------------------------------------
def test_why_it_keeps_subject():
    hierarchy.wait_first_load()
    ctx = {"domain": "wfm"}
    turns = [{"question": "Revenue billed vs collected for Triveni this month", "context": ctx, "summary": "x",
              "resolved_question": "Revenue billed vs collected for Triveni this month"},
             {"question": "Compare it with Kashi", "context": ctx, "summary": "y",
              "resolved_question": "Compare Triveni revenue billed vs collected with Kashi this month"}]
    t = {"kind": "data", "question": "Why is Kashi's revenue collection lower than Triveni's collection this month?"}
    assert triage.keep_pronoun_subject(_state("Why is it lower?"), t, turns)
    assert t["question"] == "Why is Triveni's revenue collection lower than Kashi's collection this month?"
    ok = {"kind": "data", "question": "Why is Triveni's collection efficiency lower than Kashi's this month?"}
    assert not triage.keep_pronoun_subject(_state("Why is it lower?"), ok, turns)


# --- 'Correlation or causation?' as a follow-up is answered from data --------------------------------------------------
def test_correlation_followup_is_data():
    d = {"kind": "explain", "domain": "wfm", "topics": ["sat_progress"],
         "question": "Is there a correlation or causation between SAT completion and billing/revenue collected?"}
    assert triage._parse_triage(d, _state("Correlation or causation?"), "m")["kind"] == "data"
    defn = {"kind": "explain", "domain": "wfm", "question": "What does correlation mean?"}
    assert triage._parse_triage(defn, _state("what does correlation mean?"), "m")["kind"] == "explain"


# --- 'how many more to reach the target' carries the latest measure -----------------------------------------------------
def test_target_words_are_not_a_measure():
    assert triage._content_words("How many more are needed to reach the target?") == set()
    assert "sat" in triage._content_words("How many more SAT meters are needed to reach 95%?")


# --- receivables 'this month' = latest row; mapping audit 'last month' = the audit month before the latest ---------------
def test_receivables_and_mapping_month_checks():
    assert entity_check._receivables_month_problems(
        "with latest as (select * from public.revenue_receivables where created_at >= date_trunc('month', current_date))")
    assert not entity_check._receivables_month_problems("select * from public.revenue_receivables")
    sql = ("SELECT verdict FROM hes_gold.mapping_audit WHERE meter_no = 'AL1157049' AND data_date = "
           "CAST(date_format(date_add('month', -1, current_date), '%Y-%m-01') AS date)")
    assert any("audit month before the latest" in p
               for p in entity_check._mapping_problems("Compare consumer AL1157049 with its feeder for last month", [sql]))


# --- month-wise asked of a snapshot: say it is not stored ---------------------------------------------------------------
def test_series_ask_on_snapshot():
    assert writer._SERIES_ASK_RE.search("Just the first one, month-wise.")
    assert writer._SERIES_ASK_RE.search("pending SAT trend")
    assert not writer._SERIES_ASK_RE.search("pending SAT for Kashi")


def test_varanasi_kept_when_model_wrote_kashi():
    hierarchy.wait_first_load()
    names = hierarchy.place_names()
    latest = {"question": "Which Varanasi zone has the most meters with low PF for 3 or more months?",
              "resolved_question": "Which Varanasi zone has the most meters with low power factor for 3 or more months?",
              "summary": "VARANASI II has the most, 72,276, against 53,020 in VARANASI I."}
    got, changes = scope.apply("Forecast next month.", "Forecast the number of low power factor meters for next month "
                               "in kashi", latest, True, names)
    assert got.endswith("in Varanasi") and changes.get("varanasi")


def test_partial_month_brackets_in_text():
    spans = {"2026-07": "19-31 Jul", "2026-11": "1-19 Nov"}
    s = ("Installations: 30,865 in Jul 2026, 45,732 in Aug 2026 and 44,501 in Nov 2026 (partial). "
         "Nov 2026 is still running.")
    out = writer._bracket_partial_months(s, spans)
    assert "Jul 2026 (19-31 Jul)" in out and "Nov 2026 (1-19 Nov)." in out and "Aug 2026 (" not in out
    assert out.count("(1-19 Nov)") == 1
    done = "Nov 2026 (1-19 Nov) had 44,501."
    assert writer._bracket_partial_months(done, spans) == done
    assert writer._bracket_partial_months("November 2026 was slow", spans) == "November 2026 (1-19 Nov) was slow"
    assert writer._bracket_partial_months(s, {}) == s


def test_partial_month_brackets_skip_dates():
    spans = {"2026-07": "19-31 Jul", "2026-11": "1-19 Nov"}
    s = ("From 19 Jul 2026 to 19 Nov 2026, installations rose from 45,732 in Aug 2026 to 107,355 in Oct 2026, "
         "then 44,501 in the partial Nov 2026 period.")
    out = writer._bracket_partial_months(s, spans)
    assert "19 Jul 2026 (" not in out and "19 Nov 2026 (" not in out
    assert "partial Nov 2026 (1-19 Nov) period" in out
