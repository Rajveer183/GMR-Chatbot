"""A question the router matched to a verified example word for word falls back to that
example's own SQL when the planner gives up on it (graph/nodes.py _verified_example_plan)."""

from gmr_agents.agents.sql_writer.agent import _verified_example_plan
from gmr_agents.catalog import business_rules


def _index(question):
    return next(i for i, e in enumerate(business_rules.examples()) if question in e["q"])


def test_exact_routed_wording_gets_the_example_sql():
    i = _index("Estimate PMPM billing value for this month.")
    plan = _verified_example_plan({"source": "router", "match_ratio": 1.0, "examples": [f"e{i}"]})
    assert plan["sql"] == str(business_rules.examples()[i]["sql"]).strip()
    assert plan["answer_kind"] != "unanswerable"


def test_extra_queries_come_along():
    i = _index("Why is SAT 13 so much lower than the others?")
    plan = _verified_example_plan({"source": "router", "match_ratio": 0.98, "examples": [f"e{i}"]})
    assert plan["extra_queries"] and "sat_milestones" in plan["extra_queries"][0]["sql"]


def test_loose_or_unrouted_matches_keep_the_planner():
    i = _index("Estimate PMPM billing value for this month.")
    # 0.85..0.95 can carry an extra filter ('... in Kashi') the example SQL does not have.
    assert _verified_example_plan({"source": "router", "match_ratio": 0.9, "examples": [f"e{i}"]}) is None
    assert _verified_example_plan({"source": "llm", "match_ratio": 1.0, "examples": [f"e{i}"]}) is None
    assert _verified_example_plan({"source": "router", "match_ratio": 1.0, "examples": []}) is None


# --- verified-SQL fast path (SQL_VERIFIED_DIRECT) ---------------------------------------- #

from unittest import mock  # noqa: E402

from gmr_agents.agents.router import triage as RT  # noqa: E402
from gmr_agents.agents.sql_writer import agent as A  # noqa: E402
from shared import config  # noqa: E402


def _direct(question, on=True):
    triage = RT._route({"question": question, "trace": [], "trace_id": "t"}, None)
    with mock.patch.object(config, "SQL_VERIFIED_DIRECT", on):
        return A._verified_direct_plan(triage, question) if triage else None


def _first_wording(pred):
    return next(e["q"][0] for e in business_rules.examples() if pred(e))


def test_direct_path_is_off_unless_enabled():
    q = _first_wording(lambda e: "gold_dashboard_stats_snapshot" in e["sql"])
    assert _direct(q, on=False) is None


def test_own_wording_of_a_dashboard_card_example_runs_its_sql():
    i = next(i for i, e in enumerate(business_rules.examples()) if e["q"][0] == "How many eligible meters in Kashi are still to pass SAT?")
    plan = _direct("how many eligible meters in kashi are still to pass SAT")  # case / punctuation aside
    assert plan is not None and plan["sql"] == str(business_rules.examples()[i]["sql"]).strip()


def test_other_wordings_of_the_example_keep_the_planner():
    # The same Kashi SQL lists these as patterns; run as-is they gave Kashi's figure for Agra.
    assert _direct("Agra mein SAT kitna pending hai?") is None
    assert _direct("How many meters are pending SAT?") is None


def test_dated_sql_under_a_relative_wording_keeps_the_planner():
    assert A._SQL_DATE_RE.search("WHERE period_start >= DATE '2026-09-01'")
    assert A._RELATIVE_TIME_RE.search("How many meters were installed last month?")
    assert not A._RELATIVE_TIME_RE.search("How much was supplied in Feb '26 and how did it compare with Jan '26?")


def test_planner_chart_questions_keep_the_planner():
    sql = "SELECT zone, SUM(meter_count) AS installed FROM public.some_table GROUP BY zone"
    assert not A._code_presented(sql, "installs by zone")
    assert A._code_presented("SELECT SUM(meter_count) AS installed FROM public.some_table", "total installs")
