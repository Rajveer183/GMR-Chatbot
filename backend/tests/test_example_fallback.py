"""A question the router matched to a verified example word for word falls back to that
example's own SQL when the planner gives up on it (graph/nodes.py _verified_example_plan)."""

from gmr_agents.graph.nodes import _verified_example_plan
from gmr_agents.schema import business_rules


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
