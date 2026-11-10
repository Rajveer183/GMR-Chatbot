"""'Why' about a difference or change is a data question (10 Nov 2026: two were triaged explain and fetched no figures)."""

from gmr_agents.agents.router import triage as RT


def _parse(q, **kw):
    base = {"kind": "explain", "question": q, "domain": "wfm", "topics": [], "tables": []}
    return RT._parse_triage({**base, **kw}, {"question": q}, "t")


def test_why_with_a_comparison_is_data():
    assert _parse("Why is the voltage deviation in Kashi higher than in Agra?")["kind"] == "data"
    assert _parse("Why does the installation curve sag roughly every seven days?")["kind"] == "data"


def test_definitional_why_stays_explain():
    assert _parse("Why is a load violation counted only when demand is above 160% of the sanctioned load?")["kind"] == "explain"


def test_meter_lookup_gets_the_install_tables():
    t = _parse("meter HP8500592 kab se SAT ka wait kar raha hai?", kind="data", tables=["gold_non_sat_ageing_snapshot"])
    assert "unified_installations" in t["tables"]
