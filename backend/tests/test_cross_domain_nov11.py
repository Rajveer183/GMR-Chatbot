"""11 Nov 2026: questions naming both systems plan on both catalogs."""
from gmr_agents.agents.sql_writer import cross_domain as XD
from gmr_agents.catalog.catalog import get_catalog


def test_detects_questions_that_need_both_systems():
    t = {"domain": "wfm", "kind": "data", "topics": ["om_non_comm"], "tables": []}
    assert XD.cross_domain("Do zones with more non-comm O&M tickets show more meters stopped communicating in MDM?", t) == "mdm"
    assert XD.cross_domain("How many meters were installed in Agra?", t) is None
    assert XD.cross_domain("Feeder loss by circle", {"domain": "mdm", "kind": "data", "topics": ["mdm_energy_loss"]}) is None
    # Routed by a verified example's wording: never widened.
    assert XD.cross_domain("non-comm tickets vs voltage", {**t, "source": "router"}) is None


def test_widen_adds_other_topic_and_tables():
    t = {"domain": "wfm", "kind": "data", "topics": ["om_non_comm"], "tables": ["gold_om_open_ticket_detail_snapshot"]}
    w = XD.widen("meters stopped communicating in MDM vs non-comm tickets", t, "mdm")
    assert w["cross_domain"] == "mdm" and w["topics"][0] == "om_non_comm" and len(w["topics"]) >= 2
    assert len(w["tables"]) > 1


def test_combined_catalog_resolves_both():
    both = get_catalog("all")
    names = {t.table_name for t in both.list_tables()}
    wfm = {t.table_name for t in get_catalog("wfm").list_tables()}
    mdm = {t.table_name for t in get_catalog("mdm").list_tables()}
    assert wfm <= names and mdm <= names


def test_line_up_matches_places_across_results():
    import pandas as pd
    from gmr_agents.presentation import stats
    w = pd.DataFrame({"project": ["agra", "kashi"], "wfm_installed": [100, 200]})
    h = pd.DataFrame({"project": ["AGRA", "KASHI", "NOT PLACED"], "hes_meters": [90, 210, 5]})
    out = stats.line_up(w, [h])
    assert "- AGRA: wfm_installed 100" in out and "difference -10" in out
    assert "NOT PLACED" in out and "only in one result" in out
