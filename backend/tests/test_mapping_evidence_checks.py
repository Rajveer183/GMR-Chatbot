"""Mapping-audit evidence tables (mapping_audit_candidates / mapping_audit_evidence): a query must name a meter or
a feeder and pin the month -- the tables hold a row per consumer x candidate feeder (x 30-minute slot)."""

from gmr_agents.validation.entity_check import _mapping_problems

LATEST = "data_date = (SELECT MAX(data_date) FROM hes_gold.mapping_audit_summary)"


def test_one_meter_candidates_pass():
    sql = ("SELECT candidate_feeder, supporting, is_declared FROM hes_gold.mapping_audit_candidates "
           f"WHERE meter_no = 'SC11051607' AND {LATEST}")
    assert _mapping_problems("Which feeders was meter SC11051607 scored against?", [sql]) == []


def test_feeder_scoped_evidence_passes():
    sql = ("SELECT meter_no, COUNT(DISTINCT event_id) FROM hes_gold.mapping_audit_evidence "
           f"WHERE {LATEST} AND declared_feeder IN (SELECT feeder_code FROM hes_gold.mapping_audit_feeder "
           "WHERE UPPER(feeder_name) LIKE '%SAHAWAR%') GROUP BY meter_no")
    assert _mapping_problems("Proof pack for re-tags on feeder SAHAWAR", [sql]) == []


def test_unscoped_evidence_is_rejected():
    sql = f"SELECT COUNT(*) FROM hes_gold.mapping_audit_evidence WHERE {LATEST}"
    problems = _mapping_problems("How many outage windows were recorded?", [sql])
    assert any("huge" in p for p in problems)


def test_evidence_without_a_month_is_rejected():
    sql = "SELECT candidate_feeder FROM hes_gold.mapping_audit_candidates WHERE meter_no = 'SC11051607'"
    assert any("data_date" in p for p in _mapping_problems("candidate feeders of meter SC11051607", [sql]))


def test_proof_pack_subquery_is_not_a_bad_feeder_join():
    # Live 28 Oct: the proof pack joins mapping_audit and filters declared_feeder IN (SELECT feeder_code FROM
    # mapping_audit_feeder ...); the feeder-join check fired on it and cost a repair round.
    sql = ("SELECT e.meter_no FROM hes_gold.mapping_audit_evidence e JOIN hes_gold.mapping_audit a "
           f"ON a.meter_no = e.meter_no AND a.data_date = e.data_date WHERE e.{LATEST} "
           "AND e.declared_feeder IN (SELECT feeder_code FROM hes_gold.mapping_audit_feeder "
           "WHERE UPPER(feeder_name) LIKE '%SAHAWAR DEHAT%') GROUP BY e.meter_no")
    assert not any("join mapping_audit_feeder" in p for p in _mapping_problems("Proof pack for feeder SAHAWAR DEHAT", [sql]))


def test_feeder_join_without_its_key_still_fails():
    sql = ("SELECT a.meter_no FROM hes_gold.mapping_audit a JOIN hes_gold.mapping_audit_feeder f "
           f"ON f.data_date = a.data_date WHERE a.{LATEST} AND UPPER(f.feeder_name) LIKE '%BARDEEHA%'")
    assert any("join mapping_audit_feeder" in p for p in _mapping_problems("consumers moving to feeder BARDEEHA", [sql]))


def test_outage_log_counts_events_not_slots():
    # Live 28 Oct: per-feeder COUNT(*) of 30-minute slots answered '5 outages, all dark' for AL1157377.
    sql = ("SELECT candidate_feeder, declared_feeder, COUNT(*) AS slots FROM hes_gold.mapping_audit_evidence "
           f"WHERE meter_no = 'AL1157377' AND {LATEST} GROUP BY candidate_feeder, declared_feeder")
    assert any("event_id" in p for p in _mapping_problems("Show the outage windows for meter AL1157377", [sql]))


def test_per_feeder_evidence_needs_the_current_feeder():
    sql = ("SELECT candidate_feeder, COUNT(DISTINCT event_id) AS outages FROM hes_gold.mapping_audit_evidence "
           f"WHERE meter_no = 'AL4712306' AND {LATEST} GROUP BY candidate_feeder")
    assert any("current" in p for p in _mapping_problems("outage windows for meter AL4712306", [sql]))


def test_summary_joined_to_meter_table_is_rejected():
    # Live 28 Oct: 'correction queue in Triveni' joined the two on the month -> 508,125,282,840.
    sql = ("SELECT SUM(s.consumer_count) FROM hes_gold.mapping_audit_summary s JOIN hes_gold.mapping_audit a "
           f"ON s.data_date = a.data_date WHERE a.verdict = 'REASSIGN' AND s.{LATEST}")
    assert any("never JOIN" in p for p in _mapping_problems("How many consumers in the correction queue are in Triveni?", [sql]))


def test_month_pin_subquery_is_not_a_summary_join():
    sql = ("SELECT COUNT(*) FROM hes_gold.mapping_audit WHERE verdict = 'REASSIGN' AND LOWER(cluster) = 'triveni' "
           f"AND {LATEST}")
    assert not any("never JOIN" in p for p in _mapping_problems("correction queue in Triveni", [sql]))
