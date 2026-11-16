from datetime import date

import pytest

from gmr_agents.presentation import feeder_compare as fc

PRIOR = [{"question": "which feeder is meter AL1157377 on", "context": {}}]


@pytest.mark.parametrize("q, meter", [
    ("compare outages of all feeders in the substation for consumer AL1157377", "AL1157377"),
    ("compare AL1157377 with the other feeders of its substation", "AL1157377"),
    ("feeder-wise outage comparison for meter al1157377", "AL1157377"),
    ("AL1157049 vs all feeders of its substation", "AL1157049"),
    ("show the outage match of SC10025001 across feeders", "SC10025001"),
    ("comparative analysis of the mapping for AL1157049", "AL1157049"),
])
def test_comparison_questions_trigger(q, meter):
    assert fc.request(q)["meter"] == meter


@pytest.mark.parametrize("q", [
    # One consumer, but no comparison asked: the pipeline answers just the question (user, 16 Nov).
    "which feeder is AL1157049 on?",
    "mapping of meter AL1157377",
    "why was al1157377 reassigned",
    "auto indexing result for SC10025001",
    "what is the verdict for AL1157377",
    # Other domains / no consumer.
    "consumption of meter AL1157377",
    "compare consumption of AL1157377 with last month",
    "outages of meter AL1157377",
    "voltage of AL1157377 feeder",
    "installs in agra",
    "which feeders have the most reassignments",
    "compare it with july",
])
def test_non_comparison_questions_do_not_trigger(q):
    assert fc.request(q, PRIOR) is None


def test_consumer_number_and_month():
    r = fc.request("compare all feeders for consumer no 8902396000 in july 2026")
    assert r == {"meter": None, "consumer_id": "8902396000", "month": date(2026, 7, 1)}


def test_follow_up_takes_the_previous_meter():
    assert fc.request("compare all feeders of this consumer's substation", PRIOR)["meter"] == "AL1157377"
    assert fc.request("compare all feeders of this consumer's substation", []) is None


def test_may_is_not_a_month_without_a_year():
    assert fc.request("may I compare AL1157377 with the other feeders")["month"] is None


def test_serial_compared_with_consumer_id_is_sent_back():
    from gmr_agents.checks import entity_check
    bad = ("SELECT meter_no, feeder_code FROM hes_gold.dim_meter WHERE regexp_replace(TRIM(consumer_id), '^0+', '') = "
           "regexp_replace('AL1157049', '^0+', '')")
    assert any("meter serial" in p for p in entity_check.check("which feeder is consumer AL1157049 on?", [bad]))
    ok = ("SELECT meter_no FROM hes_gold.dim_meter WHERE regexp_replace(TRIM(consumer_id), '^0+', '') = "
          "regexp_replace('8902396000', '^0+', '')")
    assert not any("meter serial" in p for p in entity_check.check("feeder of consumer 8902396000", [ok]))
