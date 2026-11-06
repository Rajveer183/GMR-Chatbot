"""A retrieval index that trails a hot-reloaded business_rules.yaml / examples.yaml must not crash questions
(6 Nov 2026: IndexError in example_menu and KeyError 'mdm_dt_load' in triage while the YAML was swapped under a
running backend -- 63 of 142 questions failed in that hour)."""

from gmr_agents.agents.router import triage as RT
from gmr_agents.catalog import business_rules as BR
from gmr_agents.retrieval import knowledge


def test_out_of_range_example_ids_are_skipped(monkeypatch):
    n = len(BR.examples())
    monkeypatch.setattr(knowledge, "examples_for", lambda q, k: ["0", str(n + 5)])
    assert len(BR.similar_examples("anything", 2)) == 1
    assert BR.example_menu("anything", 2).count("\n") == 0


def test_unknown_topic_keys_are_skipped(monkeypatch):
    monkeypatch.setattr(BR, "rank_topics", lambda q, k: ["no_such_topic", "mdm_power_quality"])
    topics = RT._fallback_topics("average VDI", "mdm", ["voltage_deviation_index"])
    assert "no_such_topic" not in topics and topics[0] == "mdm_power_quality"
