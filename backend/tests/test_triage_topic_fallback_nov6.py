"""Triage replies without a valid topic key get topics by code (6 Nov 2026).

nemotron-3.5-lightning answered "topics": [1] for 'How many meters are pending SAT?'; the
query writer then had no KPI rules. The fallback puts the topic owning triage's table first.
"""

import unittest

from gmr_agents.agents.router import triage as RT


class TopicFallbackTests(unittest.TestCase):
    def _parse(self, reply: dict, question: str) -> dict:
        return RT._parse_triage(reply, {"question": question}, "test-model")

    def test_numeric_topics_fall_back_to_table_owner(self):
        t = self._parse({"kind": "data", "question": "How many meters are pending SAT?", "domain": "wfm",
                         "topics": [1], "tables": ["gold_non_sat_ageing_snapshot"]},
                        "How many meters are pending SAT?")
        self.assertEqual(t["topics"][0], "non_sat_ageing")
        self.assertLessEqual(len(t["topics"]), 2)

    def test_valid_topics_are_kept(self):
        t = self._parse({"kind": "data", "question": "What is our remaining stock?", "domain": "wfm",
                         "topics": ["inventory_utilization"], "tables": []}, "What is our remaining stock?")
        self.assertEqual(t["topics"], ["inventory_utilization"])

    def test_fallback_stays_in_domain(self):
        t = self._parse({"kind": "data", "question": "What share of auto-indexing is complete?", "domain": "mdm",
                         "topics": [], "tables": []}, "What share of auto-indexing is complete?")
        self.assertEqual(t["topics"], ["mdm_autoindexing"])

    def test_chitchat_gets_no_topics(self):
        t = self._parse({"kind": "chitchat", "question": "hi", "topics": [], "reply": "Hello"}, "hi")
        self.assertEqual(t["topics"], [])


if __name__ == "__main__":
    unittest.main()
