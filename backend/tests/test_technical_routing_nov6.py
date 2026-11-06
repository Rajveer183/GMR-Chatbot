"""Technical dashboard cards (power quality, DT load, load violation) route to MDM by code (6 Nov 2026).

Before 6 Nov the triage prompt called voltage and load questions out_of_scope; the cards' hes_gold tables
are now in the MDM catalog, so their wording forces domain mdm and the card's topic first.
"""

import unittest

from gmr_agents.agents.router import triage as RT


class TechnicalRoutingTests(unittest.TestCase):
    def _parse(self, question: str, **reply) -> dict:
        base = {"kind": "out_of_scope", "question": question, "domain": "wfm", "topics": [], "tables": []}
        return RT._parse_triage({**base, **reply}, {"question": question}, "test-model")

    def test_topic_per_card(self):
        cases = {
            "What is the average voltage deviation index?": "mdm_power_quality",
            "Average VUI in Kashi": "mdm_power_quality",
            "% time with unacceptable current imbalance": "mdm_power_quality",
            "How many meters are in low-voltage pockets?": "mdm_power_quality",
            "How many DTs are overloaded?": "mdm_dt_load",
            "Top 10 transformers by peak loading": "mdm_dt_load",
            "DT load rise since last month": "mdm_dt_load",
            "What % of consumers have load violation?": "mdm_load_violation",
            "consumers exceeding sanctioned load by feeder": "mdm_load_violation",
        }
        for question, topic in cases.items():
            t = self._parse(question)
            self.assertEqual((t["kind"], t["domain"], t["topics"][0]), ("data", "mdm", topic), question)

    def test_load_violation_by_feeder_is_not_a_mapping_question(self):
        t = self._parse("% consumers with load violation by feeder")
        self.assertEqual(t["topics"][0], "mdm_load_violation")

    def test_wfm_and_power_factor_wording_is_left_alone(self):
        self.assertIsNone(RT.technical_topic("How many DT meters were installed in Agra?"))
        t = self._parse("How many DT meters were installed in Agra?", kind="data")
        self.assertEqual(t["domain"], "wfm")
        t = self._parse("Low power factor DTs by feeder")
        self.assertEqual(t["kind"], "out_of_scope")

    def test_explain_keeps_its_kind(self):
        t = self._parse("What does voltage unbalance index mean?", kind="explain")
        self.assertEqual((t["kind"], t["domain"]), ("explain", "mdm"))


if __name__ == "__main__":
    unittest.main()
