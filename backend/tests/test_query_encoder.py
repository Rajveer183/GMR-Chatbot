"""Unit tests for the deterministic query encoder (gmr_agents/validation/query_encoder.py).

    python -m unittest discover -s tests -v        (from backend/)
"""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from gmr_agents.validation.query_encoder import encode, prompt_line  # noqa: E402


class EncoderTests(unittest.TestCase):
    def test_entities_period_and_breakdown(self):
        e = encode("Show SAT pending in Kashi as of 20 Sep 2026 and break it down by zone.")
        self.assertEqual(e["projects"], ["kashi"])
        self.assertEqual((e["years"], e["months"]), (["2026"], ["september"]))
        self.assertTrue(e["as_of"])
        self.assertEqual(e["breakdown_by"], ["zone"])

    def test_aliases(self):
        self.assertEqual(encode("meter instalation in kasi last week")["projects"], ["kashi"])
        self.assertEqual(encode("varanasi SAT")["projects"], ["kashi"])

    def test_top_n_and_direction(self):
        self.assertEqual((encode("Top 5 divisions by installs")["top_n"], encode("Top 5 divisions")["direction"]), (5, "desc"))
        e = encode("Bottom three zones by SAT %")
        self.assertEqual((e["top_n"], e["direction"]), (3, "asc"))
        self.assertEqual(encode("Which zone had the fewest installations?")["direction"], "asc")

    def test_metric_dependent_words_get_no_direction(self):
        for q in ("Which OEM is the most reliable?", "Which project is the best?", "Who is the worst performer?"):
            self.assertNotIn("direction", encode(q), q)

    def test_signals(self):
        self.assertIn("compare", encode("Compare Agra and Kashi installations")["signals"])
        self.assertIn("share", encode("What share of installs is from Triveni?")["signals"])
        self.assertIn("change", encode("By what percent did installations change from August to September?")["signals"])
        e = encode("If Agra cuts its pending SAT by 20%, how many remain?")
        self.assertEqual(e["what_if_pct"], 20.0)
        self.assertIn("what_if", e["signals"])

    def test_sat_stage_category_oem(self):
        e = encode("Split SAT 10 by DT, Consumer and Feeder for AEW consumer meters")
        self.assertEqual(e["sat_stages"], ["SAT-10"])
        self.assertEqual(e["oems"], ["aew"])
        self.assertIn("CONSUMER", e["categories"])

    def test_relative_and_hinglish(self):
        self.assertEqual(encode("installs in the last 30 days")["relative_period"], ["last 30 days"])
        self.assertEqual(encode("Kal kitne meter lage?")["relative_period"], ["kal"])

    def test_empty_and_prompt_line(self):
        self.assertEqual(encode(""), {})
        self.assertEqual(prompt_line({}), "")
        self.assertIn("top_n=5", prompt_line(encode("top 5 divisions")))


if __name__ == "__main__":
    unittest.main()
