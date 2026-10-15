"""Unit tests for the answer number check (gmr_agents/validation/number_check.py).

    python -m unittest discover -s tests -v        (from backend/)
"""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import pandas as pd  # noqa: E402

from gmr_agents.validation import number_check as nc  # noqa: E402

SAT = pd.DataFrame({"project": ["TRIVENI", "KASHI", "AGRA"], "total_mi": [1277180, 1352702, 1636209],
                    "total_sat": [1106912, 1190471, 1457191], "pct_sat": [86.67, 88.01, 89.06]})
MONEY = pd.DataFrame({"project": ["Total"], "invoiced_cr": [561.0], "realised_cr": [369.3],
                      "collection_efficiency_pct": [65.83]})


class NumberCheckTests(unittest.TestCase):
    def test_transcription_slip_found_and_fixed(self):
        al = nc.allowed_values([SAT])
        text = "TRIVENI has 1,127,180 total MI and 1,106,912 SAT (86.67%)."
        self.assertEqual(nc.unsupported(text, al), ["1,127,180"])
        fixed, left = nc.repair(text, al, [SAT])
        self.assertIn("1,277,180", fixed)
        self.assertEqual(left, [])

    def test_derived_values_allowed(self):
        al = nc.allowed_values([SAT])
        self.assertEqual(nc.unsupported("Agra leads Kashi by 2,83,507 meters; Triveni lags by 2.39 points; "
                                        "total 4,266,091 (42.66 lakh).", al), [])

    def test_malformed_grouping(self):
        self.assertEqual(nc.unsupported("Invoiced 332,6409 meters.", nc.allowed_values([SAT])), ["332,6409"])

    def test_trailing_comma_not_part_of_number(self):
        self.assertEqual(nc.unsupported("Pending is 1,106,912, the most.", nc.allowed_values([SAT])), [])

    def test_percentage_is_not_money(self):
        al = nc.allowed_values([MONEY])
        self.assertEqual(nc.unsupported("Rs 369.3 crore of Rs 561.0 crore (65.83%).", al), [])
        self.assertIn("65.83 crore", nc.unsupported("Rs 65.83 crore realised.", al))

    def test_invented_figure_snapped_to_close_value(self):
        al = nc.allowed_values([MONEY])
        fixed, left = nc.repair("Invoiced Rs 561.37 crore.", al, [MONEY])
        self.assertEqual((fixed, left), ("Invoiced Rs 561 crore.", []))

    def test_bad_aside_removed_from_single_sentence(self):
        calc = pd.DataFrame([{"expression": "4214678 * 0.12", "result": 505761.36}])
        al = nc.allowed_values([calc], "What is 12% of 4,214,678?")
        fixed, left = nc.repair("12% of 4,214,678 = 505,761.36 (Rs 505.76 lakh).", al, [calc])
        self.assertEqual((fixed, left), ("12% of 4,214,678 = 505,761.36.", []))

    def test_unfixable_sentence_dropped_when_others_remain(self):
        al = nc.allowed_values([SAT])
        fixed, _left = nc.repair("Kashi has 1,352,702 meters. Somewhere 99,999 are missing.", al, [SAT])
        self.assertEqual(fixed, "Kashi has 1,352,702 meters.")

    def test_years_dates_small_counts_ignored(self):
        self.assertEqual(nc.unsupported("In 2026, on 2026-10-03, the top 3 projects (SAT-13).", nc.allowed_values([SAT])), [])


if __name__ == "__main__":
    unittest.main()
