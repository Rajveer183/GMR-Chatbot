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

from gmr_agents.checks import number_check as nc  # noqa: E402

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


def test_repair_never_cuts_inside_a_number_or_hyphenated_word():
    # 10 Nov retest: 'with 10,279 ...' was cut to 'with 10' and 'have non-SAT ...' to 'have non.'
    import pandas as pd
    from gmr_agents.checks import number_check as N
    df = pd.DataFrame({"division": ["EDD PIPARI", "EDD SORON"], "n": [10279, 9905]})
    text = ("The ten divisions are led by EDD PIPARI with 10,279 meters, followed by EDD SORON at 9,905 and 7,777 "
            "others. All ten divisions have non-SAT 120+ meters above 6,123.")
    fixed, left = N.repair(text, N.allowed_values([df]), [df])
    assert fixed == "The ten divisions are led by EDD PIPARI with 10,279 meters."
    assert not left
    df2 = pd.DataFrame({"a": [15.32, 11.47]})
    fixed, _ = N.repair("Kashi leads at 15.32, while Agra is at 11.47 – roughly 3,333 fewer – and Triveni trails.",
                        N.allowed_values([df2]), [df2])
    assert fixed == "Kashi leads at 15.32, while Agra is at 11.47 and Triveni trails."


def test_column_name_numbers_and_cross_query_differences_are_supported():
    import pandas as pd
    from gmr_agents.checks import number_check as N
    df = pd.DataFrame({"division": ["EDD PIPARI"], "non_sat_120_plus": [10279]})
    assert not N.unsupported("EDD PIPARI has 10,279 meters aged 120+ days.", N.allowed_values([df]))
    main, extra = pd.DataFrame({"installed_not_sat": [524818]}), pd.DataFrame({"eligible_pending": [263243]})
    assert not N.unsupported("Of these, 263,243 are eligible and 261,575 not yet due.", N.allowed_values([main, extra]))


def test_repair_drops_dangling_lead_in_only_when_it_cut():
    import pandas as pd
    from gmr_agents.checks import number_check as N
    m = pd.DataFrame({"n": [524818]})
    fixed, _ = N.repair("524,818 meters have not cleared SAT. Of these, 999,999 are odd.", N.allowed_values([m]), [m])
    assert fixed == "524,818 meters have not cleared SAT."
    assert N.repair("Agra leads overall. 524,818 meters.", N.allowed_values([m]), [m])[0] == "Agra leads overall. 524,818 meters."
