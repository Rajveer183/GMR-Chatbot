"""Unit tests for the highest / lowest claim check (validation/extreme_check.py), the
meter-type entity check and the partial-period / FACTS helpers in llm_pipeline.

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

from gmr_agents.validation import entity_check, extreme_check  # noqa: E402

_DATA = {"2026-01-01": (391505, 14732, 48), "2026-02-01": (331959, 10061, 29), "2026-04-01": (152858, 6638, 63),
         "2026-06-01": (72178, 5370, 480), "2026-08-01": (42387, 3311, 33)}
CATS = pd.DataFrame([{"month": m, "category": cat, "meters_installed": v}
                     for m, vals in _DATA.items()
                     for cat, v in zip(("CONSUMER", "DT", "FEEDER", "Total"), (*vals, sum(vals)))]
                    + [{"month": "", "category": "Total", "meters_installed": 1}])


class ExtremeCheckTest(unittest.TestCase):
    def test_wrong_min_within_category(self):
        wrong = extreme_check.wrong_claims("January had the lowest feeder count (48).", CATS, "month", set())
        self.assertEqual(len(wrong), 1)
        self.assertIn("29", wrong[0])

    def test_wrong_max_after_date(self):
        self.assertTrue(extreme_check.wrong_claims("DT peaked in April 2026 at 6,638.", CATS, "month", set()))
        self.assertTrue(extreme_check.wrong_claims("Consumer was lowest in Apr 2026 (152,858).", CATS, "month", set()))

    def test_correct_claims_pass(self):
        for text in ("Feeder peaked in June at 480.", "The lowest consumer month was August 2026 with 42,387.",
                     "DT was highest in January 2026 (14,732)."):
            self.assertEqual(extreme_check.wrong_claims(text, CATS, "month", set()), [], text)

    def test_partial_period_left_out(self):
        df = pd.DataFrame({"month": ["2026-08-01", "2026-09-01", "2026-10-01"], "installs": [45731, 67653, 4106]})
        self.assertEqual(extreme_check.wrong_claims("lowest in August (45,731)", df, "month", {"2026-10-01"}), [])
        self.assertTrue(extreme_check.wrong_claims("lowest in August (45,731)", df, "month", set()))

    def test_drop_sentences(self):
        out = extreme_check.drop_sentences("Total is 1. DT peaked at 6,638. Feeder peaked at 480.", CATS, "month", set())
        self.assertEqual(out, "Total is 1. Feeder peaked at 480.")


class MeterTypeEntityTest(unittest.TestCase):
    def test_wrong_type_flagged(self):
        sql = ("SELECT SUM(defect_count) FROM public.gold_meter_defect_event_day_agg WHERE category = 'CONSUMER' "
               "AND meter_type = 'LTCT-Consumer_meter'")
        problems = entity_check.check("burnt 1ph consumer meters in august 2026", [sql + " AND period_start >= '2026-08-01'"])
        self.assertTrue(any("1PH-Consumer_meter" in p for p in problems), problems)

    def test_right_type_ok(self):
        sql = ("SELECT SUM(defect_count) FROM t WHERE category = 'CONSUMER' AND meter_type = '1PH-Consumer_meter' "
               "AND period_start >= DATE '2026-08-01'")
        self.assertEqual(entity_check.check("burnt single phase consumer meters in august 2026", [sql]), [])

    def test_athena_spelling_counts(self):
        sql = "SELECT COUNT(*) FROM u WHERE new_meter_type = 'NSM1-PH' AND mi_date >= DATE '2026-01-01'"
        self.assertEqual(entity_check.check("1 phase meters installed in 2026", [sql]), [])


class ActiveDaysTest(unittest.TestCase):
    def test_sum_active_days_flagged(self):
        sql = ("SELECT SUM(active_days) FROM public.gold_mi_technician_productivity_day_agg "
               "WHERE LOWER(zone) = 'mirzapur' AND period_start >= DATE '2026-08-01'")
        self.assertTrue(any("COUNT(DISTINCT period_start)" in p
                            for p in entity_check.check("active days in mirzapur zone in august 2026", [sql])))

    def test_count_distinct_ok(self):
        sql = ("SELECT COUNT(DISTINCT period_start) AS active_days FROM public.gold_mi_technician_productivity_day_agg "
               "WHERE LOWER(zone) = 'mirzapur' AND period_start >= DATE '2026-08-01'")
        self.assertEqual(entity_check.check("active days in mirzapur zone in august 2026", [sql]), [])


class TimeBreakdownTest(unittest.TestCase):
    ONE = ("SELECT ROUND(SUM(installation_count)::numeric / NULLIF(SUM(technician_day_count), 0), 2) "
           "FROM public.gold_mi_technician_productivity_day_agg WHERE LOWER(project) = 'kashi' "
           "AND period_start >= DATE '2026-08-01' AND period_start <= DATE '2026-08-31'")

    def test_dropped_weekly_flagged(self):
        problems = entity_check.check("productivity weekly in august 2026 in kashi", [self.ONE])
        self.assertTrue(any("weekly breakdown" in p for p in problems), problems)

    def test_weekly_bucket_ok(self):
        sql = ("SELECT week_start, SUM(installation_count) FROM (SELECT *, date_trunc('month', period_start)::date "
               "+ ((EXTRACT(DAY FROM period_start)::int - 1) / 7) * 7 AS week_start FROM t "
               "WHERE LOWER(project) = 'kashi' AND period_start >= DATE '2026-08-01') d GROUP BY ROLLUP (week_start)")
        self.assertEqual(entity_check.check("productivity weekly in august 2026 in kashi", [sql]), [])

    def test_group_by_position_ok(self):
        sql = ("SELECT date_trunc('month', period_start)::date AS month, SUM(defect_count) FROM t "
               "WHERE period_start >= DATE '2026-01-01' GROUP BY 1 ORDER BY 1")
        self.assertEqual([p for p in entity_check.check("monthly burnt meters in 2026", [sql]) if "breakdown" in p], [])

    def test_not_a_breakdown(self):
        for q in ("CXO weekly summary", "PMPM monthly billing estimate", "daily average installs in kashi in august 2026"):
            self.assertEqual([p for p in entity_check.check(q, [self.ONE]) if "breakdown" in p], [], q)


if __name__ == "__main__":
    unittest.main()
