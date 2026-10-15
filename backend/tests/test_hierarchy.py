"""Unit tests for the place hierarchy repair (gmr_agents/sql/hierarchy.py).

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

from gmr_agents.sql import hierarchy as H  # noqa: E402

MAP = {("division", "sub_division"): {"2614221-edsd chomua": "edd kosi mathura"},
       ("circle", "division"): {"edd jaunpur nagar": "edc i jaunpur"}}


class LearnTest(unittest.TestCase):
    def test_orphans_only_unique_parent(self):
        rows = [(None, "c1", "d1", "s1"), (None, "c1", None, "s1"),        # s1 orphan -> d1
                (None, "c1", "d2", "s2"), (None, "c2", "d3", "s2"), (None, None, None, "s2"),
                (None, None, "dx", "s3"), (None, None, "dy", "s3"), (None, None, None, "s3")]  # s3 ambiguous
        learnt = H._learn(rows)
        self.assertEqual(learnt[("division", "sub_division")], {"s1": "d1"})


class RewriteTest(unittest.TestCase):
    def test_group_by_rollup_and_grouping(self):
        sql = ("SELECT CASE WHEN GROUPING(division) = 1 THEN 'Total' ELSE UPPER(division) END AS division, "
               "SUM(meter_count) AS m FROM public.gold_mi_sat_invoice_snapshot GROUP BY ROLLUP (division) "
               "ORDER BY GROUPING(division) DESC, m DESC")
        out = H.rewrite(sql, MAP)
        self.assertEqual(out.count("CASE LOWER(TRIM(sub_division))"), 4)  # SELECT x2, GROUP BY, ORDER BY GROUPING

    def test_bare_projection_keeps_name_and_outer_untouched(self):
        sql = ("SELECT division, m FROM (SELECT division, SUM(meter_count) AS m "
               "FROM public.gold_mi_sat_invoice_snapshot GROUP BY division) t")
        out = H.rewrite(sql, MAP)
        self.assertTrue(out.startswith("SELECT division, m FROM (SELECT COALESCE("))
        self.assertIn("END) AS division", out)

    def test_untouched_without_mapping_or_place(self):
        sql = "SELECT SUM(meter_count) FROM public.gold_mi_sat_invoice_snapshot"
        self.assertEqual(H.rewrite(sql, MAP), sql)
        self.assertEqual(H.rewrite("SELECT division FROM public.gold_x GROUP BY division", {}),
                         "SELECT division FROM public.gold_x GROUP BY division")

    def test_athena_like_non_gold_untouched(self):
        sql = "SELECT division, COUNT(*) FROM gmr_wfm_silver.unified_installations GROUP BY division"
        self.assertEqual(H.rewrite(sql, MAP), sql)


class StockLabelTest(unittest.TestCase):
    def test_stock_blank_zone_relabelled(self):
        df = pd.DataFrame({"zone": ["Total", "AGRA", "Unmapped"], "remaining_stock": [119589, 0, 119589]})
        out = H.relabel_stock("SELECT ... FROM public.gold_meter_inventory_event_day_agg", df)
        self.assertEqual(list(out["zone"]), ["Total", "AGRA", H.IN_STOCK])

    def test_other_tables_keep_unmapped(self):
        df = pd.DataFrame({"zone": ["Unmapped"], "m": [7]})
        out = H.relabel_stock("SELECT ... FROM public.gold_mi_sat_invoice_snapshot", df)
        self.assertEqual(list(out["zone"]), ["Unmapped"])


if __name__ == "__main__":
    unittest.main()
