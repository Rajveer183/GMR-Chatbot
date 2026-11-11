"""checks/freshness.py: 'data up to' per table + stale warning, from the data skeleton."""
import unittest

from gmr_agents.checks import freshness, grounding

SKELETON = {
    "built_at": "2026-11-11T03:30:00+00:00",
    "tables": [
        {"table": "low_voltage_pockets", "dates": {"data_date": ["2026-06-01", "2026-08-01"]}},
        {"table": "gold_om_closed_tickets_day_agg", "dates": {"closed_date": ["2024-09-07", "2026-11-10"]}},
        {"table": "old_daily", "dates": {"reading_date": ["2026-01-01", "2026-10-20"]}},
        {"table": "feeder_loss", "dates": {"data_date": ["2026-06-01", "2026-07-01"],
                                           "current_date": ["2026-11-11", "2026-11-11"]}},
        {"table": "revenue_receivables", "dates": {"created_at": ["2026-08-31 20:11:06", "2026-09-02 07:22:30"]}},
        {"table": "plan", "dates": {"due_date": ["2026-01-01", "2027-03-31"]}},
        {"table": "two_cols", "dates": {"install_date": ["2025-01-02", "2026-11-10"],
                                        "sat_date": ["2025-01-02", "2026-09-15"]}},
        {"table": "monthly_old", "dates": {"month_date": ["2026-01-01", "2026-08-01"]}},
        {"table": "monthly_current", "dates": {"month": ["2025-01-01", "2026-10-01"]}},
    ],
}


def one(table, sql=""):
    got = freshness.check([table], sql, SKELETON)
    return got[0] if got else None


class FreshnessTests(unittest.TestCase):
    def test_daily_data_ending_on_the_1st_is_not_monthly(self):
        f = one("low_voltage_pockets")
        self.assertFalse(f["monthly"])
        self.assertTrue(f["stale"])
        self.assertEqual(freshness.label(f), "data up to 1 Aug 2026")
        self.assertEqual(f["behind"], "102 days")

    def test_month_column_months_behind(self):
        f = one("monthly_old")
        self.assertTrue(f["monthly"] and f["stale"])
        self.assertEqual(freshness.label(f), "data up to Aug 2026")
        self.assertEqual(f["behind"], "3 months")

    def test_last_month_is_current_for_monthly(self):
        self.assertFalse(one("monthly_current")["stale"])

    def test_daily_up_to_yesterday_is_fresh(self):
        f = one("public.gold_om_closed_tickets_day_agg")
        self.assertFalse(f["stale"])
        self.assertEqual(freshness.label(f), "data up to 10 Nov 2026")

    def test_daily_weeks_behind_is_stale(self):
        f = one("old_daily")
        self.assertTrue(f["stale"])
        self.assertEqual(f["behind"], "22 days")

    def test_bookkeeping_and_today_columns_ignored(self):
        self.assertIsNone(one("revenue_receivables"))
        self.assertEqual(one("feeder_loss")["column"], "data_date")

    def test_future_dates_ignored(self):
        self.assertIsNone(one("plan"))

    def test_column_used_by_sql_wins(self):
        self.assertEqual(one("two_cols", "WHERE sat_date >= DATE '2026-01-01'")["column"], "sat_date")
        self.assertEqual(one("two_cols")["column"], "install_date")

    def test_unknown_table_skipped(self):
        self.assertEqual(freshness.check(["nope"], "", SKELETON), [])
        self.assertEqual(freshness.check([], "", {}), [])

    def test_warnings_group_tables_with_same_date(self):
        items = freshness.check(["low_voltage_pockets", "feeder_loss", "gold_om_closed_tickets_day_agg"], "", SKELETON)
        w = freshness.warnings(items)
        self.assertEqual(len(w), 2)
        self.assertIn("low_voltage_pockets: data up to 1 Aug 2026", w[0])

    def test_citation_carries_data_up_to(self):
        fresh = freshness.check(["low_voltage_pockets"], "", SKELETON)
        cites = grounding.data_citations({"tables": ["hes_gold.low_voltage_pockets"]}, {}, None, fresh)
        self.assertEqual(cites[0]["location"], "hes_gold.low_voltage_pockets (data up to 1 Aug 2026)")


if __name__ == "__main__":
    unittest.main()
