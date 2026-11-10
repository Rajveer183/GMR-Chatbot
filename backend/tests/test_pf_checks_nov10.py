"""Power-factor hard set (10 Nov 2026): the wrong SQL the live pipeline wrote, and the checks that now catch it.

    python -m pytest tests/test_pf_checks_nov10.py      (from backend/)
"""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from gmr_agents.checks import entity_check as ec  # noqa: E402

D = ("(SELECT MAX(month_date) FROM (SELECT month_date FROM hes_gold.dt_load_profile WHERE month_date >= "
     "DATE '2026-06-01' GROUP BY month_date HAVING COUNT(DISTINCT data_date) >= 10) c)")


def problems(q, sql):
    return ec.check(q, [sql])


def has(ps, text):
    return any(text in p for p in ps)


class LowPfAgeingTests(unittest.TestCase):
    Q = "How many meters were flagged for low power factor in both June and July?"

    def test_severity_without_ageing_flagged(self):
        sql = ("SELECT COUNT(*) FROM (SELECT meter_no FROM hes_gold.lpf2_low_power_factor_meters WHERE data_date IN "
               "(DATE '2026-06-01', DATE '2026-07-01') AND severity NOT IN ('Normal','NoData') GROUP BY meter_no "
               "HAVING COUNT(DISTINCT data_date) = 2) t")
        self.assertTrue(has(problems(self.Q, sql), "Low PF finding"))

    def test_with_ageing_ok(self):
        sql = ("SELECT COUNT(*) FROM (SELECT meter_no FROM hes_gold.lpf2_low_power_factor_meters WHERE data_date IN "
               "(DATE '2026-06-01', DATE '2026-07-01') AND severity NOT IN ('Normal','NoData') AND ageing_days >= 3 "
               "GROUP BY meter_no HAVING COUNT(DISTINCT data_date) = 2) t")
        self.assertFalse(has(problems(self.Q, sql), "Low PF finding"))

    def test_one_meter_lookup_ok(self):
        sql = (f"SELECT severity, ageing_days FROM hes_gold.lpf2_low_power_factor_meters WHERE data_date = {D} "
               "AND meter_no = 'AL1557419' AND severity <> 'NoData'")
        self.assertFalse(has(problems("PF status of meter AL1557419", sql), "Low PF finding"))


class LowPfMedianTests(unittest.TestCase):
    def test_percentile_of_rollup_mean_flagged(self):
        sql = (f"SELECT APPROX_PERCENTILE(avg_low_pf_pct, 0.5) FROM hes_gold.lpf2_low_power_factor WHERE data_date = {D} "
               "AND severity NOT IN ('Normal','NoData') AND category = 'dt_meter'")
        self.assertTrue(has(problems("Median low PF % of flagged DT meters", sql), "median / percentile"))

    def test_percentile_over_meters_ok(self):
        sql = (f"SELECT APPROX_PERCENTILE(low_pf_pct, 0.5) FROM hes_gold.lpf2_low_power_factor_meters WHERE data_date = {D} "
               "AND severity NOT IN ('Normal','NoData') AND ageing_days >= 3 AND category = 'dt_meter'")
        self.assertFalse(has(problems("Median low PF % of flagged DT meters", sql), "median / percentile"))


class PfDeteriorationBandSumTests(unittest.TestCase):
    Q = "For each project show low power factor meters and PF deterioration meters side by side"

    def test_band_sum_per_project_flagged(self):
        sql = (f"SELECT zone, SUM(unique_meter_count) FROM hes_gold.power_factor_deterioration WHERE data_date = {D} "
               "GROUP BY zone")
        self.assertTrue(has(problems(self.Q, sql), "several PF bands"))

    def test_band_rows_ok(self):
        sql = (f"SELECT circle, pf_band, SUM(unique_meter_count) FROM hes_gold.power_factor_deterioration "
               f"WHERE data_date = {D} GROUP BY circle, pf_band")
        self.assertFalse(has(problems("PF deterioration band mix by circle", sql), "several PF bands"))


class ForecastPinTests(unittest.TestCase):
    def test_future_month_pinned_flagged(self):
        sql = ("SELECT COUNT(DISTINCT meter_no) FROM hes_gold.lpf2_low_power_factor_meters "
               "WHERE data_date = DATE '2026-09-01' AND severity NOT IN ('Normal','NoData') AND ageing_days >= 3")
        q = "If the trend continues, how many meters will have low power factor in September 2026?"
        self.assertTrue(has(problems(q, sql), "later month has no rows"))

    def test_month_series_ok(self):
        sql = ("SELECT data_date, SUM(unique_meter_count) FROM hes_gold.lpf2_low_power_factor WHERE severity NOT IN "
               "('Normal','NoData') AND data_date BETWEEN DATE '2026-06-01' AND DATE '2026-08-01' GROUP BY data_date")
        q = "If the trend continues, how many meters will have low power factor in September 2026?"
        self.assertFalse(has(problems(q, sql), "later month has no rows"))


class FlatProjectionChartTests(unittest.TestCase):
    """A projection repeated on every month row was drawn as a flat line across June-August."""

    def _chart(self, cols):
        import pandas as pd
        from gmr_agents.presentation.blocks import _build_chart
        df = pd.DataFrame({"data_month": ["2026-06-01", "2026-07-01", "2026-08-01"],
                           "low_pf_meters": [585748, 711642, 665748], **cols})
        plan = {"sql": "SELECT 1", "question": "low PF meters in September", "labels": {}, "table_title": "t",
                "display": {"chart": {"type": "line", "x": "data_month", "y": ["low_pf_meters", *cols]}}, "metrics": []}
        return _build_chart(plan, df)

    def test_flat_projection_dropped(self):
        out = self._chart({"projected_sep_2026_low_pf_meters": [963430] * 3})
        self.assertNotIn("series", out)
        self.assertEqual(out["y"], [585748.0, 711642.0, 665748.0])

    def test_constant_target_kept(self):
        out = self._chart({"target": [700000] * 3})
        self.assertEqual(len(out["series"]), 2)


class ZoneStemTests(unittest.TestCase):
    SQL = (f"SELECT SUM(unique_meter_count) FROM hes_gold.lpf2_low_power_factor WHERE data_date = {D} AND "
           "severity NOT IN ('Normal','NoData') AND UPPER(zone) IN ({z})")
    Q = "how many consumers in Varanasi zones are running below 0.9 power factor most of the time?"

    def test_project_zones_for_stem_flagged(self):
        sql = self.SQL.format(z="'AZAMGARH', 'VARANASI I', 'VARANASI II'")
        self.assertTrue(has(problems(self.Q, sql), "Varanasi zones"))

    def test_stem_zones_ok(self):
        sql = self.SQL.format(z="'VARANASI I', 'VARANASI II'")
        self.assertFalse(has(problems(self.Q, sql), "Varanasi zones"))

    def test_kashi_project_ok(self):
        sql = self.SQL.format(z="'AZAMGARH', 'VARANASI I', 'VARANASI II'")
        self.assertFalse(has(problems("low PF consumers in Kashi project", sql), "zones'"))


class ProjectionTileTests(unittest.TestCase):
    def test_tile_reads_projected_row(self):
        import pandas as pd
        from gmr_agents.presentation.blocks import _blocks
        df = pd.DataFrame({"month_label": ["Jun 2026", "Jul 2026", "Aug 2026", "Sep 2026"],
                           "low_pf_meters": [585748, 711642, 665748, 963430],
                           "basis": ["actual", "actual", "actual (partial, 2-3 days)", "projected (June -> July trend)"]})
        plan = {"sql": "SELECT 1", "question": "low PF meters in September", "labels": {}, "table_title": "t",
                "display": {"chart": {"type": "bar", "x": "month_label", "y": ["low_pf_meters"]}},
                "metrics": [{"alias": "low_pf_meters"}]}
        kpi = _blocks(plan, {"result": df})["kpis"][0]
        self.assertEqual(kpi["value"], 963430)
        self.assertTrue(kpi["label"].startswith("Projected"))

    def test_plain_series_keeps_first_row(self):
        import pandas as pd
        from gmr_agents.presentation.blocks import _blocks
        df = pd.DataFrame({"month_label": ["Jun 2026", "Jul 2026"], "low_pf_meters": [585748, 711642]})
        plan = {"sql": "SELECT 1", "question": "low PF month-wise", "labels": {}, "table_title": "t",
                "display": {"chart": {"type": "bar", "x": "month_label", "y": ["low_pf_meters"]}},
                "metrics": [{"alias": "low_pf_meters"}]}
        self.assertEqual(_blocks(plan, {"result": df})["kpis"][0]["value"], 585748)


if __name__ == "__main__":
    unittest.main()
