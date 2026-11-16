"""16 Nov 2026 advanced-analytics additions: reports, forecast intervals + backtest, festival dates,
invented constants, scaffold cross joins, ROUND(double) repair, conditional multi-questions."""

import unittest
from datetime import date

import numpy as np
import pandas as pd

from gmr_agents.checks import entity_check
from gmr_agents.execution import executor
from gmr_agents.presentation import report, stats
from gmr_agents.resolvers import festivals, multi_question


class ReportDetectionTest(unittest.TestCase):
    def test_reports(self):
        for q in ["Generate the weekly report", "Generate a daily report for yesterday", "Weekly report for Kashi",
                  "Give me the monthly report for October with exceptions and comparisons", "report for last week",
                  "daily MIS", "Prepare the monthly MIS report for Agra"]:
            self.assertTrue(report.is_report(q), q)

    def test_not_reports(self):
        for q in ["Show the feeder loss report", "SAT report by division", "what is the status of installations",
                  "How many meters were installed yesterday?", "Give me a downloadable report of division-wise installations"]:
            self.assertFalse(report.is_report(q), q)

    def test_risks_and_actions(self):
        # 16 Nov: went to the LLM ('risk' is a topic word) -> four red flags, no actions, an invented week.
        q = "Write a CXO weekly summary with three risks and three actions."
        self.assertTrue(report.is_report(q))
        self.assertEqual(report.risk_ask(q), (3, 3))
        self.assertTrue(report.is_report("Give me the top 2 risks and 4 actions for Kashi"))
        self.assertEqual(report.risk_ask("top 2 risks and 4 actions"), (2, 4))
        self.assertEqual(report.risk_ask("weekly summary with key risks"), (3, None))
        self.assertEqual(report.risk_ask("Generate the weekly report"), (None, None))
        self.assertFalse(report.is_report("weekly theft risk summary"))
        self.assertFalse(report.is_report("What are the risks of feeder losses and actions to reduce them?"))
        r = report.request(q, date(2026, 11, 16))
        self.assertEqual((r["period"], r["start"], r["end"], r["cxo"]), ("weekly", date(2026, 11, 9), date(2026, 11, 15), True))

    def test_watchlist_ranking(self):
        r = report.request("weekly summary with three risks and three actions", date(2026, 11, 16))
        f = {
            "watch": pd.DataFrame([{"project": "agra", "open_tickets": 100, "open_over_60_days": 50, "installed_meters": 1000,
                                    "sat_meters": 900, "waiting_sat_over_90_days": 10, "last_sat_batch": "2026-11-01",
                                    "not_realised_over_60_days": 0, "defective_meters": 1, "defect_base_meters": 1000,
                                    "lumpsum_invoiced": 10.0, "lumpsum_realised": 9.5}]),
            "om_by_place": pd.DataFrame([{"project": "AGRA", "open_now": 100, "open_over_60_days": 50}]),
            "by_place": pd.DataFrame([{"project": "AGRA", "installed_this_period": 700, "installed_previous_period": 700, "change": 0}]),
            "kpis": pd.DataFrame([{"kpi": "Meters installed", "this_period": 700, "previous_period": 700}]),
            "position": pd.DataFrame([{"remaining_stock": 100000}]),
        }
        w = report._watchlist(r, f, False)
        self.assertEqual(len(w), 7)
        self.assertEqual(w[0]["risk"], "O&M ticket backlog")           # 50% vs 20% -> the only red flag
        self.assertEqual(sum(c["severity"] > 0 for c in w), 1)
        self.assertEqual([c["severity"] for c in w], sorted((c["severity"] for c in w), reverse=True))

    def test_periods(self):
        t = date(2026, 11, 16)
        r = report.request("Generate a daily report for yesterday", t)
        self.assertEqual((r["period"], r["start"], r["prev_start"]), ("daily", date(2026, 11, 15), date(2026, 11, 14)))
        r = report.request("Generate the weekly report", t)
        self.assertEqual((r["start"], r["end"], r["prev_end"]), (date(2026, 11, 9), date(2026, 11, 15), date(2026, 11, 8)))
        r = report.request("monthly report for October", t)
        self.assertEqual((r["start"], r["end"], r["prev_start"]), (date(2026, 10, 1), date(2026, 10, 31), date(2026, 9, 1)))
        r = report.request("monthly report this month for Agra", t)
        self.assertEqual((r["end"], r["prev_end"], r["project"]), (date(2026, 11, 15), date(2026, 10, 15), "agra"))
        r = report.request("monthly report for December", t)  # a month not yet reached = last year's
        self.assertEqual(r["start"], date(2025, 12, 1))
        self.assertIsNotNone(report.request("give me a report for Kashi", t)["assumed"])


class ForecastTest(unittest.TestCase):
    def test_intervals_and_backtest(self):
        d = pd.DataFrame({"month": pd.date_range("2025-11-01", periods=12, freq="MS"),
                          "installs": [50, 60, 58, 70, 75, 80, 90, 85, 95, 100, 110, 108]})
        text = stats.analyse("Forecast installations for the next 3 months with confidence intervals", d,
                             today=pd.Timestamp("2026-11-16"))
        self.assertIn("forecast per month with 95% range: Nov 2026", text)
        self.assertIn("historical check (backtest)", text)
        self.assertNotIn("Monte Carlo", text)  # 'confidence interval' no longer runs the trend-blind bootstrap

    def test_forecast_band_widens(self):
        f = stats._forecast(np.arange(1, 13, dtype=float) * 10, "month", 3)
        self.assertEqual(f["method"], "trend")
        widths = f["hi"] - f["lo"]
        self.assertTrue(widths[2] >= widths[0])

    def test_constant_date_column_is_cross_section(self):
        rng = np.random.default_rng(3)
        a = rng.normal(100, 20, 30)
        df = pd.DataFrame({"data_month": ["2026-07-01"] * 30, "circle": [f"C{i}" for i in range(30)],
                           "low_pf": a, "theft": a * 0.5 + rng.normal(0, 3, 30)})
        self.assertIn("correlation low_pf vs theft", stats.analyse("is low pf correlated with theft", df))


class FestivalTest(unittest.TestCase):
    def test_dates(self):
        t = date(2026, 11, 16)
        self.assertEqual(festivals.resolve("installed since Diwali?", t)[0][1], date(2026, 11, 8))
        self.assertEqual(festivals.resolve("installs after Holi 2025", t)[0][1], date(2025, 3, 14))
        self.assertEqual(festivals.resolve("stock before next Holi", t)[0][1], date(2027, 3, 22))
        self.assertEqual(festivals.resolve("how many installed yesterday", t), [])


class GuardsTest(unittest.TestCase):
    def test_invented_constant(self):
        q = "What would happen to the overall target if the current performance continued?"
        self.assertTrue(entity_check._invented_constants(q, ["SELECT 5000000 AS target"]))
        self.assertFalse(entity_check._invented_constants("if the target is 50 lakh", ["SELECT 5000000 AS target"]))
        self.assertFalse(entity_check._invented_constants("installs in 2026", ["SELECT 2026 AS yr"]))

    def test_round_numeric(self):
        fixed = executor._round_numeric("SELECT ROUND(PERCENT_RANK() OVER (ORDER BY x) * 100, 1) AS p FROM t")
        self.assertIn("CAST(", fixed.upper())
        self.assertIsNone(executor._round_numeric("SELECT ROUND(x) FROM t"))

    def test_conditional_not_split(self):
        self.assertEqual(len(multi_question.split(
            "What if daily installations increase by 15%? When will we finish the remaining meters?")[0]), 1)
        self.assertEqual(len(multi_question.split(
            "How many meters were installed yesterday? And what is the SAT % by project?")[0]), 2)


if __name__ == "__main__":
    unittest.main()
