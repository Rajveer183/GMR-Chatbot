"""checks/wording_check.py: pipeline-word leaks and causes the answer was never given."""

import unittest

import pandas as pd

from gmr_agents.checks import wording_check as W


class LeakTest(unittest.TestCase):
    def test_row_count_clause_dropped(self):
        self.assertEqual(W.leaks("RESULT contains 200 rows of meter complaint data; the total is 5."), "The total is 5.")

    def test_labels_rewritten(self):
        self.assertEqual(W.leaks("June is not available in RESULT; only July."), "June is not available in the data; only July.")
        self.assertNotIn("FACTS", W.leaks("Per FACTS the total is 9."))

    def test_plain_text_untouched(self):
        for t in ("Agra leads with 96,372.", "The result of the audit is clear.", "Results improved in Kashi."):
            self.assertEqual(W.leaks(t), t)


class CauseTest(unittest.TestCase):
    def setUp(self):
        df = pd.DataFrame({"day": pd.date_range("2026-09-12", periods=3), "installs": [1, 2, 3],
                           "flag": ["spike", "dip", "normal"]})
        self.v = W.vocabulary([df], "Highlight unusual days in daily meter installations")

    def test_invented_cause_dropped(self):
        t = ("15 days were flagged as dips and 25 as spikes. These anomalies are driven by irregular batch cycles and "
             "feeder audit schedules rather than sustained trends.")
        bad = W.unsupported_causes(t, self.v)
        self.assertEqual(len(bad), 1)
        self.assertEqual(W.drop_sentences(t, bad), "15 days were flagged as dips and 25 as spikes.")

    def test_cause_from_the_data_kept(self):
        v = W.vocabulary([pd.DataFrame({"project": ["kashi", "agra"], "installed": [1, 2]})], "why did installs rise")
        self.assertEqual(W.unsupported_causes("Installations rose, driven by the Kashi and Agra projects.", v), [])

    def test_never_empties_the_summary(self):
        t = "The dip is due to festival holidays and monsoon flooding."
        self.assertEqual(W.drop_sentences(t, W.unsupported_causes(t, self.v)), t)


if __name__ == "__main__":
    unittest.main()


class ImpossibleEqualityTest(unittest.TestCase):
    """sql_guard._check_impossible_equalities: one column pinned to two different single values."""

    def _check(self, sql):
        import sqlglot
        from gmr_agents.guards import sql_guard as G
        return G._check_impossible_equalities(sqlglot.parse_one(sql, read="presto"))

    def test_two_period_ctes_on_inner_joins(self):
        sql = ("WITH latest_month AS (SELECT MAX(month_date) AS lm FROM t), prev_month AS (SELECT DATE_ADD('month', -1, lm) "
               "AS pm FROM latest_month) SELECT r.zone, SUM(r.n) FROM o r JOIN latest_month lm ON r.data_date = lm.lm "
               "JOIN prev_month pm ON r.data_date = pm.pm GROUP BY r.zone")
        self.assertIn("no row can match", self._check(sql))

    def test_two_literals(self):
        self.assertTrue(self._check("SELECT * FROM t WHERE zone = 'A' AND zone = 'B'"))

    def test_legitimate_shapes_pass(self):
        for sql in ("SELECT a.project FROM a JOIN b ON a.project = b.project JOIN c ON a.project = c.project",
                    "WITH lm AS (SELECT MAX(d) m FROM t) SELECT * FROM t JOIN lm x ON t.d = x.m WHERE t.zone = 'X'",
                    "WITH lm AS (SELECT MAX(d) m FROM t), pm AS (SELECT DATE_ADD('month', -1, m) p FROM lm) "
                    "SELECT * FROM t LEFT JOIN lm ON t.d = lm.m LEFT JOIN pm ON t.d = pm.p",
                    "SELECT * FROM t WHERE zone = 'A' OR zone = 'B'"):
            self.assertIsNone(self._check(sql), sql)


class EmptyPeriodTest(unittest.TestCase):
    """planner._empty_periods: a named period's row all zero / empty although the table holds that period."""

    def setUp(self):
        from unittest import mock
        self.p = [mock.patch("gmr_agents.retrieval.data_skeleton.load", return_value={"tables": [
            {"table": "feeder_loss", "dates": {"data_date": ["2026-06-01", "2026-07-01"]}}]})]
        for p in self.p:
            p.start()

    def tearDown(self):
        for p in self.p:
            p.stop()

    def test_dropped_month_flagged(self):
        from gmr_agents.agents.sql_writer import planner as PL
        df = pd.DataFrame({"period": ["July", "June"], "feeders": [7154, 0], "loss_pct": [61.42, None]})
        self.assertEqual(PL._empty_periods(df, ["feeder_loss"]), ["Jun 2026"])

    def test_real_data_and_out_of_range_pass(self):
        from gmr_agents.agents.sql_writer import planner as PL
        self.assertEqual(PL._empty_periods(pd.DataFrame({"period": ["July", "June"], "n": [5, 4]}), ["feeder_loss"]), [])
        self.assertEqual(PL._empty_periods(pd.DataFrame({"period": ["July", "August"], "n": [5, 0]}), ["feeder_loss"]), [])
        self.assertEqual(PL._empty_periods(pd.DataFrame({"zone": ["A", "B"], "n": [5, 0]}), ["feeder_loss"]), [])


class ZeroRowsGenuineTest(unittest.TestCase):
    def test_athena_case_sensitive_filter_is_retried(self):
        from gmr_agents.agents.sql_writer import planner as PL
        self.assertFalse(PL._zero_is_genuine("SELECT 1 FROM hes_gold.outage_ranking r WHERE r.cluster = 'kashi'"))

    def test_lower_with_spaces_is_case_insensitive(self):
        from gmr_agents.agents.sql_writer import planner as PL
        self.assertTrue(PL._case_insensitive("SELECT 1 FROM t WHERE LOWER(project) = 'kashi'", "kashi"))
        self.assertTrue(PL._case_insensitive("SELECT 1 FROM t WHERE LOWER(project) IN ('kashi', 'agra')", "agra"))
        self.assertFalse(PL._case_insensitive("SELECT 1 FROM t WHERE project = 'kashi'", "kashi"))


class MissingValuesTest(unittest.TestCase):
    """pandas 3 keeps NaN through astype(str): the code checks must not crash on missing text cells."""

    def test_year_check_and_vocabulary_with_nan(self):
        import numpy as np
        from gmr_agents.checks import year_check
        df = pd.DataFrame({"zone": ["A", None, np.nan], "month": ["2026-07-01", None, "Jun 2026"], "n": [1, 2, 3]})
        self.assertEqual(year_check.fix("In July 2025 loss rose.", [df], "q")[0], "In July 2026 loss rose.")
        self.assertIn("zone", W.vocabulary([df], "q"))
