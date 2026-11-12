"""Investigation (presentation/investigate.py) and investigation memory (ops/investigations.py)."""

import os
import tempfile
import unittest
from unittest import mock

import pandas as pd

from gmr_agents.presentation import investigate
from gmr_agents.ops import investigations

TODAY = pd.Timestamp("2026-11-11")


def _long(rows, cols=("month", "zone", "installed")):
    df = pd.DataFrame(rows, columns=list(cols))
    df[cols[0]] = pd.to_datetime(df[cols[0]])
    return df


class AskedTest(unittest.TestCase):
    def test_triggers(self):
        for q in ("Why did installations drop last month?", "Investigate the fall in closures",
                  "Root-cause the change in feeder loss", "What could explain the rise in defective meters?",
                  "Test each explanation against the data"):
            self.assertTrue(investigate.asked(q), q)

    def test_plain_questions_do_not_trigger(self):
        for q in ("Show pending SAT by project", "How many meters were installed yesterday?",
                  "Forecast installations for next month"):
            self.assertFalse(investigate.asked(q), q)


class AssessTest(unittest.TestCase):
    def test_not_a_why_question(self):
        df = _long([("2026-09-01", "A", 10), ("2026-10-01", "A", 5), ("2026-09-01", "B", 4), ("2026-10-01", "B", 4)])
        self.assertEqual(investigate.assess("Show installations by zone", df, today=TODAY), ("", None))

    def test_localised_drop_across_two_splits(self):
        main = _long([("2026-09-01", "Z1", 500), ("2026-10-01", "Z1", 200),
                      ("2026-09-01", "Z2", 300), ("2026-10-01", "Z2", 290),
                      ("2026-09-01", "Z3", 200), ("2026-10-01", "Z3", 210)])
        by_type = _long([("2026-09-01", "1PH", 800), ("2026-10-01", "1PH", 520),
                         ("2026-09-01", "3PH", 150), ("2026-10-01", "3PH", 130),
                         ("2026-09-01", "LTCT", 50), ("2026-10-01", "LTCT", 50)], ("month", "meter_type", "installed"))
        text, out = investigate.assess("Why did installations drop last month?", main, [("by meter type", by_type)], today=TODAY)
        self.assertEqual(len(out["dimensions"]), 2)
        self.assertEqual(out["lead"]["name"], "Z1")              # 300 of 300 drop vs 1PH 280 of 300
        self.assertEqual(out["contradictions"], [])
        self.assertIn("localised in one zone -- SUPPORTED", text)
        self.assertIn("data coverage -- NOT SUPPORTED", text)
        self.assertEqual(out["confidence"]["level"], "high")
        self.assertIn("1,000 -> 700", text)

    def test_coverage_gap_overrides(self):
        # June: Agra only; July: Agra + Kashi zones -> the 'rise' is Kashi starting to report.
        main = _long([("2026-06-01", "AGRA-1", 100), ("2026-07-01", "AGRA-1", 104),
                      ("2026-06-01", "AGRA-2", 80), ("2026-07-01", "AGRA-2", 78),
                      ("2026-07-01", "VARANASI-1", 400), ("2026-07-01", "VARANASI-2", 350)],
                     ("month", "zone", "outage_count"))
        text, out = investigate.assess("Why has outage frequency increased?", main, today=TODAY)
        cov = [h for h in out["dimensions"][0]["hypotheses"] if h["h"] == "data coverage"][0]
        self.assertEqual(cov["status"], "SUPPORTED")
        self.assertIn("like-for-like", cov["evidence"])
        self.assertIn("180 -> 182", cov["evidence"])
        self.assertEqual(out["confidence"]["level"], "low")

    def test_whole_period_missing(self):
        main = _long([("2026-07-01", "Z1", 0), ("2026-08-01", "Z1", 40), ("2026-07-01", "Z2", 0), ("2026-08-01", "Z2", 60)])
        text, out = investigate.assess("Why did it rise?", main, today=TODAY)
        self.assertIn("no figures at all", text)

    def test_contradiction_between_splits(self):
        main = _long([("2026-09-01", "Z1", 500), ("2026-10-01", "Z1", 400), ("2026-09-01", "Z2", 300), ("2026-10-01", "Z2", 300)])
        other = _long([("2026-09-01", "OEM1", 400), ("2026-10-01", "OEM1", 300), ("2026-09-01", "OEM2", 300),
                       ("2026-10-01", "OEM2", 300)], ("month", "oem", "installed"))
        text, out = investigate.assess("Why did installations fall?", main, [("by oem", other)], today=TODAY)
        self.assertTrue(out["contradictions"])
        self.assertIn("CONTRADICTION", text)
        self.assertNotEqual(out["confidence"]["level"], "high")

    def test_rate_measure(self):
        main = _long([("2026-06-01", "C1", 10.0), ("2026-07-01", "C1", 25.0), ("2026-06-01", "C2", 12.0),
                      ("2026-07-01", "C2", 12.5), ("2026-06-01", "C3", 8.0), ("2026-07-01", "C3", 8.2)],
                     ("month", "circle", "loss_pct"))
        text, out = investigate.assess("Root-cause the change in feeder loss", main, today=TODAY)
        d = out["dimensions"][0]
        self.assertTrue(d["rate"])
        self.assertIn("points", text)
        self.assertNotIn("% of the change", text)
        self.assertEqual(out["lead"]["name"], "C1")

    def test_wide_shape(self):
        df = pd.DataFrame({"zone": ["A", "B", "C"], "prev_month_closed": [100, 80, 60], "curr_month_closed": [40, 82, 58]})
        text, out = investigate.assess("Investigate why closures fell", df, today=TODAY)
        self.assertEqual(out["lead"]["name"], "A")

    def test_snapshot_without_periods(self):
        df = pd.DataFrame({"project": ["Agra", "Kashi"], "sat_pct": [80.0, 70.0]})
        self.assertEqual(investigate.assess("Why is Kashi's SAT % lower than Agra's?", df, today=TODAY), ("", None))

    def test_partial_period_left_out(self):
        main = _long([("2026-09-01", "Z1", 500), ("2026-10-01", "Z1", 200), ("2026-11-01", "Z1", 20),
                      ("2026-09-01", "Z2", 300), ("2026-10-01", "Z2", 290), ("2026-11-01", "Z2", 10)])
        text, out = investigate.assess("Why did installations drop?", main, today=TODAY)
        self.assertEqual(out["periods"], ["Sep 2026", "Oct 2026"])
        self.assertIn("still in progress", text)


class RealDataShapesTest(unittest.TestCase):
    """Shapes seen on live data, 11 Nov."""

    def test_premise_wrong(self):
        main = _long([("2026-09-01", "kashi", 23150), ("2026-10-01", "kashi", 45322), ("2026-09-01", "agra", 24070),
                      ("2026-10-01", "agra", 34757), ("2026-09-01", "triveni", 20433), ("2026-10-01", "triveni", 27238)],
                     ("month", "project", "installed"))
        text, out = investigate.assess("Why did meter installations drop last month?", main, today=TODAY)
        self.assertIn("PREMISE CHECK", text)
        self.assertIn("ROSE", text)
        self.assertEqual(out["lead"]["name"], "kashi")       # 56% of the change from 34% of the base
        text2, _ = investigate.assess("Why did meter installations rise last month?", main, today=TODAY)
        self.assertNotIn("PREMISE", text2)

    def test_follows_volume_is_not_a_lead(self):
        main = _long([("2026-09-01", "CONSUMER", 64893), ("2026-10-01", "CONSUMER", 103870), ("2026-09-01", "DT", 2743),
                      ("2026-10-01", "DT", 3417), ("2026-09-01", "FEEDER", 17), ("2026-10-01", "FEEDER", 30)],
                     ("month", "category", "installed"))
        text, out = investigate.assess("Why did installations rise?", main, today=TODAY)
        self.assertIsNone(out["lead"])
        self.assertIn("follows volume", text)

    def test_single_period_is_a_coverage_gap(self):
        df = _long([("2026-07-01", "VARANASI I", 9000), ("2026-07-01", "VARANASI II", 14680), ("2026-07-01", "AZAMGARH", 15097)],
                   ("data_month", "zone", "total_outages"))
        text, out = investigate.assess("Why has the outage frequency increased in Varanasi?", df, today=TODAY)
        self.assertIn("ONE PERIOD ONLY", text)
        self.assertIn("do NOT say the earlier data does not exist", text)     # unknown table: never claims it is missing
        with mock.patch.object(investigate, "_table_starts", return_value=pd.Timestamp("2026-07-01")):
            self.assertIn("source table starts at Jul 2026", investigate.assess(
                "Why has the outage frequency increased in Varanasi?", df, today=TODAY, tables=["t"])[0])
        with mock.patch.object(investigate, "_table_starts", return_value=pd.Timestamp("2026-06-01")):
            self.assertIn("from Jun 2026", investigate.assess(
                "Why has the outage frequency increased in Varanasi?", df, today=TODAY, tables=["t"])[0])
        self.assertTrue(out["coverage_gap"])
        self.assertEqual(out["confidence"]["level"], "low")

    def test_coverage_tests_like_for_like_only(self):
        main = _long([("2026-06-01", "A1", 100), ("2026-07-01", "A1", 150), ("2026-06-01", "A2", 100),
                      ("2026-07-01", "A2", 160), ("2026-06-01", "A3", 100), ("2026-07-01", "A3", 140),
                      ("2026-07-01", "K1", 900), ("2026-07-01", "K2", 800)], ("month", "zone", "outages"))
        text, out = investigate.assess("Why did outages increase?", main, today=TODAY)
        self.assertIn("zone (like-for-like)", text)
        self.assertNotIn("K1 0 ->", text.split("H:")[-1])


class MemoryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.p = mock.patch.object(investigations, "_path", lambda: os.path.join(self.tmp, "inv.jsonl"))
        self.p.start()

    def tearDown(self):
        self.p.stop()

    def _inv(self):
        main = _long([("2026-09-01", "Z1", 500), ("2026-10-01", "Z1", 200), ("2026-09-01", "Z2", 300),
                      ("2026-10-01", "Z2", 290), ("2026-09-01", "Z3", 200), ("2026-10-01", "Z3", 210)])
        return investigate.assess("Why did installations drop?", main, today=TODAY)[1]

    def test_record_related_recall(self):
        investigations.record(trace_id="t1", session_id="s1", question="Why did meter installations drop last month?",
                              investigation=self._inv(), plan={"sql": "SELECT 1", "data_engine": "postgres"},
                              summary="Installations fell 1,000 -> 700.", insights=["Review crew deployment in Z1"])
        rel = investigations.related("Why have installations dropped?")
        self.assertEqual(rel[0]["id"], "t1")
        self.assertEqual(investigations.related("What is SAT percentage?"), [])
        got = investigations.recall("Have we investigated installation drops before?")
        self.assertIn("Z1", got["summary"])
        self.assertIn("Review crew deployment", got["summary"])
        self.assertEqual(got["tables"][0]["rows"][0]["Confidence"], "medium")  # one split only

    def test_recall_empty(self):
        self.assertIn("No earlier investigation", investigations.recall("Have we investigated outages before?")["summary"])

    def test_recall_regex(self):
        for q in ("Have we investigated installation drops before?", "What did we find last time about closures?",
                  "Has it improved since we investigated?", "Did the recommendation work?"):
            self.assertTrue(investigations.is_recall(q), q)
        for q in ("Why did installations drop?", "Show pending SAT by project", "Investigate the fall in closures",
                  "How many meters improved since last month?"):
            self.assertFalse(investigations.is_recall(q), q)

    def test_recheck_compares_lead(self):
        investigations.record(trace_id="t2", session_id=None, question="Why did installations drop?",
                              investigation=self._inv(), plan={"sql": "SELECT 1", "data_engine": "postgres"},
                              summary="", insights=[])
        newer = _long([("2026-10-01", "Z1", 200), ("2026-11-01", "Z1", 420), ("2026-10-01", "Z2", 290),
                       ("2026-11-01", "Z2", 300), ("2026-10-01", "Z3", 210), ("2026-11-01", "Z3", 200)])
        with mock.patch("gmr_agents.execution.executor.run_sql", return_value=newer), \
                mock.patch("pandas.Timestamp.now", return_value=pd.Timestamp("2026-12-05")):
            got = investigations.recheck("t2")
        self.assertTrue(got["ok"], got)
        self.assertFalse(got["same_window"])
        self.assertIn("Z1: 200 -> 420", got["summary"])


if __name__ == "__main__":
    unittest.main()


class RepairHintTest(unittest.TestCase):
    """planner._repair_hint: subquery alias missing a column; column missing from every table used."""

    class _Col:
        def __init__(self, n):
            self.column_name = n

    class _T:
        def __init__(self, name, cols, db="public"):
            self.table_name, self.database = name, db
            self.columns = [RepairHintTest._Col(c) for c in cols]

    class _Cat:
        def __init__(self, tables):
            self.t = {t.table_name: t for t in tables}

        def get_table(self, n):
            return self.t.get(n.lower())

        def list_tables(self):
            return list(self.t.values())

    def setUp(self):
        self.cat = self._Cat([self._T("snap", ["project", "zone", "meter_count"]),
                              self._T("ageing", ["project", "zone", "age_key", "meter_count"]),
                              self._T("installs", ["project", "oem", "meter_count"])])

    def test_subquery_alias(self):
        from gmr_agents.agents.sql_writer import planner as PL
        sql = "SELECT z.zone FROM (SELECT project FROM public.snap UNION ALL SELECT project FROM public.ageing) z"
        h = PL._repair_hint(sql, "Column 'zone' could not be resolved for table: 'z'. Line: 1", self.cat)
        self.assertIn("'z' is a subquery", h)
        self.assertNotIn("Tables that DO have", h)       # zone exists in the tables used

    def test_column_in_another_table(self):
        from gmr_agents.agents.sql_writer import planner as PL
        h = PL._repair_hint("SELECT oem FROM public.snap", "Unknown column: oem", self.cat)
        self.assertIn("Tables that DO have 'oem'", h)
        self.assertIn("installs", h)
        self.assertNotIn("is a subquery", h)


if __name__ == "__main__":
    unittest.main()


class SplitSqlTest(unittest.TestCase):
    COLS = {"project": "varchar", "discom": "varchar", "zone": "varchar", "circle": "varchar",
            "progress_category": "varchar", "meter_count": "bigint", "period_start": "date"}

    def test_zone_query_gets_place_and_category_splits(self):
        sql = ("SELECT zone, SUM(meter_count) AS n FROM public.t WHERE period_start IN (DATE '2026-09-01', "
               "DATE '2026-10-01') GROUP BY zone ORDER BY n")
        got = investigate.split_sqls(sql, self.COLS, "postgres")
        self.assertEqual([d for d, _ in got], ["project", "progress_category"])
        self.assertIn("GROUP BY project", got[0][1])
        self.assertNotIn("zone", got[0][1])

    def test_pinned_filter_is_not_a_split(self):
        sql = "SELECT zone, SUM(meter_count) AS n FROM public.t WHERE LOWER(project) = 'agra' GROUP BY zone"
        self.assertEqual([d for d, _ in investigate.split_sqls(sql, self.COLS, "postgres")], ["circle", "progress_category"])

    def test_complex_queries_untouched(self):
        for sql in ("SELECT zone, SUM(meter_count) FROM public.t GROUP BY ROLLUP(zone)",
                    "WITH a AS (SELECT zone, meter_count FROM public.t) SELECT zone, SUM(meter_count) FROM a GROUP BY zone",
                    "SELECT t.zone, SUM(u.meter_count) FROM public.t t JOIN public.u u ON t.id = u.id GROUP BY t.zone",
                    "SELECT zone, SUM(meter_count) FROM public.t GROUP BY 1",
                    "SELECT period_start, SUM(meter_count) FROM public.t GROUP BY period_start"):
            self.assertEqual(investigate.split_sqls(sql, self.COLS, "postgres"), [], sql)


class DoubleCountTest(unittest.TestCase):
    def test_flags_same_measure_added_to_itself(self):
        from gmr_agents.agents.sql_writer import planner as PL
        df = pd.DataFrame({"project": ["Total", "AGRA", "KASHI"], "eligible_pending": [527236, 186480, 176544],
                           "installed_not_sat": [527236, 186480, 176544], "total_pending": [1054472, 372960, 353088]})
        self.assertTrue(PL._double_counted(df))

    def test_real_breakdowns_pass(self):
        from gmr_agents.agents.sql_writer import planner as PL
        df = pd.DataFrame({"project": ["AGRA", "KASHI"], "eligible_pending": [96372, 75066],
                           "not_eligible": [12000, 9000], "total": [108372, 84066]})
        self.assertEqual(PL._double_counted(df), [])
        same = pd.DataFrame({"zone": ["A", "B"], "installed": [5, 7], "target": [5, 7]})   # equal but not summed
        self.assertEqual(PL._double_counted(same), [])
