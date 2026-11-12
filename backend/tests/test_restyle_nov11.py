"""Chart restyle / conversation recap code paths (presentation/restyle.py) and their main.py wiring."""

import unittest

from gmr_agents.presentation import restyle
from gmr_agents.resolvers import multi_question

PREV = {"summary": "435,935 meters were installed May-Oct 2026.", "kpis": [], "insights": ["x"],
        "tables": [{"id": "primary_table", "columns": ["Month", "Installed"], "rows": []}],
        "charts": [{"id": "primary_chart", "title": "Installed by month", "type": "line",
                    "x": ["May", "Jun", "Jul", "Aug", "Sep", "Oct"], "y": [10, 50, 30, 80, 60, 70],
                    "dashboard": "mi_progress", "point_keys": ["a"] * 6}],
        "debug": {"trace_id": "abc"}, "suggested_followups": ["f1"]}


class RequestTest(unittest.TestCase):
    def test_matches(self):
        cases = {"make that a bar chart": ("bar", None), "Show it as a pie chart": ("pie", None),
                 "convert this to a line graph": ("line", None), "can you make it a donut": ("donut", None),
                 "as a horizontal bar chart please": ("horizontal_bar", None),
                 "change the chart to a stacked bar": ("stacked_bar", None),
                 "now as a bar sorted descending": ("bar", "desc"), "in a pie chart": ("pie", None),
                 "make it a bar chart sorted descending": ("bar", "desc"),
                 "show that as a table": ("table", None), "switch to line chart": ("line", None)}
        for q, want in cases.items():
            self.assertEqual(restyle.restyle_request(q), want, q)

    def test_real_questions_do_not_match(self):
        for q in ("Show monthly installations as a bar chart", "Make a pie chart of defective meters by defect type",
                  "show pending SAT by project", "Show feeder loss by project for July", "Give me a line chart of SAT % by month",
                  "make that a bar chart for Agra only"):
            self.assertIsNone(restyle.restyle_request(q), q)

    def test_not_split_into_two_questions(self):
        parts, _ = multi_question.split("make it a bar chart sorted descending")
        self.assertEqual(len(parts), 1)


class RestyleTest(unittest.TestCase):
    def test_bar(self):
        r = restyle.restyle(PREV, "bar", None)
        c = r["charts"][0]
        self.assertEqual(c["type"], "bar")
        self.assertEqual(c["x"], PREV["charts"][0]["x"])
        self.assertNotIn("dashboard", c)
        self.assertEqual(r["tables"], PREV["tables"])
        self.assertEqual(r["debug"]["trace_id"], "abc")          # table paging keeps reading the original query
        self.assertEqual(PREV["charts"][0]["type"], "line")      # original untouched

    def test_sorted(self):
        c = restyle.restyle(PREV, "bar", "desc")["charts"][0]
        self.assertEqual(c["y"], [80, 70, 60, 50, 30, 10])
        self.assertEqual(c["x"][0], "Aug")

    def test_pie_fallback_explained(self):
        many = {**PREV, "charts": [{**PREV["charts"][0], "x": [str(i) for i in range(12)], "y": list(range(12))}]}
        r = restyle.restyle(many, "pie", None)
        self.assertNotEqual(r["charts"][0]["type"], "pie")
        self.assertIn("cannot show it well", r["summary"])

    def test_no_chart(self):
        self.assertIsNone(restyle.restyle({**PREV, "charts": []}, "bar", None))
        self.assertIsNotNone(restyle.restyle({**PREV, "charts": []}, "table", None))


class RecapTest(unittest.TestCase):
    def test_is_recap(self):
        for q in ("Summarise what we have found so far in this conversation", "recap this chat",
                  "What have we found so far?", "give me a summary of our discussion"):
            self.assertTrue(restyle.is_recap(q), q)
        for q in ("Summarise pending SAT by project", "Give me a two-line executive summary of this chart",
                  "summary of O&M tickets this month"):
            self.assertFalse(restyle.is_recap(q), q)

    def test_recap(self):
        msgs = [{"role": "user", "content": "Show pending SAT by project"},
                {"role": "assistant", "content": "Across all projects 526,493 meters are pending SAT. Agra leads.",
                 "response": {}},
                {"role": "user", "content": "only Agra"},
                {"role": "assistant", "content": "x", "response": {"refused": True}},
                {"role": "user", "content": "Kashi by zone"},
                {"role": "assistant", "content": "Kashi has 75,066 pending, most in Varanasi Urban.", "response": {}}]
        got = restyle.recap(msgs)
        self.assertEqual(len(got["tables"][0]["rows"]), 2)
        self.assertIn("526,493", got["summary"])
        self.assertIn("75,066", got["summary"])


if __name__ == "__main__":
    unittest.main()
