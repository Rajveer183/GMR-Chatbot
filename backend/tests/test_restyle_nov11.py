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


class ShowChartAgainTest(unittest.TestCase):
    """16 Nov: 'also show mw graphs fro the same' was triaged chitchat ("I can't draw charts") and got no chart."""

    def test_bare_chart_asks_match(self):
        for q in ("also show mw graphs fro the same", "show me the graph", "show graphs for the same",
                  "give me a chart for this", "plot it", "can you show the chart too", "graph dikhao",
                  "graph bhi dikhao", "same ka graph dikhao", "show me graphs of it"):
            self.assertEqual(restyle.restyle_request(q), (None, None), q)

    def test_new_chart_questions_do_not_match(self):
        for q in ("show graphs of installations in Agra", "Show monthly installations as a bar chart",
                  "show me a chart of pending SAT by project", "Make a pie chart of defective meters by defect type"):
            self.assertNotEqual(restyle.restyle_request(q), (None, None), q)

    def test_same_chart_again(self):
        r = restyle.restyle(PREV, None, None)
        self.assertEqual(r["charts"], PREV["charts"])
        self.assertEqual(r["tables"], PREV["tables"])
        self.assertIn("chart", r["summary"])
        self.assertIsNone(restyle.restyle({**PREV, "charts": []}, None, None))

    def test_text_only_turn_in_between_is_skipped(self):
        from gmr_agents import main

        class Store:
            def get_session(self, _sid):
                return {"messages": [
                    {"role": "user", "content": "Which project installed the most meters in the last 30 days?"},
                    {"role": "assistant", "content": "Kashi ...", "response": PREV},
                    {"role": "user", "content": "explain this simply"},
                    {"role": "assistant", "content": "Sure ...", "response": {"summary": "Sure", "debug": {"trace_id": "z"}}}]}

        got = main._last_charted_answer(Store(), "s")
        self.assertEqual(got["response"]["debug"]["trace_id"], "abc")


class TriageChartAskTest(unittest.TestCase):
    def _kind(self, q, kind="chitchat"):
        from gmr_agents.agents.router import triage as T
        return T._parse_triage({"kind": kind, "question": q, "domain": "wfm", "topics": ["installation"],
                                "tables": ["gold_mi_installation_day_agg"], "reply": "I can't draw charts"},
                               {"question": q}, "m")["kind"]

    def test_chart_ask_is_data(self):
        self.assertEqual(self._kind("also show mw graphs fro the same"), "data")
        self.assertEqual(self._kind("plot installations by project for last 30 days"), "data")

    def test_dashboard_ui_stays_chitchat(self):
        self.assertEqual(self._kind("change the chart type on my dashboard"), "chitchat")
        self.assertEqual(self._kind("hello"), "chitchat")
        self.assertEqual(self._kind("plot the weather chart", kind="out_of_scope"), "out_of_scope")


if __name__ == "__main__":
    unittest.main()
