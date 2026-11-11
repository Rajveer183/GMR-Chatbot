"""ops/learning.py, alerts.py, schema_watch.py, export.py + their endpoints: feedback, root
causes, recurring failures, regression cases, schema diff, alerts and conversation export.

    python -m unittest tests.test_ops_learning_nov11 -v      (from backend/)

No database and no LLM: everything here reads logs / stored answers.
"""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from fastapi.testclient import TestClient  # noqa: E402

from gmr_agents import main  # noqa: E402
from gmr_agents.ops import alerts, export, learning, schema_watch  # noqa: E402


def q(trace, *, term="COMPLETED", kind="data", refused=False, rows=3, steps=None, question="how many meters",
      sql="SELECT 1", at="2026-11-11T10:00:00", summary="ok", error=None):
    return {"trace_id": trace, "session_id": "s1", "at": at, "question": question, "kind": kind, "refused": refused,
            "rows": rows, "sql": sql, "summary": summary, "error": error, "termination": term, "steps": steps or [],
            "execution": {"termination": term}}


class FeedbackTests(unittest.TestCase):
    def test_validation(self):
        with self.assertRaises(ValueError):
            learning.record_feedback("not a trace!", "up")
        with self.assertRaises(ValueError):
            learning.record_feedback("abcd1234", "meh")

    def test_unknown_reason_becomes_other_and_upvote_has_no_reason(self):
        self.assertEqual(learning.record_feedback("abcd1234", "down", reason="weird")["reason"], "other")
        self.assertIsNone(learning.record_feedback("abcd1234", "up", reason="wrong_number")["reason"])

    def test_endpoint_roundtrip_latest_vote_wins_and_clear_removes(self):
        c = TestClient(main.app)
        sid = "test-sess-feedback"
        self.assertEqual(c.post("/api/feedback", json={"trace_id": "feed0001", "rating": "down", "session_id": sid,
                                                       "reason": "wrong_period"}).status_code, 200)
        c.post("/api/feedback", json={"trace_id": "feed0001", "rating": "up", "session_id": sid})
        c.post("/api/feedback", json={"trace_id": "feed0002", "rating": "down", "session_id": sid})
        c.post("/api/feedback", json={"trace_id": "feed0002", "rating": "clear", "session_id": sid})
        votes = c.get("/api/feedback", params={"session_id": sid}).json()["votes"]
        self.assertEqual(votes.get("feed0001"), "up")
        self.assertNotIn("feed0002", votes)
        self.assertEqual(c.post("/api/feedback", json={"trace_id": "x", "rating": "up"}).status_code, 422)


class RootCauseTests(unittest.TestCase):
    def test_causes(self):
        self.assertIsNone(learning.root_cause(q("a")))
        self.assertEqual(learning.root_cause(q("a", term="LLM_UNAVAILABLE", refused=True)), "llm_unavailable")
        self.assertEqual(learning.root_cause(q("a", term="TOOL_EXECUTION_FAILED",
                                               steps=[{"step": "execute", "error": "canceling statement due to statement timeout"}])),
                         "sql_timeout")
        self.assertEqual(learning.root_cause(q("a", term="VALIDATION_FAILED",
                                               steps=[{"step": "validate", "outcome": "rejected", "reason": "Unknown table(s): foo"}])),
                         "hallucinated_schema")
        self.assertEqual(learning.root_cause(q("a", rows=0)), "empty_result")
        self.assertEqual(learning.root_cause(q("a"), {"rating": "down", "reason": "wrong_number"}), "downvoted:wrong_number")

    def test_out_of_scope_refusal_is_not_a_failure(self):
        self.assertIsNone(learning.root_cause(q("a", kind="out_of_scope", refused=True, term="REFUSED")))
        self.assertEqual(learning.root_cause(q("a", refused=True, term="REFUSED")), "refused")

    def test_mask_groups_same_failure(self):
        a = learning.mask('column "month" does not exist\nLINE 7: ORDER BY ...')
        b = learning.mask('column "year" does not exist\nLINE 9: ...')
        self.assertEqual(a, b)
        self.assertNotIn("LINE", a)

    def test_signatures_skip_ok_and_kept(self):
        steps = [{"step": "validate", "outcome": "ok"}, {"step": "validate", "outcome": "entity_mismatch_kept"},
                 {"step": "validate", "outcome": "rejected", "reason": "Unknown table(s): x_1"},
                 {"step": "execute", "error": "boom 42"}]
        sigs = learning.signatures(q("a", steps=steps))
        self.assertEqual(len(sigs), 2)
        self.assertTrue(sigs[0].startswith("validate:rejected"))


class AnalyseTests(unittest.TestCase):
    def setUp(self):
        fail = [{"step": "validate", "outcome": "rejected", "reason": "Unknown table(s): foo_1"}]
        self.qs = [q("t1"), q("t2", term="VALIDATION_FAILED", steps=fail, question="What is VDI?"),
                   q("t3", term="VALIDATION_FAILED", steps=fail, question="what is vdi"),
                   q("t4", term="LLM_UNAVAILABLE", refused=True, question="busy one"),
                   q("t5", question="Installs yesterday", sql="SELECT 2")]
        self.votes = {"t5": {"trace_id": "t5", "rating": "up", "at": "2026-11-11T10:01:00"},
                      "t1": {"trace_id": "t1", "rating": "down", "reason": "wrong_period", "at": "2026-11-11T10:02:00"}}

    def test_unanswered_groups_normalised_text_and_skips_outages(self):
        r = learning.analyse(self.qs, self.votes)
        top = r["unanswered"]
        self.assertEqual(top[0]["count"], 2)                 # 'What is VDI?' == 'what is vdi'
        self.assertNotIn("busy one", [u["question"] for u in top])
        self.assertEqual(r["recurring_failures"][0]["count"], 2)
        self.assertEqual(r["feedback"]["up"], 1)
        self.assertEqual(r["feedback"]["down"], 1)

    def test_regression_cases(self):
        cases = learning.regression_cases(self.qs, self.votes)
        by = {c["trace_id"]: c for c in cases}
        self.assertEqual(by["t5"]["source"], "golden")
        self.assertEqual(by["t5"]["ref_sql"], "SELECT 2")
        self.assertEqual(by["t1"]["source"], "downvote")
        self.assertNotIn("t4", by)                           # outages are not test cases
        self.assertEqual(sum(1 for c in cases if c["q"].lower().startswith("what is vdi")), 1)


class SchemaWatchTests(unittest.TestCase):
    @staticmethod
    def t(name, rows, cols):
        return SimpleNamespace(qualified_name=name, row_count=rows,
                               columns=[SimpleNamespace(column_name=c, sql_type=ty) for c, ty in cols.items()])

    def test_diff(self):
        before = schema_watch.snapshot([self.t("a", 100, {"x": "int", "y": "varchar"}), self.t("b", 50, {"z": "int"}),
                                        self.t("c", 1000, {"k": "int"})])
        after = schema_watch.snapshot([self.t("a", 0, {"x": "bigint", "w": "date"}), self.t("c", 100, {"k": "int"}),
                                       self.t("d", 5, {"q": "int"})])
        d = schema_watch.diff(before, after)
        self.assertEqual(d["tables_added"], ["d"])
        self.assertEqual(d["tables_removed"], ["b"])
        self.assertEqual(d["columns_added"], {"a": ["w"]})
        self.assertEqual(d["columns_removed"], {"a": ["y"]})
        self.assertEqual(d["type_changes"], {"a": {"x": "int -> bigint"}})
        self.assertEqual(d["emptied"], ["a"])
        self.assertEqual(d["shrunk"][0]["table"], "c")

    def test_no_baseline_no_changes(self):
        self.assertEqual(schema_watch.diff({}, schema_watch.snapshot([self.t("a", 1, {"x": "int"})])), {})
        self.assertEqual(schema_watch.snapshot([object()]), {})   # never raises


class AlertTests(unittest.TestCase):
    def test_quality_regression_and_new_failure(self):
        fail = [{"step": "validate", "outcome": "rejected", "reason": "Unknown table(s): brand_new"}]
        base = learning.analyse([q(f"b{i}") for i in range(40)], {})
        cur_qs = [q(f"c{i}") for i in range(20)] + [q(f"f{i}", term="VALIDATION_FAILED", steps=fail) for i in range(10)]
        cur = learning.analyse(cur_qs, {})
        got = {a["kind"] for a in alerts._quality_alerts(cur, base, set())}
        self.assertIn("quality_regression", got)
        self.assertIn("new_failure", got)

    def test_quiet_when_too_few_questions(self):
        self.assertEqual(alerts._quality_alerts(learning.analyse([q("a")], {}), learning.analyse([q("b")], {}), set()), [])

    def test_markdown_renders(self):
        d = {"window_hours": 24, "at": "x", "alerts": [], "most_asked_topics": [], "root_causes": [], "unanswered": [],
             "models": [], "headline": {"questions": 0, "good_answer_rate": None, "cache_hit_rate": None, "p50_ms": None,
                                        "p95_ms": None, "tokens_mean": None, "satisfaction": None, "votes": 0}}
        self.assertIn("## Alerts", alerts.to_markdown(d))


class ExportTests(unittest.TestCase):
    SESSION = {"session_name": "Test <b>", "created_at": "c", "updated_at": "u", "messages": [
        {"role": "user", "content": "Installs | yesterday?"},
        {"role": "assistant", "content": "", "response": {
            "summary": "**1,234** meters", "kpis": [{"label": "Installs", "formatted_value": "1,234"}],
            "tables": [{"title": "By project", "columns": ["Project", "N"], "rows": [{"Project": "Kashi", "N": "1,000"}],
                        "total_rows": 1}],
            "insights": ["<script>x</script>"], "warnings": [], "debug": {"sql": "SELECT 1"}}}]}

    def test_markdown(self):
        md = export.to_markdown(self.SESSION)
        self.assertIn("## 1. Installs | yesterday?", md)
        self.assertIn("| Kashi | 1,000 |", md)
        self.assertNotIn("SELECT 1", md)
        self.assertIn("SELECT 1", export.to_markdown(self.SESSION, include_sql=True))

    def test_html_is_escaped(self):
        h = export.to_html(self.SESSION)
        self.assertNotIn("<script>", h)
        self.assertIn("&lt;b&gt;", h)
        self.assertIn("<strong>1,234</strong>", h)

    def test_endpoint(self):
        import tempfile
        from unittest import mock

        from shared.session_store import SessionStore

        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(os.path.join(tmp, "sessions.json"))   # never the live sessions file
            s = store.create_session("export test")
            with mock.patch.object(main, "sessions", store):
                c = TestClient(main.app)
                r = c.get(f"/api/sessions/{s['session_id']}/export", params={"format": "html"})
                self.assertEqual(r.status_code, 200)
                self.assertIn("attachment", r.headers["content-disposition"])
                self.assertEqual(c.get("/api/sessions/nope/export").status_code, 404)
            store.flush()


if __name__ == "__main__":
    unittest.main()
