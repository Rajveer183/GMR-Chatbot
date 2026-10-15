"""Unit tests for token counting and budgets (shared/tokens.py, llm_pipeline previews /
history, context_manager breakdown, the per-call context cap in llm_gateway).

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

from gmr_agents import context_manager, llm_pipeline as P  # noqa: E402
from shared import config, llm_gateway, tokens  # noqa: E402


class TokenizerTests(unittest.TestCase):
    def test_gmr_identifiers_are_several_tokens(self):
        # Budgets must never assume one identifier = one token.
        for text in ("consumer_detail_autoindexing", "120+ days", "91-120", "REASSIGN OUT", "gold_mi_sat_invoice_snapshot"):
            self.assertGreater(tokens.count(text), 1, text)

    def test_provider_factor_learns_and_is_clamped(self):
        before = tokens.factor("test-provider")
        for _ in range(50):
            tokens.learn("test-provider", 1000, 1200)
        self.assertAlmostEqual(tokens.factor("test-provider"), 1.2, places=2)
        tokens.learn("test-provider", 1000, 10_000)          # absurd report is clamped at 1.5x
        self.assertLessEqual(tokens.factor("test-provider"), 1.5)
        self.assertNotEqual(before, tokens.factor("test-provider"))

    def test_estimate_grows_with_text(self):
        self.assertGreater(llm_gateway.estimate_tokens("SELECT 1 " * 100), llm_gateway.estimate_tokens("SELECT 1"))


class ResultPreviewTests(unittest.TestCase):
    def test_whole_rows_within_budget(self):
        df = pd.DataFrame({"division": [f"EDD DIVISION NUMBER {i}" for i in range(25)], "installed": range(25)})
        text, n = P._result_preview(df, max_tokens=60)
        lines = text.strip().split("\n")
        self.assertEqual(n, 25)
        self.assertEqual(lines[0], "division,installed")
        self.assertLess(len(lines), 26)
        for line in lines[1:]:
            self.assertRegex(line, r"^EDD DIVISION NUMBER \d+,\d+$")   # never a half row

    def test_empty(self):
        self.assertEqual(P._result_preview(pd.DataFrame()), ("(no rows)", 0))


class HistoryBudgetTests(unittest.TestCase):
    def _turns(self, n, answer_len=600):
        return [{"question": f"How many meters pending SAT in Kashi for SAT-{i}?", "summary": "x" * answer_len,
                 "context": {"sql": "SELECT 1 FROM public.gold_mi_sat_invoice_snapshot"}} for i in range(n)]

    def test_long_history_fits_budget_and_keeps_focus(self):
        text = P._history_block(self._turns(8))
        self.assertLessEqual(llm_gateway.estimate_tokens(text), config.MAX_HISTORY_TOKENS + 50)
        self.assertIn("Conversation focus", text)
        self.assertIn("kashi", text.lower())

    def test_short_history_untouched(self):
        turns = self._turns(2, answer_len=50)
        self.assertIn("x" * 50, P._history_block(turns))

    def test_empty_history(self):
        self.assertEqual(P._history_block([]), "(none)")


class BreakdownAndCapTests(unittest.TestCase):
    def test_context_components_reported(self):
        S = context_manager.Section
        built = context_manager.build([S("rules", 0, ["RULE"], required=True), S("question", 1, ["QUESTION: x"], required=True)],
                                      budget=10_000)
        self.assertEqual(set(built.metrics()["components"]), {"rules", "question"})

    def test_hard_context_cap(self):
        huge = "word " * (config.MAX_CONTEXT_TOKENS + 100)
        with self.assertRaises(llm_gateway._Skip) as ctx:
            llm_gateway._admit("nvidia", "cap-test-model", "", huge, 100)
        self.assertIn("CONTEXT_LIMIT_EXCEEDED", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
