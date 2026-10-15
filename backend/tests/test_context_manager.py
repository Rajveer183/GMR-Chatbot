"""Unit tests for the prompt token budget (gmr_agents/context_manager.py).

    python -m unittest discover -s tests -v        (from backend/)
"""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from gmr_agents.context_manager import Section, build, fits_hard_limit  # noqa: E402

WORDS = "lorem ipsum dolor sit amet " * 20  # ~150 tokens per item


def sections():
    return [Section("rules", 0, ["RULE: read-only SELECT only"], required=True),
            Section("schema", 2, ["SCHEMA: t(a, b)"], required=True),
            Section("examples", 4, [f"example {i} {WORDS}" for i in range(3)], min_items=1, header="EXAMPLES:"),
            Section("chart_guide", 6, [f"chart {i} {WORDS}" for i in range(5)], joiner="\n", min_items=1),
            Section("past", 6, [f"past {i} {WORDS}" for i in range(2)]),
            Section("question", 1, ["QUESTION: how many meters?"], required=True)]


class BudgetTests(unittest.TestCase):
    def test_fits_untouched(self):
        built = build(sections(), budget=100_000)
        self.assertFalse(built.trimmed or built.dropped)
        self.assertIn("example 2", built.prompt)

    def test_question_is_last_and_rules_first(self):
        built = build(sections(), budget=100_000)
        self.assertTrue(built.prompt.startswith("RULE:"))
        self.assertTrue(built.prompt.rstrip().endswith("QUESTION: how many meters?"))

    def test_lowest_priority_trimmed_first(self):
        full = build(sections(), budget=100_000).tokens
        built = build(sections(), budget=int(full * 0.55))  # forces cuts whatever the tokenizer
        self.assertIn("past", built.dropped)                 # P6, optional -> gone first
        self.assertIn("chart 0", built.prompt)               # min_items keeps the best chart rule
        self.assertNotIn("chart 4", built.prompt)
        self.assertIn("example 0", built.prompt)             # examples (P4) reduced only if still needed

    def test_required_never_cut(self):
        built = build(sections(), budget=10)                 # impossible budget
        for must in ("RULE:", "SCHEMA:", "QUESTION:"):
            self.assertIn(must, built.prompt)
        self.assertTrue(built.over_budget)

    def test_min_items_respected(self):
        built = build(sections(), budget=10)
        self.assertIn("example 0", built.prompt)
        self.assertNotIn("example 1", built.prompt)

    def test_budget_counts_system_prompt(self):
        small = build(sections(), budget=1400)
        with_system = build(sections(), budget=1400, system="S " * 2000)
        self.assertGreater(len(small.prompt), len(with_system.prompt))

    def test_metrics_and_hard_limit(self):
        built = build(sections(), budget=1000)
        m = built.metrics()
        self.assertTrue(m["truncation"])
        self.assertLessEqual(m["tokens"], 1000)
        self.assertTrue(fits_hard_limit(built, 1000))
        self.assertFalse(fits_hard_limit(build(sections(), budget=100_000), 100))


if __name__ == "__main__":
    unittest.main()
