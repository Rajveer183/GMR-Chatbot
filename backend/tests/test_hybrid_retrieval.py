"""Unit tests for hybrid retrieval pieces (gmr_agents/schema/retrieval.py):
RRF fusion, per-item ranking / de-duplication, BM25 exact matching, tokenizer.

    python -m unittest discover -s tests -v        (from backend/)

Pure functions only -- no embedding model, no index files touched.
"""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np  # noqa: E402

from gmr_agents.schema.retrieval import BM25, item_ranks, rrf_fuse, tokens  # noqa: E402


class RrfTests(unittest.TestCase):
    def test_score_formula(self):
        fused = rrf_fuse({"bm25": {"a": 0}, "dense": {"a": 2}}, {"bm25": 1.0, "dense": 1.0}, 60)
        self.assertAlmostEqual(fused["a"], 1 / 61 + 1 / 63)

    def test_both_rankers_beat_one(self):
        fused = rrf_fuse({"bm25": {"both": 1, "only_bm25": 0}, "dense": {"both": 1}}, {}, 60)
        self.assertGreater(fused["both"], fused["only_bm25"])

    def test_item_in_one_ranking_still_returned(self):
        fused = rrf_fuse({"bm25": {"x": 0}, "dense": {"y": 0}}, {}, 60)
        self.assertEqual(set(fused), {"x", "y"})

    def test_empty_rankings(self):
        self.assertEqual(rrf_fuse({}, {}, 60), {})
        self.assertEqual(rrf_fuse({"bm25": {}, "dense": {"y": 0}}, {}, 60), {"y": 1 / 61})

    def test_weights(self):
        fused = rrf_fuse({"bm25": {"a": 0}, "dense": {"b": 0}}, {"bm25": 2.0, "dense": 1.0}, 60)
        self.assertGreater(fused["a"], fused["b"])


class ItemRankTests(unittest.TestCase):
    def test_duplicate_texts_count_once_by_best_text(self):
        owners = ["e1", "e1", "e2", "e3"]
        scores = np.array([0.9, 0.5, 0.7, 0.0])
        self.assertEqual(item_ranks(np.argsort(-scores), scores, owners), {"e1": 0, "e2": 1})

    def test_zero_scores_not_retrieved(self):
        scores = np.zeros(3)
        self.assertEqual(item_ranks(np.argsort(-scores), scores, ["a", "b", "c"]), {})


class Bm25Tests(unittest.TestCase):
    DOCS = ["Split SAT 10 by DT, Consumer and Feeder",
            "consumer_detail_autoindexing has one row per consumer",
            "meters pending SAT for more than 120+ days",
            "installation trend over the last 12 months"]

    def _top(self, query):
        bm = BM25([tokens(d) for d in self.DOCS], 1.5, 0.75)
        s = bm.scores(tokens(query))
        return int(np.argmax(s)) if s.max() > 0 else None

    def test_exact_keyword(self):
        self.assertEqual(self._top("sat 10 feeder"), 0)

    def test_identifier_with_underscores(self):
        self.assertEqual(self._top("consumer_detail_autoindexing"), 1)

    def test_numbers_and_plus_kept(self):
        self.assertEqual(self._top("120+ days"), 2)

    def test_no_match(self):
        self.assertIsNone(self._top("photosynthesis"))


class TokenizerTests(unittest.TestCase):
    def test_keeps_numbers_symbols_and_domain_terms(self):
        t = tokens("SAT-10 PMPM 120+ days 7.05% AT&C")
        for want in ("sat", "10", "pmpm", "120+", "05%", "at&c"):
            self.assertIn(want, t)

    def test_stop_words_removed(self):
        self.assertNotIn("the", tokens("what is the SAT rate"))

    def test_short_or_identifier_only_query(self):
        self.assertEqual(tokens("SAT"), ["sat"])
        self.assertEqual(tokens(""), [])


if __name__ == "__main__":
    unittest.main()
