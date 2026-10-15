"""Unit tests for the grounding guard (gmr_agents/validation/grounding.py).

    python -m unittest discover -s tests -v        (from backend/)

The LLM's structured answer is supplied directly; no model or database is used.
"""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from gmr_agents.validation import grounding  # noqa: E402
from gmr_agents.validation.grounding import Evidence  # noqa: E402

REVIEW = Evidence("term:review", "business_rules.yaml#glossary", "review",
                  "review: Auto-indexing: evidence exists but is inconclusive; needs manual check.", 0.85)
TOPIC = Evidence("rule:mdm_autoindexing", "business_rules.yaml#topics", "MDM feeder auto-indexing",
                 "Pending verification = review + no data (target 0). Outage match threshold 90%.", 0.70)
CHUNKS = [REVIEW, TOPIC]


def answer(**kw):
    base = {"summary": "Review means the evidence is inconclusive and needs a manual field check.",
            "insights": [], "citations": ["term:review"], "supported": True, "abstain": False, "conflicts": []}
    return {**base, **kw}


class SelectTests(unittest.TestCase):
    def test_empty_retrieval(self):
        self.assertEqual(grounding.select([], k=5, min_relevance=0.6), [])

    def test_low_similarity_dropped(self):
        low = Evidence("term:x", "s", "x", "something unrelated", 0.41)
        self.assertEqual(grounding.select([low], k=5, min_relevance=0.62), [])

    def test_duplicates_removed_best_kept(self):
        dup = Evidence("term:review2", "business_rules.yaml#glossary", "review",
                       "review:  Auto-indexing: evidence exists but is inconclusive;  needs manual check.", 0.80)
        got = grounding.select([dup, REVIEW], k=5, min_relevance=0.6)
        self.assertEqual([c.chunk_id for c in got], ["term:review"])

    def test_top_k_cap_and_order(self):
        got = grounding.select([TOPIC, REVIEW], k=1, min_relevance=0.6)
        self.assertEqual([c.chunk_id for c in got], ["term:review"])

    def test_context_block_marks_chunks(self):
        text = grounding.context_block(CHUNKS, 10_000)
        self.assertIn("[chunk_id=term:review]", text)
        self.assertIn("[chunk_id=rule:mdm_autoindexing]", text)


class VerifyTests(unittest.TestCase):
    def test_supported_answer_passes(self):
        r = grounding.verify(answer(), CHUNKS)
        self.assertFalse(r.abstained)
        self.assertEqual([c["id"] for c in r.citations], ["term:review"])
        self.assertGreater(r.confidence, 0.5)

    def test_fabricated_citation_dropped(self):
        r = grounding.verify(answer(citations=["term:review", "doc:made_up", "https://example.com"]), CHUNKS)
        self.assertEqual([c["id"] for c in r.citations], ["term:review"])
        self.assertTrue(any("not provided" in p for p in r.problems))

    def test_citation_format_variants_accepted(self):
        r = grounding.verify(answer(citations=["chunk_id=term:review", " [rule:mdm_autoindexing] "]), CHUNKS)
        self.assertFalse(r.abstained)
        self.assertEqual([c["id"] for c in r.citations], ["term:review", "rule:mdm_autoindexing"])

    def test_only_fabricated_citations_abstain(self):
        r = grounding.verify(answer(citations=["rule:does_not_exist"]), CHUNKS)
        self.assertTrue(r.abstained)
        self.assertEqual(r.summary, grounding.ABSTAIN_NO_EVIDENCE)

    def test_no_citation_abstains(self):
        self.assertTrue(grounding.verify(answer(citations=[]), CHUNKS).abstained)

    def test_inline_citation_markers_count_and_are_removed(self):
        r = grounding.verify(answer(citations=[], summary="Review needs a manual check. [term:review]"), CHUNKS)
        self.assertFalse(r.abstained)
        self.assertNotIn("[term:review]", r.summary)

    def test_model_abstain_is_honoured(self):
        r = grounding.verify(answer(abstain=True, summary="The context does not give the July 2026 value."), CHUNKS)
        self.assertTrue(r.abstained)
        self.assertIn("July 2026", r.summary)
        self.assertEqual(r.citations, [])

    def test_supported_false_abstains(self):
        self.assertTrue(grounding.verify(answer(supported=False), CHUNKS).abstained)

    def test_supported_number_kept(self):
        r = grounding.verify(answer(summary="An outage match needs 90% overlap.", citations=["rule:mdm_autoindexing"]), CHUNKS)
        self.assertFalse(r.abstained)

    def test_unsupported_number_abstains(self):
        r = grounding.verify(answer(summary="An outage match needs 95% overlap.", citations=["rule:mdm_autoindexing"]),
                             CHUNKS)
        self.assertTrue(r.abstained)
        self.assertTrue(any("numbers not in the evidence" in p for p in r.problems))

    def test_number_from_uncited_chunk_not_accepted(self):
        # 90% exists only in the topic chunk; citing just the glossary chunk must not support it
        r = grounding.verify(answer(summary="Review applies above 90% overlap.", citations=["term:review"]), CHUNKS)
        self.assertTrue(r.abstained)

    def test_unsupported_date_in_insight_dropped(self):
        r = grounding.verify(answer(insights=["The rule changed on 2025-04-01 to 1,250 meters."]), CHUNKS)
        self.assertFalse(r.abstained)
        self.assertEqual(r.insights, [])

    def test_conflicts_reported_and_lower_confidence(self):
        plain = grounding.verify(answer(), CHUNKS).confidence
        r = grounding.verify(answer(conflicts=["one chunk says 90%, another says 95%"]), CHUNKS)
        self.assertEqual(len(r.conflicts), 1)
        self.assertLess(r.confidence, plain)

    def test_empty_summary_abstains(self):
        self.assertTrue(grounding.verify(answer(summary=""), CHUNKS).abstained)

    def test_confidence_is_not_similarity(self):
        r = grounding.verify(answer(), CHUNKS)
        self.assertNotEqual(r.confidence, REVIEW.similarity)


class DataCitationTests(unittest.TestCase):
    def test_sources_built_from_what_ran(self):
        cites = grounding.data_citations({"tables": ["gold_mi_installation_month_agg"]},
                                         {"examples": ["e0"], "topics": ["installation"]}, "2026-10-03 06:00")
        self.assertEqual(cites[0]["id"], "sql")
        self.assertIn("gold_mi_installation_month_agg", cites[0]["location"])
        self.assertIn("2026-10-03", cites[0]["location"])
        self.assertTrue(any(c["id"] == "example:0" for c in cites))

    def test_unknown_example_ignored(self):
        cites = grounding.data_citations({"tables": []}, {"examples": ["e9999", "zzz"], "topics": ["nope"]}, None)
        self.assertEqual([c["id"] for c in cites], ["sql"])


if __name__ == "__main__":
    unittest.main()
