"""Unit tests for data-aware chunking (gmr_agents/schema/chunking.py).

    python -m unittest discover -s tests -v        (from backend/)
"""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from gmr_agents.schema import chunking  # noqa: E402

LONG_STEP = "a detailed computation step " * 30          # ~230 tokens each


def kpi(**extra):
    base = {"number": 7, "title": "Demo KPI", "kpi_code": "demo", "section": "MI", "card": "Demo card",
            "aliases": ["demo kpi", "kpi 7"], "summary": "Counts demo things."}
    return {**base, **extra}


def chunks(k, lo=60, hi=450):
    return chunking.chunk_kpi(k, min_tokens=lo, max_tokens=hi)


class KpiChunkTests(unittest.TestCase):
    def test_parent_overview_first_with_aliases(self):
        cs = chunks(kpi())
        self.assertEqual(cs[0].chunk_id, "kpi:demo")
        self.assertEqual(cs[0].kind, "kpi_overview")
        self.assertIn("demo kpi", cs[0].text)

    def test_long_list_split_between_items_never_inside(self):
        steps = [{"step": f"S{i}", "detail": LONG_STEP} for i in range(5)]
        cs = [c for c in chunks(kpi(logic=steps)) if c.section.startswith("logic")]
        self.assertGreater(len(cs), 1)
        joined = "\n".join(c.text for c in cs)
        for i in range(5):
            self.assertEqual(joined.count(f"- S{i}:"), 1)          # every step present exactly once
            self.assertTrue(any(f"- S{i}: {LONG_STEP.strip()}" in c.text for c in cs))  # and whole

    def test_oversized_single_item_kept_whole(self):
        huge = "x " * 2000
        cs = [c for c in chunks(kpi(caveats=[huge])) if c.section.startswith("caveats")]
        self.assertEqual(len(cs), 1)
        self.assertIn(huge.strip(), cs[0].text)

    def test_sql_formulas_example_atomic(self):
        sql = "SELECT a,\n  b\nFROM t\nWHERE c = 1 " * 60                  # well over max_tokens
        cs = chunks(kpi(sql_recipe=sql, formulas=[{"output": "x", "formula": "a/b"}],
                        example=[{"step": "rows", "result": "1,2,3"}]))
        sql_chunks = [c for c in cs if "sql_recipe" in c.section]
        self.assertEqual(len(sql_chunks), 1)
        self.assertIn(sql.strip(), sql_chunks[0].text)
        self.assertTrue(any(c.section == "formulas" for c in cs))
        self.assertTrue(any(c.section == "example" for c in cs))

    def test_small_sections_merged(self):
        cs = chunks(kpi(filters=[{"filter": "date", "values": "from/to"}],
                        response=[{"part": "total", "shape": "number"}]))
        merged = [c for c in cs if "+" in c.section]
        self.assertEqual(len(merged), 1)
        self.assertIn("filters", merged[0].section)
        self.assertIn("response", merged[0].section)

    def test_children_linked_to_parent_and_neighbours(self):
        cs = chunks(kpi(logic=[{"step": "A", "detail": LONG_STEP}], caveats=[LONG_STEP], sql_recipe="SELECT 1"))
        kids = cs[1:]
        self.assertTrue(all(c.parent == "kpi:demo" for c in kids))
        self.assertIsNone(kids[0].prev)
        self.assertEqual(kids[0].next, kids[1].chunk_id)
        self.assertEqual(kids[1].prev, kids[0].chunk_id)
        self.assertIsNone(kids[-1].next)

    def test_every_chunk_names_its_kpi(self):
        for c in chunks(kpi(logic=[{"step": "A", "detail": "b"}], caveats=["c"]))[1:]:
            self.assertTrue(c.text.startswith("KPI 7 Demo KPI"))


class DocumentTests(unittest.TestCase):
    def test_duplicate_codes_get_unique_ids(self):
        data = {"kpis": [kpi(number=8, kpi_code="dup"), kpi(number=12, kpi_code="dup")]}
        ids = [c.chunk_id for c in chunking.chunk_document(data, min_tokens=60, max_tokens=450)]
        self.assertIn("kpi:dup-8", ids)
        self.assertIn("kpi:dup-12", ids)
        self.assertEqual(len(ids), len(set(ids)))

    def test_mixed_document(self):
        data = {"kpis": [kpi()], "gold_tables": {"gold_x": {"name": "gold_x", "built_by": "job()"}},
                "pipeline_steps": [{"step": "1. API", "detail": "pull"}]}
        kinds = {c.kind for c in chunking.chunk_document(data, min_tokens=60, max_tokens=450)}
        self.assertEqual(kinds, {"kpi_overview", "gold_table", "pipeline"})

    def test_empty_document(self):
        self.assertEqual(chunking.chunk_document({}, min_tokens=60, max_tokens=450), [])

    def test_expand_adds_parent_once_first(self):
        cs = chunks(kpi(logic=[{"step": "A", "detail": LONG_STEP}], caveats=[LONG_STEP]))
        by_id = {c.chunk_id: c for c in cs}
        out = chunking.expand([cs[2].chunk_id, cs[1].chunk_id, "missing"], by_id)
        self.assertEqual(out, ["kpi:demo", cs[2].chunk_id, cs[1].chunk_id])


if __name__ == "__main__":
    unittest.main()
