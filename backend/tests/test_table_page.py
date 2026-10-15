"""Tests for server-side table pagination (POST /api/table-page in gmr_agents/main.py).

    python -m unittest discover -s tests -v        (from backend/)

The SQL-building helpers and request validation need no database.
"""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from fastapi.testclient import TestClient  # noqa: E402

from gmr_agents import main  # noqa: E402


class PagedSqlTests(unittest.TestCase):
    def test_limit_replaced_order_kept(self):
        sql = "SELECT division, SUM(n) AS n FROM public.t GROUP BY division ORDER BY n DESC LIMIT 5"
        paged = main._paged_sql(sql, "postgres", 25, 25)
        self.assertIn("ORDER BY n DESC", paged)
        self.assertTrue(paged.endswith("OFFSET 25 LIMIT 25"))
        self.assertNotIn("LIMIT 5", paged)

    def test_without_limit_for_count(self):
        self.assertNotIn("LIMIT", main._without_limit("SELECT a FROM public.t LIMIT 200", "postgres").upper())

    def test_athena_dialect(self):
        self.assertTrue(main._paged_sql("SELECT a FROM gmr_wfm_silver.t ORDER BY a LIMIT 10", "athena", 0, 50)
                        .endswith("OFFSET 0 LIMIT 50"))


class EndpointValidationTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)

    def test_bad_page_size(self):
        r = self.client.post("/api/table-page", json={"trace_id": "x", "page": 1, "page_size": 7})
        self.assertEqual(r.status_code, 422)

    def test_bad_page(self):
        r = self.client.post("/api/table-page", json={"trace_id": "x", "page": 0, "page_size": 25})
        self.assertEqual(r.status_code, 422)

    def test_unknown_answer(self):
        r = self.client.post("/api/table-page", json={"trace_id": "no-such-trace", "page": 1, "page_size": 25})
        self.assertEqual(r.status_code, 404)


if __name__ == "__main__":
    unittest.main()
