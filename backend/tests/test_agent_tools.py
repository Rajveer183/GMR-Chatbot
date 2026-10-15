"""Unit tests: calculator tool, database read-only check, cost estimate / cap.

    python -m unittest discover -s tests -v        (from backend/)
"""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from db_connection.readonly import ReadOnlyViolation, assert_read_only  # noqa: E402
from gmr_agents.tools import calculator, calling, registry  # noqa: E402
from shared import config, llm_gateway  # noqa: E402


class CalculatorTests(unittest.TestCase):
    def test_arithmetic(self):
        self.assertEqual(calculator.evaluate("((1200 - 150) / 1200) * 100"), 87.5)
        self.assertEqual(calculator.evaluate("2 ** 10 + 7 // 2 - 5 % 3"), 1024 + 3 - 2)
        self.assertEqual(calculator.evaluate("-(3 + 4) * +2"), -14)

    def test_functions(self):
        self.assertEqual(calculator.evaluate("round(1277180 / 1106912 * 100, 2)"), 115.38)
        self.assertEqual(calculator.evaluate("max(3, 9, 4) + abs(-2) + sqrt(16)"), 15.0)

    def test_thousands_separators(self):
        self.assertAlmostEqual(calculator.evaluate("4,214,678 * 0.12"), 505761.36)

    def test_rejects_code(self):
        for expr in ("__import__('os').system('dir')", "open('x')", "(1).__class__", "x + 1", "lambda: 1",
                     "[i for i in range(9)]", "'a' * 3", "print(1)", "round(x=1)", "True + 1"):
            with self.assertRaises(calculator.CalculatorError, msg=expr):
                calculator.evaluate(expr)

    def test_resource_limits(self):
        for expr in ("9 ** 999999", "pow(10, 1000)", "10 ** 30 * 10 ** 30", "1" + "+1" * 200, "-" * 80 + "1"):
            with self.assertRaises(calculator.CalculatorError, msg=expr[:30]):
                calculator.evaluate(expr)

    def test_errors(self):
        for expr in ("1/0", "", "2 +", "sqrt(-1)"):
            with self.assertRaises(calculator.CalculatorError, msg=expr):
                calculator.evaluate(expr)

    def test_through_plumbing(self):
        ok = calling.run({"tool": "calculator", "args": {"expression": "10 * 2.5"}}, registry.TOOLS).payload
        self.assertEqual(ok, {"success": True, "tool": "calculator", "result": {"expression": "10 * 2.5", "result": 25.0}})
        bad = calling.run({"tool": "calculator", "args": {"expression": 42}}, registry.TOOLS).payload
        self.assertEqual(bad["error_kind"], "invalid_args")
        boom = calling.run({"tool": "calculator", "args": {"expression": "1/0"}}, registry.TOOLS).payload
        self.assertEqual(boom["error_kind"], "invalid_expression")


class ReadOnlyCheckTests(unittest.TestCase):
    WRITES = ["DELETE FROM t", "UPDATE t SET a = 1", "INSERT INTO t VALUES (1)", "DROP TABLE t",
              "TRUNCATE t", "ALTER TABLE t ADD c int", "CREATE TABLE x AS SELECT 1", "GRANT ALL ON t TO public",
              "MERGE INTO t USING s ON 1=1 WHEN MATCHED THEN DELETE", "COPY t TO '/tmp/x'",
              "SELECT 1; DROP TABLE t", "WITH x AS (DELETE FROM t RETURNING *) SELECT * FROM x",
              "/* hi */ DELETE FROM t", "SELECT * FROM t FOR UPDATE", "CALL proc()", "VACUUM t", ""]
    READS = ["SELECT 1", "with a as (select 1) select * from a;", "SHOW TABLES", "DESCRIBE gold_x",
             "SELECT 'DELETE FROM t' AS note", 'SELECT "update", replace(a, \'x\', \'y\') FROM t -- drop table t',
             "SELECT updated_at, deleted_flag FROM t"]

    def test_writes_blocked(self):
        for sql in self.WRITES:
            with self.assertRaises(ReadOnlyViolation, msg=sql):
                assert_read_only(sql)

    def test_reads_allowed(self):
        for sql in self.READS:
            assert_read_only(sql)


class CostTests(unittest.TestCase):
    def setUp(self):
        self._pricing, self._cap = dict(config.LLM_PRICING), config.QUESTION_MAX_COST_USD

    def tearDown(self):
        config.LLM_PRICING.clear(); config.LLM_PRICING.update(self._pricing)
        config.QUESTION_MAX_COST_USD = self._cap

    def test_estimate_model_overrides_provider(self):
        config.LLM_PRICING.update({"groq": (1.0, 2.0), "groq:big": (10.0, 20.0)})
        self.assertEqual(llm_gateway.estimate_cost("groq", "small", 1_000_000, 500_000), 2.0)
        self.assertEqual(llm_gateway.estimate_cost("groq", "big", 100_000, 0), 1.0)
        self.assertEqual(llm_gateway.estimate_cost("unknown", "x", 10**6, 10**6), 0.0)

    def test_cost_cap_blocks_and_is_recorded(self):
        config.LLM_PRICING.update({"nvidia": (100.0, 100.0)})
        config.QUESTION_MAX_COST_USD = 0.01
        with llm_gateway.question_scope("cost-test", 60) as scope:
            with self.assertRaises(llm_gateway._Skip) as ctx:
                llm_gateway._admit("nvidia", "cost-model", "", "x" * 2000, 500)
            self.assertIn("COST_LIMIT_EXCEEDED", str(ctx.exception))
            self.assertIn("COST_LIMIT_EXCEEDED", llm_gateway.limit_hit())
            self.assertEqual(scope.limit_hit, llm_gateway.limit_hit())


if __name__ == "__main__":
    unittest.main()
