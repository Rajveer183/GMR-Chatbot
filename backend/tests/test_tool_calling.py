"""Unit tests for the tool-calling plumbing (gmr_agents/tools/calling.py) and the
gateway's tolerant JSON parsing (shared/llm_gateway.extract_json).

    python -m unittest discover -s tests -v        (from backend/)

No database or LLM is touched: tools here are small fakes registered in a local
allowlist, and run_sql's real schema is exercised without its executor.
"""

from __future__ import annotations

import datetime as dt
import decimal
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pydantic import BaseModel, ConfigDict  # noqa: E402

from gmr_agents.tools import calling, registry  # noqa: E402
from gmr_agents.tools.calling import ToolSpec  # noqa: E402
from shared.llm_gateway import extract_json  # noqa: E402


class AddArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    a: int
    b: int = 0


class NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Weird:
    def __repr__(self) -> str:
        return "<Weird>"


def _boom(args, ctx):
    raise RuntimeError("disk on fire")


REGISTRY = {
    "add": ToolSpec("add", "a + b", AddArgs, execute=lambda args, ctx: {"sum": args.a + args.b}),
    "ping": ToolSpec("ping", "no args", NoArgs, execute=lambda args, ctx: "pong"),
    "explode": ToolSpec("explode", "raises", NoArgs, execute=_boom),
    "nothing": ToolSpec("nothing", "returns None", NoArgs, execute=lambda args, ctx: None),
    "weird": ToolSpec("weird", "non-JSON output", NoArgs, execute=lambda args, ctx: {
        "when": dt.date(2026, 10, 4), "amount": decimal.Decimal("1.50"), "nan": float("nan"),
        "np": np.int64(7), "arr": np.array([1, 2]), "obj": Weird(), "tags": {"x"},
        "df": pd.DataFrame({"zone": ["agra", "kashi"], "n": [3, 4]})}),
    "picky": ToolSpec("picky", "domain validator", NoArgs, execute=lambda args, ctx: "ran",
                      validate=lambda args, ctx: "not today"),
}


def run(raw):
    return calling.run(raw, REGISTRY).payload


class ParseTests(unittest.TestCase):
    def test_valid_json_text(self):
        self.assertEqual(run('{"tool": "add", "args": {"a": 2, "b": 3}}'),
                         {"success": True, "tool": "add", "result": {"sum": 5}})

    def test_already_decoded_object(self):
        self.assertTrue(run({"tool": "add", "args": {"a": 1}})["success"])

    def test_fenced_and_prose(self):
        self.assertTrue(run('Sure!\n```json\n{"tool": "ping", "args": {}}\n```')["success"])

    def test_malformed_json(self):
        out = run('{"tool": "add", "args": {"a": 2,,}')
        self.assertFalse(out["success"])
        self.assertEqual(out["error_kind"], "parse")

    def test_truncated_json(self):
        self.assertEqual(run('{"tool": "add", "args": {"a": 2')["error_kind"], "parse")

    def test_empty_response(self):
        for raw in (None, "", "   "):
            self.assertEqual(run(raw)["error_kind"], "parse")

    def test_not_an_object(self):
        self.assertEqual(run('["add", 1, 2]')["error_kind"], "parse")


class ToolNameTests(unittest.TestCase):
    def test_missing_tool(self):
        out = run({"args": {"a": 1}})
        self.assertEqual((out["error_kind"], out["tool"]), ("missing_tool", None))

    def test_null_or_non_string_tool(self):
        self.assertEqual(run({"tool": None, "args": {}})["error_kind"], "missing_tool")
        self.assertEqual(run({"tool": 42, "args": {}})["error_kind"], "missing_tool")

    def test_unknown_tool(self):
        out = run({"tool": "os.system", "args": {"cmd": "rm -rf /"}})
        self.assertEqual(out["error_kind"], "unknown_tool")
        self.assertIn("Available tools", out["error"])

    def test_never_resolves_python_names(self):
        # names of real Python callables are just unknown strings to the registry
        for name in ("eval", "exec", "__import__", "open", "_boom"):
            self.assertEqual(run({"tool": name, "args": {}})["error_kind"], "unknown_tool")


class ArgsTests(unittest.TestCase):
    def test_missing_args_ok_when_none_required(self):
        self.assertEqual(run({"tool": "ping"})["result"], "pong")

    def test_null_args_ok_when_none_required(self):
        self.assertEqual(run({"tool": "ping", "args": None})["result"], "pong")

    def test_missing_args_reports_required(self):
        out = run({"tool": "add"})
        self.assertEqual(out["error_kind"], "invalid_args")
        self.assertIn("missing required argument 'a'", out["error"])

    def test_args_must_be_object(self):
        for bad in ("a=1", [1, 2], 5):
            out = run({"tool": "add", "args": bad})
            self.assertEqual(out["error_kind"], "invalid_args", bad)
            self.assertIn("must be a JSON object", out["error"])

    def test_wrong_type(self):
        out = run({"tool": "add", "args": {"a": "two"}})
        self.assertEqual(out["error_kind"], "invalid_args")
        self.assertIn("argument 'a'", out["error"])

    def test_strict_no_coercion(self):
        self.assertEqual(run({"tool": "add", "args": {"a": "2"}})["error_kind"], "invalid_args")

    def test_unexpected_argument(self):
        out = run({"tool": "add", "args": {"a": 1, "c": 9}})
        self.assertIn("unexpected argument 'c'", out["error"])

    def test_domain_validator_rejects(self):
        out = run({"tool": "picky", "args": {}})
        self.assertEqual((out["error_kind"], out["error"]), ("rejected", "not today"))


class ExecutionTests(unittest.TestCase):
    def test_tool_exception_is_structured(self):
        out = run({"tool": "explode", "args": {}})
        self.assertEqual(out, {"success": False, "tool": "explode", "error": "disk on fire", "error_kind": "execution"})

    def test_tagged_exception_kind(self):
        def fail(args, ctx):
            raise registry.ToolExecutionError("database unreachable", "connectivity")
        reg = {"t": ToolSpec("t", "", NoArgs, execute=fail)}
        self.assertEqual(calling.run({"tool": "t"}, reg).payload["error_kind"], "connectivity")

    def test_tool_returning_none(self):
        self.assertEqual(run({"tool": "nothing"}), {"success": True, "tool": "nothing", "result": {}})

    def test_non_serializable_output(self):
        outcome = calling.run({"tool": "weird"}, REGISTRY)
        text = json.dumps(outcome.payload)  # must not raise
        res = outcome.payload["result"]
        self.assertEqual(res["when"], "2026-10-04")
        self.assertEqual(res["amount"], 1.5)
        self.assertIsNone(res["nan"])
        self.assertEqual((res["np"], res["arr"]), (7, [1, 2]))
        self.assertEqual(res["obj"], "<Weird>")
        self.assertEqual(res["df"]["row_count"], 2)
        self.assertIn("kashi", text)
        self.assertIsInstance(outcome.raw["df"], pd.DataFrame)  # the app still gets the raw value

    def test_every_payload_is_serializable(self):
        for raw in ('{"tool": "weird"}', "", '{"tool": "add", "args": {"a": "x"}}', {"tool": "explode"}):
            json.dumps(calling.run(raw, REGISTRY).payload)


class RunSqlSchemaTests(unittest.TestCase):
    """The real run_sql tool's argument schema (no database needed)."""

    def _prepare(self, args):
        return calling.prepare({"tool": "run_sql", "args": args},
                               {"run_sql": registry.TOOLS["run_sql"].__class__(
                                   **{**registry.TOOLS["run_sql"].__dict__, "validate": None})})

    def test_sql_required(self):
        self.assertEqual(self._prepare({}).payload["error_kind"], "invalid_args")

    def test_sql_must_be_string(self):
        self.assertEqual(self._prepare({"sql": 123}).payload["error_kind"], "invalid_args")

    def test_empty_sql(self):
        self.assertEqual(self._prepare({"sql": ""}).payload["error_kind"], "invalid_args")

    def test_extra_arg(self):
        self.assertIn("unexpected argument 'db'", self._prepare({"sql": "SELECT 1", "db": "x"}).payload["error"])

    def test_valid(self):
        self.assertIsInstance(self._prepare({"sql": "SELECT 1"}), calling.ToolCall)


class GatewayParseTests(unittest.TestCase):
    def test_trailing_comma(self):
        self.assertEqual(extract_json('{"summary": "ok", "insights": [1, 2,],}'), {"summary": "ok", "insights": [1, 2]})

    def test_text_after_object(self):
        self.assertEqual(extract_json('{"summary": "x"} Hope this helps! {not json}'), {"summary": "x"})

    def test_braces_inside_strings(self):
        self.assertEqual(extract_json('{"sql": "SELECT \'}\' AS b", "x": 1}'), {"sql": "SELECT '}' AS b", "x": 1})

    def test_truncated_rescued_when_required_complete(self):
        text = '{"kind": "data", "question": "How many meters?", "parts": [], "examples": ["e1", "e2'
        self.assertIsNone(extract_json(text))  # without `required`: no guessing
        self.assertEqual(extract_json(text, ("kind", "question")),
                         {"kind": "data", "question": "How many meters?", "parts": []})

    def test_truncated_sql_never_rescued(self):
        text = '{"reasoning": "count", "sql": "SELECT COUNT(*) FROM public.gold_mi_installation_day_agg WHERE proj'
        self.assertIsNone(extract_json(text, ("sql",)))

    def test_truncated_after_sql_is_rescued(self):
        text = '{"sql": "SELECT 1", "answer_kind": "number", "display": {"cards": ["n"'
        self.assertEqual(extract_json(text, ("sql",)), {"sql": "SELECT 1", "answer_kind": "number"})


if __name__ == "__main__":
    unittest.main()
