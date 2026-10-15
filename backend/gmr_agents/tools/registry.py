"""Tool allowlist.

Who controls it: application code. The controller LLM can only *name* a tool; the
plumbing layer (tools/calling.py) looks the name up in TOOLS and nowhere else, so
a name that is not registered here can never run. Adding a capability = one
ToolSpec here: an argument schema, an optional domain validator, an executor.

    run_sql     args {"sql": str}; validator = the SQL guard (one read-only SELECT on
                catalog tables); executor = tools/sql_executor.py (Postgres / Athena)
    calculator  args {"expression": str}; AST-whitelisted arithmetic (tools/calculator.py)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from gmr_agents.guards import sql_validator
from gmr_agents.tools import calculator, sql_executor
from gmr_agents.tools.calling import ToolSpec


@dataclass
class SqlToolContext:
    """App-side inputs a SQL call needs that the LLM must not supply: which
    catalog (database) to check against, and -- once validated -- the guard's
    verdict and the executor plan built from it."""
    catalog: Any
    verdict: dict | None = None
    plan: dict | None = None


class RunSqlArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    sql: str = Field(min_length=1, description="One read-only SELECT statement.")


class ToolExecutionError(RuntimeError):
    """Raised by an executor; `error_kind` lets the agent loop route on it."""

    def __init__(self, message: str, error_kind: str = "execution"):
        super().__init__(message)
        self.error_kind = error_kind


def _validate_sql(args: RunSqlArgs, ctx: SqlToolContext) -> str | None:
    verdict = sql_validator.validate_sql(args.sql, ctx.catalog)
    if not verdict["ok"]:
        return verdict["error"]["message"]
    ctx.verdict = verdict
    return None


def _execute_sql(args: RunSqlArgs, ctx: SqlToolContext) -> dict:
    if ctx.plan is None:
        raise ToolExecutionError("run_sql was called without a validated plan", "execution")
    outcome = sql_executor.run_sql(ctx.plan, ctx.catalog)
    if not outcome["ok"]:
        raise ToolExecutionError(outcome["error"]["message"], outcome["error"]["kind"])
    return outcome["result"]


def _summarize_sql(raw: dict) -> dict:
    """What the LLM gets back: the rows (first ones), not the executor internals."""
    return {"data": raw.get("result"), "total_groups": raw.get("group_count")}


class CalculatorArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    expression: str = Field(min_length=1, max_length=calculator.MAX_LEN,
                            description="Arithmetic only: numbers, + - * / ** ( ), round/abs/min/max/sqrt/log.")


def _execute_calculator(args: CalculatorArgs, ctx) -> dict:
    try:
        return {"expression": args.expression, "result": calculator.evaluate(args.expression)}
    except calculator.CalculatorError as exc:
        raise ToolExecutionError(f"calculator: {exc}", "invalid_expression") from None


TOOLS: dict[str, ToolSpec] = {
    "run_sql": ToolSpec(
        name="run_sql",
        description="Run ONE read-only SELECT against the WFM / MDM gold tables and return the rows.",
        args_model=RunSqlArgs,
        validate=_validate_sql,
        execute=_execute_sql,
        summarize=_summarize_sql,
    ),
    "calculator": ToolSpec(
        name="calculator",
        description="Exact arithmetic on numbers the user gave (no data needed).",
        args_model=CalculatorArgs,
        execute=_execute_calculator,
    ),
}


def get(name: str) -> ToolSpec | None:
    return TOOLS.get(name)
