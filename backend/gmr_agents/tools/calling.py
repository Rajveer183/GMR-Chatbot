"""Tool-calling plumbing: LLM tool-call JSON -> a safe, normalized tool result.

    LLM -> tool-call JSON -> parse -> resolve tool (allowlist) -> validate args
        -> tool's own validator -> execute -> normalize result -> back to the LLM

Who controls it: application code. This layer only parses, validates, dispatches,
catches errors and normalizes; what a tool does (and its domain checks, e.g. the
SQL guard for run_sql) lives with the tool in tools/registry.py.

Guarantees
  * Never raises for bad LLM output: every failure is a structured result
        {"success": false, "tool": <name or null>, "error": "...", "error_kind": "..."}
    error_kind: parse | missing_tool | unknown_tool | invalid_args | rejected | execution
  * A tool runs only if its name is a key of the explicit registry passed in --
    never getattr / globals / import by an LLM-provided string.
  * Arguments are validated by the tool's Pydantic model (extra keys forbidden,
    types checked, required fields enforced) before anything executes.
  * Success results are JSON-serializable ({"success": true, "tool", "result"});
    the raw Python value (e.g. a DataFrame) is kept separately for the app.
  * Logs name the tool, the argument KEYS and sizes, and the outcome -- never
    argument values (they may hold SQL, identifiers or credentials).
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import decimal
import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from pydantic import BaseModel, ValidationError

from shared.llm_gateway import extract_json

log = logging.getLogger(__name__)

_RESULT_ROWS = 50  # rows of a table result passed back to the LLM (the app keeps them all)


@dataclass(frozen=True)
class ToolSpec:
    """One allowlisted tool.

    args_model  Pydantic model of the LLM-supplied arguments (use extra="forbid")
    execute     fn(args: args_model, context) -> raw result
    validate    optional fn(args, context) -> None | str: a domain check that runs
                after schema validation; a returned string rejects the call
    summarize   optional fn(raw) -> JSON-able view for the LLM (default: generic)
    """
    name: str
    description: str
    args_model: type[BaseModel]
    execute: Callable[[BaseModel, Any], Any]
    validate: Callable[[BaseModel, Any], str | None] | None = None
    summarize: Callable[[Any], Any] | None = None
    idempotent: bool = True   # same validated args may reuse an earlier result (guards/idempotency.py)


@dataclass
class ToolCall:
    tool: str
    args: BaseModel


@dataclass
class ToolOutcome:
    """payload = what goes back to the LLM (always JSON-serializable);
    raw = the tool's Python return value for the application (None on failure)."""
    payload: dict[str, Any]
    raw: Any = None
    call: ToolCall | None = None

    @property
    def success(self) -> bool:
        return bool(self.payload.get("success"))


def _failure(tool: str | None, kind: str, error: str) -> ToolOutcome:
    return ToolOutcome({"success": False, "tool": tool, "error": error, "error_kind": kind})


# --------------------------------------------------------------------------- #
# parse
# --------------------------------------------------------------------------- #

def parse_tool_call(raw: str | Mapping | None) -> dict[str, Any] | ToolOutcome:
    """LLM output (text or an already-decoded object) -> {"tool": str|None, "args": Any}.
    Malformed / empty / non-object output -> a `parse` failure (never an exception)."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return _failure(None, "parse", "The tool call was empty.")
    obj = extract_json(raw) if isinstance(raw, str) else raw
    if not isinstance(obj, Mapping):
        return _failure(None, "parse", "The tool call is not a JSON object (malformed or truncated JSON).")
    tool = obj.get("tool")
    return {"tool": tool if isinstance(tool, str) else None, "args": obj.get("args", None),
            "has_args": "args" in obj}


# --------------------------------------------------------------------------- #
# resolve + validate
# --------------------------------------------------------------------------- #

def _describe_validation(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        where = ".".join(str(p) for p in err.get("loc", ())) or "args"
        if err.get("type") == "missing":
            parts.append(f"missing required argument '{where}'")
        elif err.get("type") == "extra_forbidden":
            parts.append(f"unexpected argument '{where}'")
        else:
            parts.append(f"argument '{where}': {err.get('msg')}")
    return "; ".join(parts)


def prepare(raw: str | Mapping | None, registry: Mapping[str, ToolSpec], context: Any = None) -> ToolCall | ToolOutcome:
    """Parse + resolve + validate. Returns a ready ToolCall, or a failure outcome."""
    parsed = parse_tool_call(raw)
    if isinstance(parsed, ToolOutcome):
        log.warning("tool call: parse failure -- %s", parsed.payload["error"])
        return parsed
    name = parsed["tool"]
    if not name:
        log.warning("tool call: no tool name")
        return _failure(None, "missing_tool", f"No tool named. Available tools: {', '.join(registry)}.")
    spec = registry.get(name)  # explicit allowlist lookup -- the only way a tool is found
    if spec is None:
        log.warning("tool call: unknown tool %r", name[:60])
        return _failure(name, "unknown_tool", f"Unknown tool '{name}'. Available tools: {', '.join(registry)}.")

    args = parsed["args"]
    if args is None:
        args = {}  # missing or null args -> {}; the schema then reports any REQUIRED argument as missing
    if not isinstance(args, Mapping):
        log.warning("tool call: %s args are %s, not an object", name, type(args).__name__)
        return _failure(name, "invalid_args", f"'args' must be a JSON object, got {type(args).__name__}.")
    log.info("tool call: requested %s args=%s", name, {k: _size(v) for k, v in args.items()})
    try:
        model = spec.args_model.model_validate(dict(args))
    except ValidationError as exc:
        message = _describe_validation(exc)
        log.warning("tool call: %s invalid args -- %s", name, message)
        return _failure(name, "invalid_args", message)
    if spec.validate is not None:
        try:
            problem = spec.validate(model, context)
        except Exception as exc:  # a broken validator must not crash the loop either
            log.exception("tool call: %s validator raised", name)
            return _failure(name, "rejected", f"validation could not complete: {type(exc).__name__}")
        if problem:
            log.warning("tool call: %s rejected -- %s", name, str(problem)[:200])
            return _failure(name, "rejected", str(problem))
    return ToolCall(tool=name, args=model)


def _size(value: Any) -> str:
    """Shape of an argument for logs -- never its content."""
    if isinstance(value, str):
        return f"str[{len(value)}]"
    if isinstance(value, (list, tuple, dict)):
        return f"{type(value).__name__}[{len(value)}]"
    return type(value).__name__


# --------------------------------------------------------------------------- #
# execute + normalize
# --------------------------------------------------------------------------- #

def execute(call: ToolCall, registry: Mapping[str, ToolSpec], context: Any = None) -> ToolOutcome:
    """Run an already-prepared call; exceptions become an `execution` failure."""
    spec = registry[call.tool]
    try:
        raw = spec.execute(call.args, context)
    except Exception as exc:
        log.warning("tool call: %s execution failed -- %s: %s", call.tool, type(exc).__name__, str(exc)[:200])
        # A tool may tag its exception (error_kind = "connectivity", ...) so the loop can route on it.
        out = _failure(call.tool, getattr(exc, "error_kind", "execution"), str(exc)[:800] or type(exc).__name__)
        out.call = call
        return out
    view = spec.summarize(raw) if spec.summarize else raw
    result = to_jsonable({} if view is None else view)
    log.info("tool call: %s ok", call.tool)
    return ToolOutcome({"success": True, "tool": call.tool, "result": result}, raw=raw, call=call)


def run(raw: str | Mapping | None, registry: Mapping[str, ToolSpec], context: Any = None) -> ToolOutcome:
    """Full pipeline: parse -> resolve -> validate -> execute -> normalize."""
    prepared = prepare(raw, registry, context)
    if isinstance(prepared, ToolOutcome):
        return prepared
    return execute(prepared, registry, context)


def to_jsonable(value: Any, _depth: int = 0) -> Any:
    """Make a tool result safe to json.dumps: NaN/inf -> None, Decimal -> float,
    dates -> ISO text, numpy scalars/arrays -> Python, DataFrames -> {columns, rows,
    row_count} (first rows only), Pydantic models / dataclasses -> dicts, sets ->
    lists; anything else -> str(value)."""
    if _depth > 20:
        return str(value)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, decimal.Decimal):
        return to_jsonable(float(value))
    if isinstance(value, (_dt.date, _dt.datetime, _dt.time)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_jsonable(v, _depth + 1) for v in value]
    if isinstance(value, BaseModel):
        return to_jsonable(value.model_dump(), _depth + 1)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(dataclasses.asdict(value), _depth + 1)
    type_name = type(value).__module__.split(".")[0]
    if type_name == "pandas" and hasattr(value, "to_dict") and hasattr(value, "columns"):
        head = value.head(_RESULT_ROWS)
        return {"columns": [str(c) for c in value.columns], "row_count": int(len(value)),
                "rows": to_jsonable(head.to_dict(orient="records"), _depth + 1)}
    if type_name == "numpy":
        return to_jsonable(value.tolist() if hasattr(value, "tolist") else value.item(), _depth + 1)
    if hasattr(value, "isoformat"):  # pandas Timestamp and friends
        return value.isoformat()
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)
