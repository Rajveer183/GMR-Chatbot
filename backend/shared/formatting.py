"""Number formatting for the response cards."""

from __future__ import annotations

import math
from typing import Any


def _safe_number(value: Any) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(num) or math.isinf(num):
        return 0.0
    return num


def format_number(value: float, unit: str | None = None) -> str:
    num = _safe_number(value)
    if num == int(num) and abs(num) < 1e15:
        text = f"{int(num):,}"
    else:
        text = f"{num:,.2f}"
    return f"{text} {unit}".strip() if unit else text
