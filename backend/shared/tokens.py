"""Token counting for budgets, rate limits and cost estimates.

Who controls it: application code. Every provider tokenizes differently, so an
exact count before sending is impossible; what matters is a count that is close
and stable. Measured on real SQL-planner prompts (NVIDIA usage.prompt_tokens):

    chars / 3.6           -12% .. +6%   (SQL and numbers pack more tokens per char)
    tiktoken o200k_base    -12% .. -7%  (consistently a little low -> correctable)

So: count with tiktoken (the gpt-oss models' own tokenizer family), then scale by
a per-provider factor that starts from measured values and keeps learning from
the prompt_tokens each response reports (exponential moving average, clamped).
Without tiktoken it falls back to chars / 3.6.
"""

from __future__ import annotations

import functools
import threading

from shared import config

try:
    import tiktoken

    _ENC = tiktoken.get_encoding("o200k_base")
except Exception:  # tiktoken missing or no encoding file: rough fallback
    _ENC = None

_LOCK = threading.Lock()
_FACTORS: dict[str, float] = dict(config.TOKEN_FACTORS)
_DEFAULT = "default"


@functools.lru_cache(maxsize=4096)
def count(text: str) -> int:
    """Base token count (tiktoken o200k_base)."""
    if not text:
        return 0
    if _ENC is None:
        return int(len(text) / 3.6) + 1
    return len(_ENC.encode(text, disallowed_special=()))


def factor(provider: str | None = None) -> float:
    with _LOCK:
        return _FACTORS.get(provider or _DEFAULT) or _FACTORS.get(_DEFAULT, 1.0)


def estimate(text: str, provider: str | None = None) -> int:
    """Expected provider token count for `text` (base count x provider factor)."""
    n = count(text or "")
    return int(n * factor(provider) + 0.999) if n else 0


def learn(provider: str, base_count: int, actual_tokens: int | None) -> None:
    """Move the provider's factor toward actual / base (EMA 0.1, clamped 0.7 .. 1.5)."""
    if not actual_tokens or base_count < 50:
        return
    ratio = min(1.5, max(0.7, actual_tokens / base_count))
    with _LOCK:
        current = _FACTORS.get(provider) or _FACTORS.get(_DEFAULT, 1.0)
        _FACTORS[provider] = round(0.9 * current + 0.1 * ratio, 4)

