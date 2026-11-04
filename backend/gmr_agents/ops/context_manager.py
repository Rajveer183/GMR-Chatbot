"""Context-window management for LLM prompts: a token budget with priorities.

Who controls it: application code. A prompt is assembled from Sections, each with
a priority (P0 = never cut ... P6 = cut first) and a list of items (one example,
one past question, one chart rule ...). When the estimate exceeds the budget,
items are removed from the lowest-priority sections first, then whole optional
sections; required sections are never cut.

    P0  system / global rules, date                 required
    P1  the current question (+ its parts)          required, always LAST in the prompt
    P2  schema of the chosen tables                 required (already token-budgeted)
    P3  previous turn's SQL when it is a follow-up  required for a follow-up
    P4  verified examples                           keep at least one
    P5  retrieved column hints                      trimmable
    P6  past answered questions, chart guide        cut first

Positions (lost-in-the-middle): rules at the top, the question at the very end --
sections are emitted in their given order; trimming never moves them.

What this replaces / what it does not do
    * History is already compact (llm_pipeline._history_block: older turns as their
      questions, recent turns with short answers + an extracted "conversation focus"
      of projects / OEMs / categories / SAT stages / years / tables). No extra LLM
      call is spent summarising history -- that would add latency and tokens.
    * SQL results never re-enter later prompts: the answer writer sees a capped
      preview; later turns carry only key numbers and row keys.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from shared.llm_gateway import estimate_tokens

log = logging.getLogger(__name__)

_WS_RE = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS_RE.sub(" ", text.strip().lower())


def dedupe(items: list[str]) -> list[str]:
    """Drop exact and near-duplicate items, keeping the first (= most relevant,
    since every caller already lists its items best-first) occurrence. "Near-
    duplicate" = one item's normalised text is wholly contained in another's
    (a shorter hint subsumed by a longer one that says the same thing, or the
    same line retrieved twice from two different sources) -- never a partial
    text overlap, which would risk dropping two items that only share a
    clause."""
    kept: list[str] = []
    kept_norm: list[str] = []
    for item in items:
        n = _norm(item)
        if not n:
            continue
        if any(n == k or n in k or k in n for k in kept_norm):
            continue
        kept.append(item)
        kept_norm.append(n)
    return kept


@dataclass
class Section:
    name: str
    priority: int                 # 0 = most important
    items: list[str]
    header: str = ""              # printed once above the items
    required: bool = False        # never trimmed
    min_items: int = 0            # trimming keeps at least this many items
    joiner: str = "\n\n"
    trim_from: str = "end"        # "end" = items are best-first; "start" = best-last (kept nearest the question)

    def text(self) -> str:
        if not self.items:
            return ""
        body = self.joiner.join(self.items)
        return f"{self.header}\n{body}" if self.header else body


@dataclass
class BuiltContext:
    prompt: str
    tokens: int
    budget: int
    trimmed: dict[str, int] = field(default_factory=dict)   # section -> items removed
    dropped: list[str] = field(default_factory=list)         # sections removed entirely
    over_budget: bool = False                                # required content alone exceeds the budget
    components: dict[str, int] = field(default_factory=dict) # tokens per section in the final prompt

    def metrics(self) -> dict:
        return {"tokens": self.tokens, "budget": self.budget, "components": self.components or None,
                "trimmed": self.trimmed or None,
                "dropped": self.dropped or None, "truncation": bool(self.trimmed or self.dropped),
                "over_budget": self.over_budget or None}


def build(sections: list[Section], *, budget: int, system: str = "") -> BuiltContext:
    """Assemble sections (in the given order) within `budget` tokens, counting the
    system prompt too. Lowest priority is reduced first: one item at a time down to
    min_items, then the whole section if it is optional.

    Every section's items are deduplicated first (see dedupe()) -- a retrieved hint
    and a rule that already say the same thing, or the same past question pulled in
    from two sources, would otherwise cost tokens twice and crowd a DIFFERENT item
    out of the budget for no benefit."""
    for s in sections:
        deduped = dedupe(s.items)
        if len(deduped) != len(s.items):
            log.info("context: '%s' deduplicated %d -> %d item(s)", s.name, len(s.items), len(deduped))
        s.items = deduped
    system_tokens = estimate_tokens(system)

    def total() -> int:
        return system_tokens + sum(estimate_tokens(s.text()) for s in sections if s.items) + 4 * len(sections)

    trimmed: dict[str, int] = {}
    dropped: list[str] = []
    # Lowest priority first; within a priority, the section placed later in the prompt
    # first (sections are listed in order of value, e.g. past questions after the chart guide).
    order = {id(s): i for i, s in enumerate(sections)}
    for section in sorted((s for s in sections if not s.required), key=lambda s: (-s.priority, -order[id(s)])):
        while total() > budget and len(section.items) > section.min_items:
            section.items.pop(0 if section.trim_from == "start" else -1)  # drop the least relevant item
            trimmed[section.name] = trimmed.get(section.name, 0) + 1
        if total() > budget and section.items and section.min_items == 0:
            trimmed[section.name] = trimmed.get(section.name, 0) + len(section.items)
            section.items = []
        if section.name in trimmed and not section.items:
            dropped.append(section.name)
            trimmed.pop(section.name, None)
        if total() <= budget:
            break
    prompt = "\n\n".join(t for t in (s.text() for s in sections) if t)
    tokens = total()
    built = BuiltContext(prompt, tokens, budget, trimmed, dropped, tokens > budget,
                         {s.name: estimate_tokens(s.text()) for s in sections if s.items})
    if built.over_budget:
        log.warning("context: required content is ~%d tokens, over the %d budget -- sent as is", tokens, budget)
    elif trimmed or dropped:
        log.info("context: trimmed to ~%d/%d tokens (items cut %s, sections dropped %s)", tokens, budget, trimmed, dropped)
    return built


def fits_hard_limit(built: BuiltContext, hard_limit: int) -> bool:
    """Final check before the LLM call: the model's real context limit (with
    head-room). Past it the caller fails gracefully instead of sending."""
    return built.tokens <= hard_limit
