"""Grounding guard -- Retrieve -> Ground -> Generate -> Cite -> Abstain when unsupported.

Who controls it: application code. The LLM writes the explanation and names its
citations; everything that decides whether that answer may be shown is here:
relevance threshold, evidence size, citation validation, number support, abstention,
the answer-confidence score. Two paths, one rule:

    RAG grounds the knowledge (definitions, rules, formulas) -- explain answers.
    SQL grounds the numbers                                   -- data answers.

Explain answers (KPI meaning, statuses, formulas)
    1. evidence  = topic rules + glossary terms, each a chunk with id, source,
                   location and its cosine relevance to the question
    2. threshold = chunks below GROUNDING_MIN_RELEVANCE are dropped; none left ->
                   abstain without calling the LLM (no evidence, no answer)
    3. generate  = prompt shows ONLY the chunks ([chunk_id=...]) and asks for
                   {summary, insights, citations, supported, abstain, conflicts}
    4. check     = citations must be ids that were shown (fabricated ones dropped);
                   numbers must appear in the cited evidence; the model's own
                   abstain / unsupported flag is honoured; conflicts are reported
    5. confidence= from relevance, citation coverage, number support and
                   consistency -- NOT the similarity score itself
Data answers
    numbers are checked against the SQL result (validation/number_check.py); the
    sources shown are the tables queried, the data refresh time and the verified
    example / rule topics the query was built from -- produced by code, never by
    the LLM, so they cannot be fabricated.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from gmr_agents.validation import number_check
from shared import config

ABSTAIN_NO_EVIDENCE = ("I don't have enough information in the programme's rules and definitions to answer "
                       "that reliably.")
# Order used to present (and break ties between) sources -- configured in code, not chosen by the LLM.
SOURCE_PRIORITY = ("business_rules.yaml#topics", "kpi_definitions.yaml", "business_rules.yaml#glossary")


@dataclass
class Evidence:
    chunk_id: str
    source: str
    location: str
    content: str
    similarity: float


@dataclass
class GroundedResult:
    summary: str
    insights: list[str]
    citations: list[dict]                  # [{"id", "source", "location"}] -- only real, shown chunks
    abstained: bool
    confidence: float                      # answer confidence (0-1), separate from retrieval similarity
    problems: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# evidence
# --------------------------------------------------------------------------- #

def collect_explain_evidence(question: str, topics: list[str], *, k: int) -> list[Evidence]:
    """Rule topics + glossary terms with their cosine relevance, best first,
    de-duplicated, filtered by GROUNDING_MIN_RELEVANCE, at most k chunks."""
    from gmr_agents.schema import business_rules, knowledge

    topic_rel = knowledge.relevance("topics", question) or {}
    term_rel = knowledge.relevance("glossary", question) or {}
    chunks: list[Evidence] = []
    all_topics = business_rules.topics()
    for key in dict.fromkeys([*topics, *sorted(topic_rel, key=lambda t: -topic_rel[t])[:k]]):
        t = all_topics.get(key)
        if not t:
            continue
        lines = [f"{t.get('title', key)}: {t.get('description', '')}", *map(str, t.get("answer_notes") or []),
                 *map(str, t.get("rules") or [])]
        chunks.append(Evidence(f"rule:{key}", SOURCE_PRIORITY[0], str(t.get("title", key)), "\n".join(lines),
                               topic_rel.get(key, 0.0)))
    terms = business_rules.glossary()
    for key in sorted(term_rel, key=lambda t: -term_rel[t])[:k]:
        if key in terms:
            chunks.append(Evidence(f"term:{key}", SOURCE_PRIORITY[2], key, f"{key}: {terms[key]}", term_rel[key]))
    picked = select(chunks, k=k, min_relevance=config.GROUNDING_MIN_RELEVANCE)
    return picked + kpi_evidence(question, k=config.GROUNDING_KPI_CHUNKS, exclude={c.content for c in picked})


def kpi_evidence(question: str, *, k: int, exclude: set[str] | None = None) -> list[Evidence]:
    """The most relevant kpi_definitions.yaml chunks (logical units: one KPI section
    each), each child expanded with its KPI overview so it reads on its own."""
    from gmr_agents.schema import chunking, knowledge

    if k <= 0:
        return []
    rel = knowledge.relevance("kpis", question) or {}
    by_id = knowledge.kpi_chunks()
    best = [cid for cid in sorted(rel, key=lambda c: -rel[c]) if rel[cid] >= config.GROUNDING_MIN_RELEVANCE][:k]
    out = []
    for cid in chunking.expand(best, by_id):
        c = by_id[cid]
        if exclude and c.text in exclude:
            continue
        where = f"{c.meta.get('title') or c.meta.get('table') or c.section} -- {c.section}"
        out.append(Evidence(cid, SOURCE_PRIORITY[1], where, c.text, rel.get(cid, 0.0)))
    return out


def select(chunks: list[Evidence], *, k: int, min_relevance: float) -> list[Evidence]:
    """Threshold, de-duplicate (same content), order by relevance then source priority, cap at k."""
    seen, out = set(), []
    rank = {s: i for i, s in enumerate(SOURCE_PRIORITY)}
    for c in sorted(chunks, key=lambda c: (-c.similarity, rank.get(c.source, 99))):
        digest = hashlib.sha1(" ".join(c.content.split()).lower().encode("utf-8")).hexdigest()
        if c.similarity < min_relevance or digest in seen:
            continue
        seen.add(digest)
        out.append(c)
    return out[:k]


def context_block(chunks: list[Evidence], max_chars: int) -> str:
    """The evidence as the LLM sees it: one [chunk_id=...] block per chunk, capped."""
    parts, used = [], 0
    for c in chunks:
        block = f"[chunk_id={c.chunk_id}] ({c.source} -- {c.location})\n{c.content}"
        if used + len(block) > max_chars and parts:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #

_CITE_RE = re.compile(r"\s*\[(?:chunk_id=)?((?:rule|term|kpi|table):[\w\-:#+.]+|pipeline)\]")


def verify(answer: dict[str, Any], chunks: list[Evidence]) -> GroundedResult:
    """Apply the code-side checks to the LLM's structured answer."""
    by_id = {c.chunk_id: c for c in chunks}
    summary = str(answer.get("summary") or "").strip()
    insights = [str(i).strip() for i in answer.get("insights") or [] if str(i).strip()]
    problems: list[str] = []

    # Same chunk written differently ("chunk_id=term:x", "[term:x]") is the same citation.
    cited = [_norm_cite(c) for c in answer.get("citations") or [] if isinstance(c, (str, int))]
    cited += _CITE_RE.findall(" ".join([summary, *insights]))       # inline [rule:x] markers count too
    real = [c for c in dict.fromkeys(cited) if c in by_id]
    fabricated = [c for c in dict.fromkeys(cited) if c not in by_id]
    if fabricated:
        problems.append(f"dropped citations to chunks that were not provided: {', '.join(fabricated)}")
    summary = _CITE_RE.sub("", summary).strip()
    insights = [_CITE_RE.sub("", i).strip() for i in insights]

    model_abstained = bool(answer.get("abstain")) or answer.get("supported") is False
    if model_abstained or not summary:
        return GroundedResult(summary or ABSTAIN_NO_EVIDENCE, [], [], True, 0.0, problems + ["model abstained"])
    if not real:
        problems.append("no valid citation")
        return GroundedResult(ABSTAIN_NO_EVIDENCE, [], [], True, 0.0, problems)

    # Numbers (thresholds, days, percentages) must appear in the cited evidence.
    evidence_numbers = number_check.allowed_values([], " ".join(by_id[c].content for c in real))
    summary, left = number_check.repair(summary, evidence_numbers, [])
    if left:
        problems.append(f"numbers not in the evidence: {', '.join(left)}")
        return GroundedResult(ABSTAIN_NO_EVIDENCE, [], [], True, 0.0, problems)
    kept = []
    for text in insights:
        if number_check.unsupported(text, evidence_numbers):
            problems.append("dropped an insight with a number not in the evidence")
        else:
            kept.append(text)

    conflicts = [str(c) for c in answer.get("conflicts") or [] if str(c).strip()]
    confidence = _confidence(chunks, real, numbers_ok=True, conflicts=bool(conflicts))
    return GroundedResult(summary, kept, [_citation(by_id[c]) for c in real], False, confidence, problems, conflicts)


def _norm_cite(value) -> str:
    text = str(value).strip().strip("[]()").strip()
    return re.sub(r"^chunk_id\s*[=:]\s*", "", text).strip()


def _confidence(chunks: list[Evidence], cited: list[str], *, numbers_ok: bool, conflicts: bool) -> float:
    """Answer confidence: evidence relevance x citation coverage x checks passed.
    Retrieval similarity alone is only one factor -- a highly similar chunk that
    the answer does not cite, or that conflicts with another, does not make the
    answer reliable."""
    by_id = {c.chunk_id: c for c in chunks}
    relevance = max((by_id[c].similarity for c in cited), default=0.0)
    coverage = min(1.0, len(cited) / 2)            # one citation = 0.5, two or more = 1.0
    score = 0.5 * relevance + 0.3 * coverage + (0.2 if numbers_ok else 0.0)
    return round(score * (0.6 if conflicts else 1.0), 2)


def _citation(c: Evidence) -> dict:
    return {"id": c.chunk_id, "source": c.source, "location": c.location}


def data_citations(plan: dict, triage: dict, refreshed_at: str | None) -> list[dict]:
    """Sources of a data answer, built by code: the live query and the verified
    pattern / rules it was written from. Never produced by the LLM."""
    out = [{"id": "sql", "source": "Live query", "location": ", ".join(plan.get("tables") or []) +
            (f" (data refreshed {refreshed_at})" if refreshed_at else "")}]
    from gmr_agents.schema import business_rules

    examples = business_rules.examples()
    for e in triage.get("examples") or []:
        i = str(e).lstrip("e")
        if i.isdigit() and int(i) < len(examples):
            q = examples[int(i)].get("q")
            out.append({"id": f"example:{i}", "source": "examples.yaml (verified)",
                        "location": (q[0] if isinstance(q, list) else str(q))[:120]})
    titles = business_rules.topics()
    for t in (triage.get("topics") or [])[:2]:
        if t in titles:
            out.append({"id": f"rule:{t}", "source": "business_rules.yaml", "location": str(titles[t].get("title", t))})
    return out
