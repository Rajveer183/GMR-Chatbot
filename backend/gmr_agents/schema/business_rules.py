"""Business rules for the LLM planner, loaded from
schema/data/business_rules.yaml (the knowledge that used to live in the
hard-coded planner/fast_paths templates).

The file is re-read automatically when it changes on disk, so a rule can be
corrected without restarting the backend.

    global.rules          apply to every question
    topics.<key>          domain, title, description, keywords, tables,
                          rules, example {question, sql}
    explain_only.<term>   glossary for "what does X mean" questions
"""

from __future__ import annotations

import logging
import os
import re
import threading
from typing import Any

import yaml

log = logging.getLogger(__name__)

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
PATH = os.path.join(_DATA_DIR, "business_rules.yaml")
EXAMPLES_PATH = os.path.join(_DATA_DIR, "examples.yaml")
CHARTS_PATH = os.path.join(_DATA_DIR, "charts.yaml")
_LOCK = threading.Lock()
_CACHE: dict[str, dict[str, Any]] = {}


def _load(path: str) -> dict[str, Any]:
    """YAML file contents, re-read when its mtime changes; a broken edit keeps the previous version."""
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return {}
    with _LOCK:
        entry = _CACHE.setdefault(path, {"mtime": None, "data": {}})
        if entry["mtime"] != mtime:
            try:
                with open(path, encoding="utf-8") as fh:
                    entry["data"] = yaml.safe_load(fh) or {}
                entry["mtime"] = mtime
                log.info("%s loaded.", os.path.basename(path))
            except Exception:
                log.exception("%s could not be parsed -- keeping the previous version.", os.path.basename(path))
        return entry["data"]


def _data() -> dict[str, Any]:
    return _load(PATH)


def topics() -> dict[str, dict]:
    return dict(_data().get("topics") or {})


def global_rules() -> list[str]:
    return list((_data().get("global") or {}).get("rules") or [])


def glossary() -> dict[str, str]:
    return dict(_data().get("explain_only") or {})


def topic_menu(question: str | None = None, full: int = 0) -> str:
    """One line per topic for the triage prompt: key [domain] title -- description.
    With a question and `full`, only the `full` best-ranked topics carry their
    description (the rest: key and title, still choosable). Measured 2026-10-04: the
    right topic ranks 0-3 for 18/18 eval questions; full menu ~1,310 tokens."""
    all_topics = topics()
    described = set(rank_topics(question, k=full)) if question and full else set(all_topics)
    return "\n".join(
        f"- {key} [{t.get('domain', 'wfm')}] {t.get('title', key)}"
        + (f": {t.get('description', '')}" if key in described else "")
        for key, t in all_topics.items()
    )


def valid_topics(keys: list[str] | None, domain: str | None = None) -> list[str]:
    all_topics = topics()
    out = []
    for k in keys or []:
        k = str(k).strip()
        if k in all_topics and k not in out and (domain is None or all_topics[k].get("domain", "wfm") == domain):
            out.append(k)
    return out


def tables_for(topic_keys: list[str]) -> list[str]:
    all_topics = topics()
    out: list[str] = []
    for k in topic_keys:
        for t in (all_topics.get(k) or {}).get("tables") or []:
            if t not in out:
                out.append(t)
    return out


_STOP = {"the", "a", "an", "of", "in", "on", "for", "to", "is", "are", "was", "were", "what", "which", "how", "many",
          "much", "me", "our", "we", "do", "does", "did", "by", "and", "or", "with", "show", "give", "i", "it", "this",
          "that", "be", "have", "has", "been", "so", "far", "there", "at", "as", "all", "any", "only", "?"}


def _words(text: str) -> set[str]:
    out = set()
    for w in re.findall(r"[a-z0-9&%]+", str(text).lower()):
        if w in _STOP:
            continue
        out.add(w[:-1] if len(w) > 4 and w.endswith("s") else w)  # crude plural fold
    return out


def examples() -> list[dict]:
    return [e for e in (_load(EXAMPLES_PATH).get("examples") or []) if isinstance(e, dict) and e.get("sql")]


def _example_items() -> list[tuple[str, list[str]]]:
    """(example id, its texts) -- every phrasing, plus the example's note (it names
    the KPI and its terms), as indexed by the hybrid retriever."""
    items = []
    for i, ex in enumerate(examples()):
        q = ex["q"] if isinstance(ex.get("q"), list) else [ex.get("q", "")]
        texts = [str(x) for x in q if str(x).strip()]
        if ex.get("suggest") and str(ex["suggest"]) not in texts:
            texts.append(str(ex["suggest"]))  # offered verbatim as a follow-up: must route here
        if ex.get("note"):
            texts.append(str(ex["note"])[:300])
        items.append((str(i), texts))
    return items


def suggestions() -> dict[str, str]:
    """{example id: its `suggest` question} -- the only questions ever offered as
    suggested follow-ups (each asked live and its answer checked)."""
    return {str(i): str(ex["suggest"]).strip() for i, ex in enumerate(examples()) if str(ex.get("suggest") or "").strip()}


def is_suggestion(question: str) -> bool:
    """True when the question is, word for word, one of the offered follow-ups."""
    norm = " ".join(str(question).lower().split()).rstrip(" ?.")
    return any(norm == " ".join(q.lower().split()).rstrip(" ?.") for q in suggestions().values())


def expand_query(question: str) -> str:
    """The question plus the programme terms for any everyday words in it
    (business_rules.yaml retrieval_synonyms) -- used for retrieval only."""
    text = f" {str(question).lower()} "
    extra = [str(v) for k, v in (_data().get("retrieval_synonyms") or {}).items()
             if f" {str(k).lower()} " in text or (" " in str(k) and str(k).lower() in text)]
    return f"{question} {' '.join(extra)}".strip() if extra else str(question)


def _topic_items() -> list[tuple[str, list[str]]]:
    return [(key, [f"{t.get('title', key)}. {t.get('description', '')}",
                   ", ".join(map(str, t.get("keywords") or []))])
            for key, t in topics().items()]


def similar_examples(question: str, k: int = 2) -> list[dict]:
    """The k verified examples closest to the question by hybrid retrieval
    (BM25 + local embeddings, RRF -- schema/retrieval.py). Worked patterns for
    the planner prompt; the LLM still writes the SQL."""
    from gmr_agents.schema import knowledge

    all_ex = examples()
    return [all_ex[int(i)] for i in knowledge.examples_for(question, k)]


def rank_topics(question: str, k: int = 3) -> list[str]:
    """Topics most likely to hold the rules for this question (merged mode
    pre-selects their rules + schema), by hybrid retrieval over each topic's
    title, description and keywords."""
    from gmr_agents.schema import knowledge

    ranked = [key for key, _score in knowledge.search("topics", question, k)]
    return ranked or list(topics())[:1]


def warm_retrieval() -> dict[str, int]:
    """Load the models and build every retrieval index (startup)."""
    from gmr_agents.schema import knowledge

    return knowledge.warm_all()


def chart_types() -> list[str]:
    return [str(c.get("type")) for c in (_load(CHARTS_PATH).get("charts") or []) if c.get("type")]


def chart_guide(question: str | None = None) -> str:
    """Few-shot chart guide for the SQL planner: when each chart type is the
    best match, with one worked example each (schema/data/charts.yaml). With a
    question, the chart types are listed most relevant first (retrieval)."""
    data = _load(CHARTS_PATH)
    lines = [f"* {r}" for r in data.get("rules") or []]
    charts = list(data.get("charts") or [])
    if question:
        from gmr_agents.schema import knowledge

        order = {t: i for i, t in enumerate(knowledge.chart_order(question))}
        charts.sort(key=lambda c: order.get(str(c.get("type")), len(order)))
    from shared import config

    with_examples = "chart_examples" not in config.PROMPT_COMPACT
    # The best-ranked CHART_GUIDE_FULL types get 'when' + example; the rest are named only
    # (still choosable, and a type the user names is always honoured). Measured 2026-10-04:
    # the right type ranks 0-1 for 15/15 chart eval questions; full guide ~1,340 tokens.
    full = config.CHART_GUIDE_FULL if question and config.CHART_GUIDE_FULL else len(charts)
    for c in charts[:full]:
        lines.append(f"- {c.get('type')}: {c.get('when', '')}" + (f"\n  e.g. {c.get('example', '')}" if with_examples else ""))
    if charts[full:]:
        lines.append("- also available: " + ", ".join(str(c.get("type")) for c in charts[full:]))
    return "\n".join(lines)


def example_menu(question: str | None = None, k: int | None = None) -> str:
    """Example lines for the triage prompt (the LLM picks the ones that MEAN the
    same). With a question, only the k best hybrid-retrieval candidates are listed
    -- ids stay the global example ids -- which keeps the triage prompt small."""
    from gmr_agents.schema import knowledge

    all_ex = examples()
    if question and k:
        ids = [int(i) for i in knowledge.examples_for(question, k)]
    else:
        ids = list(range(len(all_ex)))
    lines = []
    for i in ids:
        q = all_ex[i]["q"] if isinstance(all_ex[i].get("q"), list) else [all_ex[i].get("q", "")]
        lines.append(f"e{i}: {q[0][:90]}")
    return "\n".join(lines)


def pick_examples(chosen: list[str] | None, question: str, k: int = 3) -> list[dict]:
    """The triage LLM's chosen examples first, topped up by hybrid retrieval."""
    all_ex = examples()
    out: list[dict] = []
    for key in chosen or []:
        key = str(key).strip().lower().lstrip("e")
        if key.isdigit() and int(key) < len(all_ex) and all_ex[int(key)] not in out:
            out.append(all_ex[int(key)])
    for ex in similar_examples(question, k):
        if len(out) >= k:
            break
        if ex not in out:
            out.append(ex)
    return out[:k]


def examples_blocks(question: str, k: int = 2, chosen: list[str] | None = None) -> list[str]:
    """One text block per verified example (best first) -- the unit the context
    manager trims by."""
    parts = []
    for ex in pick_examples(chosen, question, k):
        q = ex["q"][0] if isinstance(ex.get("q"), list) else ex.get("q", "")
        block = [f"Q: {q}"]
        if ex.get("note"):
            block.append(f"Note: {ex['note']}")
        block.append(str(ex["sql"]).strip())
        for e in ex.get("extra") or []:
            block.append(f"extra_query ({e.get('purpose', '')}):\n{str(e.get('sql', '')).strip()}")
        parts.append("\n".join(block))
    return parts


def answer_notes(topic_keys: list[str]) -> str:
    """Short per-topic context for the answer writer (topics.<key>.answer_notes):
    what a KPI means and its known caveats, so 'why' answers reason correctly."""
    all_topics = topics()
    lines: list[str] = []
    for k in topic_keys:
        lines.extend(f"- {n}" for n in (all_topics.get(k) or {}).get("answer_notes") or [])
    return "\n".join(lines)


def rules_text(topic_keys: list[str], *, with_examples: bool = True) -> str:
    all_topics = topics()
    parts = ["GLOBAL RULES:"] + [f"- {r}" for r in global_rules()]
    for k in topic_keys:
        t = all_topics.get(k)
        if not t:
            continue
        parts.append(f"\nTOPIC {k} -- {t.get('title', k)}:")
        parts.extend(f"- {r}" for r in t.get("rules") or [])
        ex = t.get("example") or {}
        if with_examples and ex.get("sql"):
            parts.append(f"Example -- Q: {ex.get('question', '')}\n{str(ex['sql']).strip()}")
    return "\n".join(parts)
