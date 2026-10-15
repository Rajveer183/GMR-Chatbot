"""Hybrid retrieval over every knowledge source the chatbot uses.

Indexes (built by gmr_agents/schema/business_rules.py):
    examples  verified question -> SQL patterns (every phrasing + note)
    topics    KPI rule areas (title, description, keywords)
    glossary  KPI / term definitions (explain answers)
    tables    the live table catalog (description + columns)
    charts    the chart guide (when each chart type fits)

Ranking for a query:
    1. Fuzzy typo fix   query words missing from the index vocabulary are mapped to
                        the closest vocabulary word when rapidfuzz ratio >= FUZZY_MIN_RATIO
                        (0.85 .. 1): "instalation" -> installation, "kasi" -> kashi.
    2. BM25             keyword ranking (Okapi) over every text of every item.
    3. Embeddings       meaning, from a local fastembed model (RETRIEVAL_EMBED_MODEL:
                        bge-small, or nomic-embed-text with its search_query /
                        search_document prefixes) -- offline, no quota.
    4. RRF fusion       score = sum of weight / (RRF_K + rank) over BM25 and embeddings;
                        an item's rank is its best text's rank.
    5. Near-exact match a text whose fuzzy ratio to the question is >= FUZZY_MIN_RATIO
                        puts its item first (highest ratio first).
    Metadata filter     search(allow=...) restricts steps 2-5 to the given items BEFORE
                        ranking (e.g. only the columns of the tables the planner chose).

Storage: BM25 in memory + vectors from the JSON knowledge store (exact cosine
Top-K, schema/topk.py).

Caching:
    vectors   outputs/knowledge.json (indexes.<name>.rows[].v16, base64 float16): incremental -- only
              new rows (owner + text) are embedded, removed rows are dropped, a model /
              prefix change re-embeds everything
    rebuilds  a changed source rebuilds its index in a background thread while the
              current index keeps serving -- no question waits on re-indexing; until the
              rebuild lands (or after it fails -- logged, retried on the next search) only
              items whose texts are unchanged are returned: changed / deleted ones never
    indexes   BM25 + vocabulary in memory, keyed by the same content hash
    queries   LRU of query embeddings (RETRIEVAL_QUERY_CACHE_SIZE)
    results   LRU of search results keyed by (index hash, query, k, settings)
              (RETRIEVAL_RESULT_CACHE_SIZE) -- triage and planner reuse one search
If the embedding model cannot load, retrieval degrades to BM25 instead of failing.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import math
import os
import re
import threading
from collections import Counter, OrderedDict

import numpy as np

from gmr_agents.schema import topk
from shared import config

log = logging.getLogger(__name__)

_LOCK = threading.RLock()
_EMBEDDER = {"model": None, "failed": False, "name": None}
_INDEX: dict[str, dict] = {}

_STOP = {"the", "a", "an", "of", "in", "on", "for", "to", "is", "are", "was", "were", "what", "which", "how", "many",
         "much", "me", "our", "we", "do", "does", "did", "by", "and", "or", "with", "show", "give", "i", "it", "this",
         "that", "be", "have", "has", "been", "so", "far", "there", "at", "as", "all", "any", "only", "can", "you",
         "please", "tell", "hai", "hain", "ka", "ki", "ke", "ko", "se", "mein", "kya"}


def tokens(text: str) -> list[str]:
    """Lower-case word tokens with stop words removed and a crude plural fold."""
    out = []
    for w in re.findall(r"[a-z0-9&%+]+", str(text).lower()):
        if w in _STOP:
            continue
        out.append(w[:-1] if len(w) > 4 and w.endswith("s") else w)
    return out


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9%&+ ]+", " ", str(text).lower())).strip()


class _LRU:
    def __init__(self, size: int):
        self.size, self.data = max(0, size), OrderedDict()

    def get(self, key):
        with _LOCK:
            if key in self.data:
                self.data.move_to_end(key)
                return self.data[key]
        return None

    def put(self, key, value) -> None:
        if not self.size:
            return
        with _LOCK:
            self.data[key] = value
            self.data.move_to_end(key)
            while len(self.data) > self.size:
                self.data.popitem(last=False)


_QUERY_VECTORS = _LRU(config.RETRIEVAL_QUERY_CACHE_SIZE)
_RESULTS = _LRU(config.RETRIEVAL_RESULT_CACHE_SIZE)


# --------------------------------------------------------------------------- #
# BM25 (Okapi)
# --------------------------------------------------------------------------- #

class BM25:
    def __init__(self, docs: list[list[str]], k1: float, b: float):
        self.docs, self.k1, self.b = docs, k1, b
        self.n = len(docs)
        self.avgdl = sum(len(d) for d in docs) / max(self.n, 1)
        df = Counter(t for d in docs for t in set(d))
        self.idf = {t: math.log(1 + (self.n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(d) for d in docs]
        self.norm = [self.k1 * (1 - self.b + self.b * len(d) / max(self.avgdl, 1e-9)) for d in docs]

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(self.n)
        q = [t for t in query if t in self.idf]
        if not q:
            return out
        for i, tf in enumerate(self.tf):
            s = 0.0
            for t in q:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.norm[i])
            out[i] = s
        return out


# --------------------------------------------------------------------------- #
# Models (local fastembed)
# --------------------------------------------------------------------------- #

def _model_dir() -> str:
    return os.path.join(config.PROJECT_ROOT, config.RETRIEVAL_MODEL_DIR)


# ONNX models fastembed does not ship, registered on first use: name -> (HF repo, dim,
# pooling, model file, extra files). BGE-M3: multilingual dense, CLS pooling, 1024-d.
_CUSTOM_MODELS = {"BAAI/bge-m3": ("BAAI/bge-m3", 1024, "CLS", "onnx/model.onnx", ["onnx/model.onnx_data"])}


def _register_custom(name: str) -> None:
    from fastembed import TextEmbedding
    from fastembed.common.model_description import ModelSource, PoolingType

    spec = _CUSTOM_MODELS.get(name)
    if not spec or any(m["model"] == name for m in TextEmbedding.list_supported_models()):
        return
    repo, dim, pooling, model_file, extra = spec
    TextEmbedding.add_custom_model(model=name, pooling=PoolingType[pooling], normalization=True,
                                   sources=ModelSource(hf=repo), dim=dim, model_file=model_file,
                                   additional_files=extra)


def _embedder():
    name = config.RETRIEVAL_EMBED_MODEL
    with _LOCK:
        if _EMBEDDER["name"] != name:
            _EMBEDDER.update(model=None, failed=False, name=name)
        if _EMBEDDER["model"] is None and not _EMBEDDER["failed"] and config.RETRIEVAL_EMBED_ENABLED:
            try:
                from fastembed import TextEmbedding

                _register_custom(name)
                _EMBEDDER["model"] = TextEmbedding(name, cache_dir=_model_dir())
                log.info("retrieval: embedding model %s loaded", name)
            except Exception:
                _EMBEDDER["failed"] = True
                log.warning("retrieval: embedding model %s unavailable -- BM25 only", name, exc_info=True)
        return _EMBEDDER["model"]


def _embed(texts: list[str], prefix: str) -> np.ndarray | None:
    model = _embedder()
    if model is None or not texts:
        return None
    vecs = np.array(list(model.embed([f"{prefix}{t}" for t in texts])), dtype=np.float32)
    return vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)


def _query_vector(query: str) -> np.ndarray | None:
    key = (config.RETRIEVAL_EMBED_MODEL, config.RETRIEVAL_QUERY_PREFIX, query)
    cached = _QUERY_VECTORS.get(key)
    if cached is not None:
        return cached
    vec = _embed([query], config.RETRIEVAL_QUERY_PREFIX)
    if vec is not None:
        _QUERY_VECTORS.put(key, vec[0])
        return vec[0]
    return None


def _row_id(owner: str, text: str) -> str:
    return hashlib.sha1(f"{owner}{text}".encode("utf-8")).hexdigest()


def _encode_vector(vec: np.ndarray) -> str:
    """float16 bytes, base64 -- a third of the JSON size of a float list, 14x faster to
    write. Ranking unchanged: on 349 logged questions every top-1 that differed from
    float32 was a tie (score gap <= 0.00008)."""
    return base64.b64encode(np.asarray(vec, dtype=np.float16).tobytes()).decode("ascii")


def _decode_vector(row: dict) -> np.ndarray | None:
    if row.get("v16"):
        return np.frombuffer(base64.b64decode(row["v16"]), dtype=np.float16).astype(np.float32)
    if row.get("vector"):  # stores written before float16 storage -- migrated on the next save
        return np.asarray(row["vector"], dtype=np.float32)
    return None


def _sync_vectors(name: str, texts: list[str], owners: list[str]) -> np.ndarray | None:
    """Embedding matrix (one normalised row per text) for this index, persisted with the
    texts in the JSON knowledge store (as base64 float16, `v16`). Incremental: vectors of
    rows whose id (owner + text) is already stored are reused, only new rows are embedded;
    rows no longer in the source are dropped. A change of embedding model / prefix
    re-embeds everything."""
    from gmr_agents.schema import knowledge_store

    enabled = config.RETRIEVAL_EMBED_ENABLED and _embedder() is not None
    tag = f"{config.RETRIEVAL_EMBED_MODEL}|{config.RETRIEVAL_DOC_PREFIX}" if enabled else None
    saved = knowledge_store.load_index(name)
    have = {r["id"]: _decode_vector(r) for r in saved["rows"]} if saved.get("embed_tag") == tag else {}
    legacy = any("vector" in r for r in saved["rows"])
    ids = [_row_id(o, t) for o, t in zip(owners, texts)]
    todo = list({i: t for i, t in zip(ids, texts) if enabled and have.get(i) is None}.items())
    if todo:
        vecs = _embed([t for _i, t in todo], config.RETRIEVAL_DOC_PREFIX)
        if vecs is None:
            return None
        for (i, _t), v in zip(todo, vecs):
            have[i] = v
    rows = [{"item": o, "id": i, "text": t, **({"v16": _encode_vector(have[i])} if enabled else {})}
            for o, t, i in zip(owners, texts, ids)]
    stale = len({r["id"] for r in saved["rows"]} - set(ids))
    if todo or stale or legacy or saved.get("embed_tag") != tag or len(saved["rows"]) != len(rows):
        knowledge_store.save_index(name, tag, rows)
        log.info("retrieval: '%s' +%d embedded, -%d stale, %d rows saved to %s",
                 name, len(todo), stale, len(rows), knowledge_store.PATH)
    if not enabled or not rows:
        return None
    mat = np.stack([have[i] for i in ids]).astype(np.float32)
    return mat / np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-9)


# --------------------------------------------------------------------------- #
# Index
# --------------------------------------------------------------------------- #

_BUILDING: set[str] = set()


def _build_safe(name: str, texts: list[str], owners: list[str], key: str) -> None:
    """Background rebuild. A failure is logged and the index stays marked stale (results
    are revalidated against the source, see search()); the next search retries."""
    try:
        _build(name, texts, owners, key)
    except Exception:
        log.exception("retrieval: rebuilding '%s' failed -- serving revalidated results only, will retry", name)
    finally:
        with _LOCK:
            _BUILDING.discard(name)


def _build(name: str, texts: list[str], owners: list[str], key: str) -> dict:
    docs = [tokens(t) for t in texts]
    idx = {"key": key, "texts": texts, "owners": owners, "norm": [_norm(t) for t in texts],
           "bm25": BM25(docs, config.RETRIEVAL_BM25_K1, config.RETRIEVAL_BM25_B),
           "vocab": sorted({t for d in docs for t in d if len(t) >= 3})}
    try:
        idx["vectors"] = _sync_vectors(name, texts, owners)
    except Exception:
        idx["vectors"] = None
        log.warning("retrieval: could not sync '%s' vectors with the JSON knowledge store", name, exc_info=True)
    with _LOCK:
        _INDEX[name] = idx
        _BUILDING.discard(name)
    return idx


def _key(items: list[tuple[str, list[str]]]) -> str:
    """Content hash of an index source (every text + its owner + the embedding model)."""
    texts = [t for _id, ts in items for t in ts]
    owners = [_id for _id, ts in items for _t in ts]
    return hashlib.sha1(json.dumps([texts, owners, config.RETRIEVAL_EMBED_MODEL]).encode("utf-8")).hexdigest()


def _index(name: str, items: list[tuple[str, list[str]]], wait: bool = False) -> dict:
    """items = [(item_id, [text, ...])]; one row per text, mapped back to its item.
    When a source changed and an index already exists, the new one is built in a
    background thread and the current one keeps serving (no question waits on a
    rebuild); `wait=True` (startup warm-up) builds in place."""
    texts = [t for _id, ts in items for t in ts]
    owners = [_id for _id, ts in items for _t in ts]
    key = _key(items)
    with _LOCK:
        idx = _INDEX.get(name)
        if idx and idx["key"] == key:
            return idx
        if idx and not wait:
            if name not in _BUILDING:
                _BUILDING.add(name)
                threading.Thread(target=_build_safe, args=(name, texts, owners, key), daemon=True,
                                 name=f"reindex-{name}").start()
                log.info("retrieval: '%s' changed -- rebuilding in the background", name)
            return idx
    return _build(name, texts, owners, key)


def _fix_typos(words: list[str], idx: dict) -> list[str]:
    """Words unknown to this index -> the closest vocabulary word, when ratio >= FUZZY_MIN_RATIO."""
    if not config.FUZZY_ENABLED or not idx["vocab"]:
        return words
    from rapidfuzz import fuzz, process

    known = set(idx["bm25"].idf)
    out = []
    for w in words:
        if w in known or len(w) < 4 or w.isdigit():
            out.append(w)
            continue
        hit = process.extractOne(w, idx["vocab"], scorer=fuzz.ratio, score_cutoff=config.FUZZY_MIN_RATIO * 100)
        out.append(hit[0] if hit else w)
    return out


def _near_exact(query: str, idx: dict) -> dict[str, float]:
    """Items with a text whose fuzzy ratio to the question is in [FUZZY_MIN_RATIO, 1]."""
    if not config.FUZZY_ENABLED:
        return {}
    from rapidfuzz import fuzz

    q = _norm(query)
    hits: dict[str, float] = {}
    for owner, text in zip(idx["owners"], idx["norm"]):
        ratio = fuzz.ratio(q, text) / 100
        if ratio >= config.FUZZY_MIN_RATIO and ratio > hits.get(owner, 0.0):
            hits[owner] = ratio
    return hits


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #

def _settings() -> tuple:
    return (config.RETRIEVAL_EMBED_MODEL, config.RETRIEVAL_BM25_WEIGHT, config.RETRIEVAL_EMBED_WEIGHT,
            config.RETRIEVAL_MIN_SIMILARITY, config.FUZZY_ENABLED, config.FUZZY_MIN_RATIO, config.RETRIEVAL_EMBED_ENABLED)


def item_ranks(order, scores, owners: list[str]) -> dict[str, int]:
    """Rows in score order -> each ITEM's rank (0 = best) by its best-scoring text.
    Rows with score <= 0 do not count as retrieved; an item with several matching
    texts is counted once (de-duplication)."""
    best: dict[str, int] = {}
    for row in order:
        if scores[row] <= 0:
            break
        owner = owners[row]
        if owner not in best:
            best[owner] = len(best)
    return best


def rrf_fuse(rankings: dict[str, dict[str, int]], weights: dict[str, float], k: float) -> dict[str, float]:
    """Reciprocal Rank Fusion: score(d) = sum over rankers of weight / (k + rank(d) + 1).
    Uses ranks only, so BM25 scores and cosine similarities never need to be comparable;
    an item found by both rankers near the top beats one found by a single ranker."""
    fused: dict[str, float] = {}
    for source, ranks in rankings.items():
        w = weights.get(source, 1.0)
        for item, r in ranks.items():
            fused[item] = fused.get(item, 0.0) + w / (k + r + 1)
    return fused


def search(name: str, items: list[tuple[str, list[str]]], query: str, k: int,
           rerank_query: str | None = None, debug: list | None = None,
           allow=None) -> list[tuple[str, float]]:
    """Top-k (item id, score) for query, best first. `query` may be synonym-expanded
    for recall; `rerank_query` (the plain question) drives reranking and fuzzy
    near-exact matching. `allow` (item ids) is a metadata pre-filter: only those
    items are ranked by BM25, embeddings and fuzzy matching, so items outside it
    cannot crowd the wanted ones out of the top k. Pass a list as `debug` to
    receive, per returned item, its BM25 rank, dense rank, RRF score, final rank
    and source (bm25 / dense / both)."""
    if not items or not str(query).strip():
        return []
    idx = _index(name, items)
    plain = rerank_query or query
    allow = frozenset(allow) if allow is not None else None
    current = {o: ts for o, ts in items}
    stale = idx["key"] != _key(items)
    if stale:
        # The source changed and the new index is still building (or its build failed):
        # never serve a changed or deleted item from the old index. An item counts only
        # when its texts are unchanged (ids like example numbers shift when a YAML entry
        # is inserted / removed, so the same id can mean a different item).
        old: dict[str, list[str]] = {}
        for o, t in zip(idx["owners"], idx["texts"]):
            old.setdefault(o, []).append(t)
        fresh = frozenset(o for o, ts in old.items() if current.get(o) == ts)
        allow = fresh if allow is None else allow & fresh
    cache_key = (name, idx["key"], stale, query, plain, k, _settings(), allow)
    cached = _RESULTS.get(cache_key) if debug is None else None
    if cached is not None:
        return cached
    mask = None if allow is None else np.fromiter((o in allow for o in idx["owners"]), bool, len(idx["owners"]))

    rrf_k = config.RETRIEVAL_RRF_K
    rankings: dict[str, dict[str, int]] = {}
    weights = {"bm25": config.RETRIEVAL_BM25_WEIGHT, "dense": config.RETRIEVAL_EMBED_WEIGHT}

    def add(order: np.ndarray, scores: np.ndarray, source: str) -> None:
        # One ranker failing / finding nothing leaves the other one's ranking intact.
        rankings[source] = item_ranks(order, scores, idx["owners"])

    words = _fix_typos(tokens(query), idx)
    bm = idx["bm25"].scores(words)
    if mask is not None:
        bm[~mask] = 0
    add(np.argsort(-bm), bm, "bm25")
    if idx.get("vectors") is not None:
        qv = _query_vector(" ".join(words) if words != tokens(query) else query)
        if qv is not None:
            # Exact cosine Top-K over the stored vectors (both sides already unit length):
            # one matrix-vector product + argpartition, no full sort (schema/topk.py).
            # With a filter, only the allowed rows are scored.
            rows = np.arange(len(idx["texts"])) if mask is None else np.flatnonzero(mask)
            hits = topk.top_k(qv, idx["vectors"][rows], config.RETRIEVAL_VECTOR_TOP_N, assume_normalized=True,
                              min_similarity=config.RETRIEVAL_MIN_SIMILARITY) if len(rows) else []
            sims = np.zeros(len(idx["texts"]))
            for h in hits:
                sims[rows[h["index"]]] = h["similarity"]
            add(np.array([rows[h["index"]] for h in hits], dtype=np.intp), sims, "dense")
    fused = rrf_fuse(rankings, weights, rrf_k)
    ranked = sorted(fused.items(), key=lambda kv: -kv[1])

    near = _near_exact(plain, idx)
    if allow is not None:
        near = {o: r for o, r in near.items() if o in allow}
    if near:
        # A near-exact wording match outranks everything (ratio 0.85..1, highest first).
        top = sorted(near.items(), key=lambda kv: -kv[1])
        ranked = [(o, 10.0 + r) for o, r in top] + [kv for kv in ranked if kv[0] not in near]

    result = ranked[:k]
    _RESULTS.put(cache_key, result)
    if debug is not None:
        bm_r, de_r = rankings.get("bm25", {}), rankings.get("dense", {})
        for final, (item, score) in enumerate(result):
            source = "both" if item in bm_r and item in de_r else "bm25" if item in bm_r else \
                "dense" if item in de_r else "fuzzy"
            debug.append({"item": item, "final_rank": final + 1, "bm25_rank": bm_r.get(item, -1) + 1 or None,
                          "dense_rank": de_r.get(item, -1) + 1 or None, "rrf_score": round(fused.get(item, 0.0), 5),
                          "score": round(score, 5), "source": source, "near_exact": item in near})
    return result


def similarities(name: str, items: list[tuple[str, list[str]]], query: str) -> dict[str, float] | None:
    """Best cosine similarity of the query to each item's texts (embeddings only --
    a RELEVANCE score in [-1, 1], unlike search()'s fused ranking score). None when
    the embedding model is unavailable."""
    idx = _index(name, items)
    qv = _query_vector(query)
    if qv is None or idx.get("vectors") is None:
        return None
    best: dict[str, float] = {}
    for hit in topk.top_k(qv, idx["vectors"], len(idx["owners"]), assume_normalized=True):
        owner = idx["owners"][hit["index"]]
        best.setdefault(owner, hit["similarity"])  # hits come best first
    return best


def item_texts(name: str, item: str) -> list[str]:
    """The indexed texts of one item (empty when the index is not built yet)."""
    idx = _INDEX.get(name)
    if idx:
        return [t for o, t in zip(idx["owners"], idx["texts"]) if o == item]
    from gmr_agents.schema import knowledge_store

    return knowledge_store.documents(name, item)


def warm(items_by_name: dict[str, list[tuple[str, list[str]]]]) -> dict[str, int]:
    """Load the models and build/refresh every index (startup). Returns texts per index."""
    sizes = {}
    for name, items in items_by_name.items():
        sizes[name] = len(_index(name, items, wait=True)["texts"])
    return sizes


def reset() -> None:
    """Forget loaded models, indexes and caches (evaluation switches models in-process)."""
    with _LOCK:
        _INDEX.clear()
        _QUERY_VECTORS.data.clear()
        _RESULTS.data.clear()
        _EMBEDDER.update(model=None, failed=False, name=None)
