"""Top-K retrieval by cosine similarity (brute force, exact, NumPy-vectorised).

    hits = top_k(query_vector, embeddings, k=5)
    # [{"index": 42, "similarity": 0.91}, {"index": 7, "similarity": 0.88}, ...]

Used by gmr_agents/schema/retrieval.py for the embedding leg of hybrid search over
the vectors stored in outputs/knowledge.json (rows are normalised once at load,
so the search passes assume_normalized=True and skips all norm work).

Why cosine similarity
    Embedding models encode meaning in a vector's DIRECTION; its length mostly
    reflects text length / token statistics. Cosine = a.b / (|a| |b|) compares
    direction only, in [-1, 1], so a short question and a long example phrasing
    with the same meaning still score high. For unit vectors it is just the dot
    product, which is why rows are normalised once and reused.

Complexity (N candidates, d dimensions, K results)
    similarities   O(N d) time -- one matrix-vector product, unavoidable for exact
                   search; O(N) extra space (the score vector; norms: another N).
    selection:
      argsort      O(N log N): a comparison sort must order ALL N scores, though
                   only K are needed.
      heapq        O(N log K): a min-heap of size K; each of the N scores is
                   compared with the heap's smallest and, if larger, replaces it
                   in O(log K). But it iterates in Python -- per-element interpreter
                   overhead (~50-100 ns) dwarfs the C loops below for NumPy input.
      argpartition O(N) average (introselect): moves the K largest to one end
                   without ordering the rest; then only those K are sorted,
                   O(K log K). Runs in C on the array. Best for NumPy arrays of
                   any real size: total O(N + K log K).
    method="auto" uses argsort for small N (simplest, and the sort cost is
    negligible), argpartition otherwise. "heap" is kept for non-NumPy iterables
    and for comparison.

Brute force vs approximate nearest neighbours (ANN) / a vector database
    Exact brute force is right while N * d fits in memory and one matrix-vector
    product meets the latency budget: ~10^5-10^6 vectors of 384 floats is a few
    hundred MB and a few ms to tens of ms. This project has ~3,200 rows -- exact
    search takes microseconds, has perfect recall and needs no extra service.
    Past ~10^7 vectors, or with strict p99 latency, filtering and concurrent
    writes, use ANN (HNSW, IVF-PQ in FAISS / ScaNN, or a vector DB such as
    pgvector, Qdrant, Milvus): sub-linear search for a small recall loss, plus
    persistence, sharding and metadata filters.
"""

from __future__ import annotations

import heapq
from typing import Literal

import numpy as np

Method = Literal["auto", "argsort", "argpartition", "heap"]
_SMALL_N = 2_000  # below this a full sort is simpler and just as fast


def cosine_similarities(query_vector: np.ndarray, embeddings: np.ndarray, *,
                        assume_normalized: bool = False, eps: float = 1e-12) -> np.ndarray:
    """Cosine similarity of the query against every row, shape (N,).

    No copy of the embedding matrix is made: the dot products come from one
    matrix-vector product and the row norms are an N-vector. A zero-norm query or
    row gets similarity 0 (instead of a division by zero / NaN)."""
    q = np.asarray(query_vector, dtype=np.result_type(embeddings.dtype, np.float32)).ravel()
    if embeddings.ndim != 2 or embeddings.shape[1] != q.shape[0]:
        raise ValueError(f"embeddings must be (N, {q.shape[0]}), got {embeddings.shape}")
    dots = embeddings @ q                                   # (N,), no copy of embeddings
    if assume_normalized:
        return dots
    q_norm = float(np.linalg.norm(q))
    if q_norm <= eps:
        return np.zeros(embeddings.shape[0], dtype=dots.dtype)
    row_norms = np.linalg.norm(embeddings, axis=1)          # (N,)
    denom = row_norms * q_norm
    return np.divide(dots, denom, out=np.zeros_like(dots), where=denom > eps)


def _select(similarities: np.ndarray, k: int, method: Method) -> np.ndarray:
    """Indices of the k largest scores, best first."""
    n = similarities.shape[0]
    if method == "auto":
        method = "argsort" if n <= _SMALL_N else "argpartition"
    if method == "argsort":                                 # O(N log N)
        return np.argsort(similarities)[::-1][:k]
    if method == "argpartition":                            # O(N) + O(K log K)
        if k >= n:
            return np.argsort(similarities)[::-1]
        part = np.argpartition(similarities, n - k)[n - k:]  # the k largest, unordered
        return part[np.argsort(similarities[part])[::-1]]
    if method == "heap":                                    # O(N log K), Python-level loop
        best = heapq.nlargest(k, enumerate(similarities.tolist()), key=lambda x: x[1])
        return np.fromiter((i for i, _s in best), dtype=np.intp, count=len(best))
    raise ValueError(f"unknown method {method!r}")


def top_k(query_vector: np.ndarray, embeddings: np.ndarray, k: int, *, method: Method = "auto",
          assume_normalized: bool = False, min_similarity: float | None = None) -> list[dict]:
    """The k most similar rows, highest similarity first:
    [{"index": i, "similarity": s}, ...].

    k must be > 0; k > N returns all N; empty embeddings return []. Rows below
    `min_similarity` (when given) are left out, so fewer than k may come back."""
    if not isinstance(k, (int, np.integer)) or k <= 0:
        raise ValueError(f"k must be a positive integer, got {k!r}")
    embeddings = np.asarray(embeddings)
    if embeddings.size == 0:
        return []
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    sims = cosine_similarities(query_vector, embeddings, assume_normalized=assume_normalized)
    k = min(int(k), sims.shape[0])
    order = _select(sims, k, method)
    hits = [{"index": int(i), "similarity": float(sims[i])} for i in order]
    if min_similarity is not None:
        hits = [h for h in hits if h["similarity"] >= min_similarity]
    return hits


if __name__ == "__main__":
    # Small demo: 6 candidates in 3-D, one zero row, query close to row 0.
    candidates = np.array([[1.0, 0.1, 0.0],
                           [0.0, 1.0, 0.0],
                           [0.9, 0.2, 0.1],
                           [0.0, 0.0, 0.0],      # zero-norm row -> similarity 0
                           [-1.0, 0.0, 0.0],
                           [0.5, 0.5, 0.5]])
    query = np.array([1.0, 0.0, 0.0])
    for m in ("argsort", "argpartition", "heap"):
        print(m, top_k(query, candidates, k=3, method=m))
    print("k > N  ", top_k(query, candidates, k=99))
    print("empty  ", top_k(query, np.empty((0, 3)), k=3))
