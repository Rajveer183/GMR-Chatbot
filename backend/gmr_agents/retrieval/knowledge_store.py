"""Local knowledge store -- plain JSON files under outputs/, no database, no vector DB:

    knowledge.json   (KNOWLEDGE_FILE)
        indexes      every retrieval index: {name: {embed_tag, rows: [{item, id, text, v16}]}}
                     -- the texts BM25 is built from and their embedding vectors
                     (base64 float16); retrieval loads them instead of re-embedding
        artifacts    named JSON documents: the data skeleton, the schema dictionary
                     {name: {built_at, body}}
    sql_results.json (SQL_RESULT_FILE)
                     local materialised query results keyed by sha1(engine + sql), each
                     tagged with the data version -- a repeated query since the last data
                     refresh is served from here without touching Postgres / Athena

Writes are atomic (temp file + replace). Read-only towards the source databases.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time

from shared import config

PATH = os.path.join(config.OUTPUT_DIR, config.KNOWLEDGE_FILE)
RESULTS_PATH = os.path.join(config.OUTPUT_DIR, config.SQL_RESULT_FILE)
# One lock per file: rewriting knowledge.json (after every answered question the past-questions
# index changes) must not stall SQL-result cache reads in the request path.
_LOCKS = {PATH: threading.RLock(), RESULTS_PATH: threading.RLock()}
_CACHE: dict[str, dict] = {}


def _lock(path: str) -> threading.RLock:
    return _LOCKS.setdefault(path, threading.RLock())


def _load(path: str, empty: dict) -> dict:
    with _lock(path):
        if path not in _CACHE:
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, ValueError):
                data = {}
            _CACHE[path] = {**empty, **data}
        return _CACHE[path]


def _save(path: str) -> None:
    with _lock(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_CACHE[path], f, ensure_ascii=False, separators=(",", ":"), default=str)
        os.replace(tmp, path)


def _knowledge() -> dict:
    return _load(PATH, {"indexes": {}, "artifacts": {}})


# --------------------------------------------------------------------------- #
# indexes (texts + embedding vectors)
# --------------------------------------------------------------------------- #

def load_index(name: str) -> dict:
    """{embed_tag, rows: [{item, id, text, v16}]} -- empty when never saved."""
    return _knowledge()["indexes"].get(name) or {"embed_tag": None, "rows": []}


def save_index(name: str, embed_tag: str | None, rows: list[dict]) -> None:
    with _lock(PATH):
        _knowledge()["indexes"][name] = {"embed_tag": embed_tag, "updated_at": time.time(), "rows": rows}
        _save(PATH)


def documents(index_name: str, item_id: str) -> list[str]:
    return [r["text"] for r in load_index(index_name)["rows"] if r["item"] == item_id]


def counts() -> dict[str, int]:
    return {name: len(idx.get("rows", [])) for name, idx in _knowledge()["indexes"].items()}


# --------------------------------------------------------------------------- #
# artifacts
# --------------------------------------------------------------------------- #

def save_artifact(name: str, data: dict) -> None:
    with _lock(PATH):
        # Round-trip through JSON so non-JSON values (dates, ...) are stored as strings.
        body = json.loads(json.dumps(data, ensure_ascii=False, default=str))
        _knowledge()["artifacts"][name] = {"built_at": time.time(), "body": body}
        _save(PATH)


def load_artifact(name: str) -> tuple[dict, float | None]:
    art = _knowledge()["artifacts"].get(name)
    return (art["body"], art["built_at"]) if art else ({}, None)


# --------------------------------------------------------------------------- #
# sql_results -- local materialised query results
# --------------------------------------------------------------------------- #

def _results() -> dict:
    return _load(RESULTS_PATH, {"results": {}})["results"]


def _key(sql: str, engine: str) -> str:
    return hashlib.sha1(f"{engine}\x1f{' '.join(sql.split())}".encode("utf-8")).hexdigest()


def get_result(sql: str, engine: str, version: str) -> tuple[list[str], list[list]] | None:
    if not config.SQL_RESULT_CACHE_ENABLED or not version:
        return None
    with _lock(RESULTS_PATH):
        hit = _results().get(_key(sql, engine))
        if not hit or hit["version"] != version:
            return None
        hit["hits"] = hit.get("hits", 0) + 1
        return hit["columns"], hit["rows"]


def put_result(sql: str, engine: str, version: str, columns: list[str], rows: list[list]) -> None:
    if not config.SQL_RESULT_CACHE_ENABLED or not version or len(rows) > config.SQL_RESULT_CACHE_MAX_ROWS:
        return
    with _lock(RESULTS_PATH):
        results = _results()
        for k in [k for k, v in results.items() if v["version"] != version]:
            del results[k]  # an old data version is never reused
        results[_key(sql, engine)] = {"version": version, "engine": engine, "sql": sql, "columns": columns,
                                      "rows": json.loads(json.dumps(rows, ensure_ascii=False, default=str)),
                                      "created_at": time.time(), "hits": 0}
        _save(RESULTS_PATH)
