"""A catalog cache written under a different table allowlist is not reused (6 Nov 2026).

A backend still running the old 5-table MDM list rewrote the shared mdm_catalog.json an hour after the list
grew to 22; the next backend to start served 5 tables and every power-quality question failed."""

import json
import time

from gmr_agents.catalog import catalog as C


def _write(tmp_path, monkeypatch, **extra):
    monkeypatch.setattr(C, "_CACHE_DIR", tmp_path)
    cat = C.AthenaCatalog("mdm")
    table = {"table_name": "mapping_audit", "database": "hes_gold", "domain": "mdm", "row_count": 1,
             "engine": "athena", "columns": [], "join_candidates": []}
    gone = {**table, "table_name": "no_longer_allowed"}
    cat._cache_path.write_text(json.dumps({"built_at": time.time(), "tables": [table, gone], **extra}))
    return cat


def test_cache_without_this_allowlist_is_refreshed(tmp_path, monkeypatch):
    cat = _write(tmp_path, monkeypatch)
    assert cat._load_cache() is None
    # last-resort fallback still serves it, minus tables outside the allowlist
    assert [t.table_name for t in cat._load_cache(ignore_age=True)] == ["mapping_audit"]


def test_cache_with_the_same_allowlist_is_used(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "_CACHE_DIR", tmp_path)
    cat = _write(tmp_path, monkeypatch, allowlist=C.AthenaCatalog("mdm")._allowlist())
    assert [t.table_name for t in cat._load_cache()] == ["mapping_audit"]
