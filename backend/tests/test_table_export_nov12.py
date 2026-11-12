"""POST /api/table-export: the answer's validated SQL re-run read-only, no LLM call."""
import io

import pandas as pd
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from gmr_agents import main
from gmr_agents.execution import executor


@pytest.fixture
def client(monkeypatch):
    seen = {}

    def fake_answer_sql(trace_id, table_index):
        if trace_id != "t1" or table_index != 0:
            raise HTTPException(status_code=404, detail="No table for that answer -- ask the question again.")
        return "SELECT zone, n FROM public.x ORDER BY n DESC LIMIT 5", "postgres", None

    def fake_run_sql(sql, database, engine):
        seen["sql"] = sql
        return pd.DataFrame({"zone": ["Agra", "Kashi"], "n": [3, 2]})

    monkeypatch.setattr(main, "_answer_sql", fake_answer_sql)
    monkeypatch.setattr(executor, "run_sql", fake_run_sql)
    return TestClient(main.app), seen


def test_csv_full_table_with_labels(client):
    c, seen = client
    r = c.post("/api/table-export", json={"trace_id": "t1", "format": "csv", "labels": ["Zone", "Meters"], "title": "Top zones"})
    assert r.status_code == 200
    assert "LIMIT 5" not in seen["sql"] and seen["sql"].endswith(f"LIMIT {main.EXPORT_MAX_ROWS}")
    assert r.text.lstrip("﻿").splitlines() == ["Zone,Meters", "Agra,3", "Kashi,2"]
    assert r.headers["x-row-count"] == "2" and r.headers["x-truncated"] == "0"
    assert 'filename="Top zones.csv"' in r.headers["content-disposition"]


def test_xlsx(client):
    c, _ = client
    r = c.post("/api/table-export", json={"trace_id": "t1", "format": "xlsx"})
    assert r.status_code == 200
    df = pd.read_excel(io.BytesIO(r.content))
    assert list(df.columns) == ["zone", "n"] and len(df) == 2


def test_label_count_mismatch_keeps_sql_names(client):
    c, _ = client
    r = c.post("/api/table-export", json={"trace_id": "t1", "labels": ["only one"]})
    assert r.text.lstrip("﻿").splitlines()[0] == "zone,n"


def test_bad_inputs(client):
    c, _ = client
    assert c.post("/api/table-export", json={"trace_id": "t1", "format": "pdf"}).status_code == 422
    assert c.post("/api/table-export", json={"trace_id": "missing"}).status_code == 404
