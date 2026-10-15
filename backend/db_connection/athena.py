"""Amazon Athena helper — SQL queries over the S3 data lake (boto3, no Spark).

    from db_connection.athena import run_query

    df = run_query("SELECT * FROM discom_gold.kpi_energy LIMIT 10")
"""
from __future__ import annotations

import io
import time
import boto3

import pandas as pd
from botocore.config import Config

from db_connection.config import AWS_REGION, PROJECT_BUCKET, PREFIX_ATHENA
from db_connection.readonly import assert_read_only

# botocore's defaults are a 60s connect AND 60s read timeout: measured live, a
# stalled HTTPS call over the VPN held meter lookups at ~65s (then retried and
# succeeded), while the same queries normally finish in 2-4s. Every call here
# is a small API request (start / poll / fetch a small CSV), so fail fast and
# retry instead of waiting out a dead connection.
_CLIENT_CONFIG = Config(connect_timeout=5, read_timeout=20, retries={"max_attempts": 4},
                        max_pool_connections=20)
_athena = boto3.client("athena", region_name=AWS_REGION, config=_CLIENT_CONFIG)
_s3 = boto3.client("s3", region_name=AWS_REGION, config=_CLIENT_CONFIG)


def _execute(query: str, database: str | None, poll_seconds: float) -> str:
    """Start a query, poll to completion, return its query execution id.

    Fast poll to start (Athena queries on small results often finish in under
    a second), backing off so a long-running query doesn't hammer the API.
    """
    # Athena has no read-only session: every statement is checked here before it is sent.
    assert_read_only(query)
    output_location = f"s3://{PROJECT_BUCKET}/{PREFIX_ATHENA}/"
    kwargs = {"QueryString": query, "ResultConfiguration": {"OutputLocation": output_location}}
    if database:
        kwargs["QueryExecutionContext"] = {"Database": database}

    qid = _athena.start_query_execution(**kwargs)["QueryExecutionId"]

    delay = 0.3
    while True:
        status = _athena.get_query_execution(QueryExecutionId=qid)["QueryExecution"]["Status"]["State"]
        if status in ("SUCCEEDED", "FAILED", "CANCELLED"):
            break
        time.sleep(delay)
        delay = min(delay * 1.5, poll_seconds)

    if status != "SUCCEEDED":
        reason = _athena.get_query_execution(QueryExecutionId=qid)["QueryExecution"]["Status"].get("StateChangeReason", "")
        raise RuntimeError(f"Athena query {status.lower()}: {reason}")
    return qid


def run_query(query: str, database: str | None = None, poll_seconds: float = 1.0) -> pd.DataFrame:
    """Run a SQL query against Athena and return the result as a DataFrame.

    `database` sets the query's default database context (so unqualified table
    names resolve), e.g. run_query("SELECT * FROM dim_meter", database="hes_gold").
    Result is fetched straight from S3 via boto3 -- not pandas' s3fs backend,
    which conflicts with the pinned botocore version here.
    """
    qid = _execute(query, database, poll_seconds)
    obj = _s3.get_object(Bucket=PROJECT_BUCKET, Key=f"{PREFIX_ATHENA}/{qid}.csv")
    return pd.read_csv(io.BytesIO(obj["Body"].read()), dtype=_text_columns(qid) or None)


def _text_columns(qid: str) -> dict[str, type]:
    """Text result columns read as text: CSV inference turned the consumer number '0061239000'
    into 61239000 (leading zeros lost, and the lookup typed from it found nothing)."""
    try:
        meta = _athena.get_query_results(QueryExecutionId=qid, MaxResults=1)["ResultSet"]["ResultSetMetadata"]
    except Exception:
        return {}
    return {c["Name"]: str for c in meta.get("ColumnInfo", []) if c.get("Type", "").lower() in ("varchar", "char", "string")}


def list_tables(database: str) -> list[str]:
    """SHOW TABLES IN <database>, as a plain list of names.

    DDL/SHOW queries write their result as a plain-text `.txt` file (one name
    per line), NOT the `.csv` a SELECT produces -- fetching `.csv` here 404s.
    """
    qid = _execute(f"SHOW TABLES IN {database}", database, poll_seconds=1.0)
    obj = _s3.get_object(Bucket=PROJECT_BUCKET, Key=f"{PREFIX_ATHENA}/{qid}.txt")
    names = obj["Body"].read().decode("utf-8").splitlines()
    return sorted(n.strip() for n in names if n.strip())

