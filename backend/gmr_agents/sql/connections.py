"""The backend's database connectors: db_connection/athena.py (Athena, via
boto3) and db_connection/postgres.py (the WFM gold Postgres).

The connectors are imported on first use -- importing athena creates boto3
clients, and db_connection/config.py reads its settings at import time.
"""

from __future__ import annotations

import pandas as pd


def athena_query(sql: str, database: str | None = None) -> pd.DataFrame:
    from db_connection.athena import run_query

    return run_query(sql, database=database)


def postgres_connection():
    from db_connection.postgres import get_connection

    return get_connection()


def pooled_postgres_connection():
    """Context manager: a reused, time-limited connection for one query."""
    from db_connection.postgres import pooled_connection

    return pooled_connection()
