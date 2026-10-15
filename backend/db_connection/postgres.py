"""Direct PostgreSQL/RDS connection helper (installation & asset master data).

    import pandas as pd
    from db_connection.postgres import get_connection, pooled_connection

    conn = get_connection()
    df = pd.read_sql("SELECT * FROM etl_unified_smart_meter_asset LIMIT 10", conn)
    conn.close()

    with pooled_connection() as conn:   # per-question queries: reused, time-limited
        df = pd.read_sql("SELECT 1", conn)
"""
from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import contextmanager

import psycopg2

from db_connection.config import DB_HOST, DB_NAME, DB_USER, DB_PASSWORD

log = logging.getLogger(__name__)

# A question's query may run this long before Postgres cancels it. Verified
# live: over a stalling VPN a query otherwise waited 94s with no error.
STATEMENT_TIMEOUT_MS = int(os.environ.get("PG_STATEMENT_TIMEOUT_MS", "30000"))
_POOL_MAX = int(os.environ.get("PG_POOL_MAX", "8"))  # idle connections kept for reuse
# TCP keepalives: a connection whose VPN path died is noticed in ~25s
# instead of hanging on the OS default (hours).
_KEEPALIVES = {"keepalives": 1, "keepalives_idle": 10, "keepalives_interval": 5, "keepalives_count": 3}


def _connect(**extra):
    # Every session is read-only: Postgres itself refuses writes, whichever code path
    # opened the connection (db_connection/readonly.py lists all the layers).
    options = " ".join(filter(None, [extra.pop("options", ""), "-c default_transaction_read_only=on"]))
    return psycopg2.connect(
        options=options,
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        sslmode="require",
        connect_timeout=5,
        **_KEEPALIVES,
        **extra,
    )


def get_connection():
    """Return a live psycopg2 connection to the analytics Postgres/RDS instance.

    connect_timeout is set explicitly -- without it, psycopg2/libpq falls
    back to the OS TCP connect timeout (verified live on this Windows
    machine: ~50-60s) when the host is genuinely unreachable, e.g. a VPN
    drop. That's the connection attempt only -- it does not shorten a slow
    but successful query; pooled_connection() below does."""
    return _connect()


# Idle connections, most recently returned last: (connection, returned_at).
# Not psycopg2.pool: it keeps only `minconn` idle connections and opens that
# many up front -- with minconn=0 every returned connection was closed (no
# reuse, measured), and a larger minconn means several 5s connect attempts
# in a row whenever the VPN is down.
_idle: list[tuple[object, float]] = []
_idle_lock = threading.Lock()
# RDS / the VPN's NAT may drop a connection idle this long; a fresh one is
# cheaper than a first query that fails on it.
_MAX_IDLE_SECONDS = 240.0


def _close_quietly(conn) -> None:
    try:
        conn.close()
    except Exception:
        pass


def _take_idle():
    now = time.monotonic()
    with _idle_lock:
        while _idle:
            conn, returned_at = _idle.pop()
            if not conn.closed and now - returned_at < _MAX_IDLE_SECONDS:
                return conn
            _close_quietly(conn)
    return None


@contextmanager
def pooled_connection():
    """A reused connection for one question's query. Opening a new SSL
    connection costs ~190ms over the VPN (measured) -- most of a template
    answer's time. Autocommit, so nothing is left "idle in transaction";
    queries are cancelled after STATEMENT_TIMEOUT_MS. A connection that
    raised is closed rather than returned, so a dead one is never reused."""
    conn = _take_idle()
    if conn is None:
        # Read-only session: even a write that slipped past the SQL guard is refused by
        # Postgres itself ("cannot execute ... in a read-only transaction").
        conn = _connect(options=f"-c statement_timeout={STATEMENT_TIMEOUT_MS}")
        conn.autocommit = True
    broken = False
    try:
        yield conn
    except Exception:
        broken = True
        raise
    finally:
        if broken or conn.closed:
            _close_quietly(conn)
        else:
            with _idle_lock:
                keep = len(_idle) < _POOL_MAX
                if keep:
                    _idle.append((conn, time.monotonic()))
            if not keep:
                _close_quietly(conn)
