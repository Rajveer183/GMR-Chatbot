"""Connection settings for Athena / S3 / Glue / Postgres -- all values come
from backend/.env (the same file shared/config.py reads). No secrets or
environment-specific values are hardcoded here.
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass


def _require(name: str) -> str:
    """Return an env var or raise a clear error if missing."""
    value = os.getenv(name)
    if value is None:
        raise RuntimeError(
            f"Required environment variable '{name}' is not set. "
            f"Check your .env file."
        )
    return value


# ── AWS ────────────────────────────────────────────────────────────
AWS_REGION            = _require("AWS_REGION")
AWS_ACCESS_KEY_ID     = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# ── S3 data lake ───────────────────────────────────────────────────
PROJECT_BUCKET    = _require("PROJECT_BUCKET")
PREFIX_BRONZE     = os.getenv("PREFIX_BRONZE", "bronze")
PREFIX_SILVER     = os.getenv("PREFIX_SILVER", "silver")
PREFIX_HES_SILVER = os.getenv("PREFIX_HES_SILVER", "silver/hes_silver")
PREFIX_GOLD       = os.getenv("PREFIX_GOLD", "gold")
PREFIX_HES_GOLD   = os.getenv("PREFIX_HES_GOLD", "gold/hes_gold")
PREFIX_ATHENA     = os.getenv("PREFIX_ATHENA", "athena-results")

# ── Glue databases (Iceberg tables) ────────────────────────────────
GLUE_DB_BRONZE     = _require("GLUE_DB_BRONZE")
GLUE_DB_SILVER     = _require("GLUE_DB_SILVER")
GLUE_DB_HES_SILVER = _require("GLUE_DB_HES_SILVER")
GLUE_DB_GOLD       = _require("GLUE_DB_GOLD")
GLUE_DB_HES_GOLD   = _require("GLUE_DB_HES_GOLD")

# ── PostgreSQL / RDS ───────────────────────────────────────────────
DB_HOST      = os.getenv("DB_HOST")
DB_NAME      = os.getenv("DB_NAME")
DB_USER      = os.getenv("DB_USER", "postgres")
DB_PASSWORD  = os.getenv("DB_PASSWORD")
DATABASE_URL = os.getenv("DATABASE_URL")
