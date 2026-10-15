"""Hand-curated table metadata for the Schema agent, loaded from
schema/data/table_metadata.yaml: which tables each domain exposes, and what
their columns mean in business terms.

  MDM_TABLES / WFM_TABLES / WFM_POSTGRES_TABLES   the curated table shortlists
  COLUMN_HINTS             (table, column) -> business meaning, shown to the SQL planner
  HEADER_LABEL_OVERRIDES   (table, column) -> display header for result tables
  TABLE_DESCRIPTIONS       table -> one-line purpose (planner context + retrieval)
  PLACE_NAME_ALIASES       alternate place names -> stored names

The live structure (columns, types, row counts, sample values) is read from
information_schema by catalog.AthenaCatalog.refresh(); this data is what
cannot be read from the database.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_DATA = yaml.safe_load((Path(__file__).resolve().parent / "data" / "table_metadata.yaml").read_text(encoding="utf-8"))


def _by_table_column(entries: dict[str, str]) -> dict[tuple[str, str], str]:
    """{"table.column": value} -> {(table, column): value}"""
    return {tuple(key.split(".", 1)): value for key, value in entries.items()}


MDM_TABLES: dict[str, list[str]] = _DATA["mdm_tables"]
WFM_TABLES: dict[str, list[str]] = _DATA["wfm_tables"]
WFM_POSTGRES_TABLES: list[str] = _DATA["wfm_postgres_tables"]
COLUMN_HINTS: dict[tuple[str, str], str] = _by_table_column(_DATA["column_hints"])
HEADER_LABEL_OVERRIDES: dict[tuple[str, str], str] = _by_table_column(_DATA["header_label_overrides"])
TABLE_DESCRIPTIONS: dict[str, str] = _DATA["table_descriptions"]
PLACE_NAME_ALIASES: dict[str, list[str]] = _DATA["place_name_aliases"]

# Every schema a query may qualify a table with: Postgres "public" plus the
# Athena databases named in the shortlists above. Kept here so the database
# names live only in table_metadata.yaml.
SCHEMA_NAMES: tuple[str, ...] = ("public", *dict.fromkeys([*MDM_TABLES, *WFM_TABLES]))
# Regex group matching any of SCHEMA_NAMES, e.g. rf"\b{SCHEMA_RE}\.(\w+)".
SCHEMA_RE: str = "(?:" + "|".join(SCHEMA_NAMES) + ")"
