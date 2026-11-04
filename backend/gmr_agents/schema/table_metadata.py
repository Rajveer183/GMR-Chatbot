"""Moved to gmr_agents/catalog/table_metadata.py -- this alias keeps old imports working."""

import sys

from gmr_agents.catalog import table_metadata as _moved

sys.modules[__name__] = _moved
