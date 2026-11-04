"""Moved to gmr_agents/catalog/schema_dictionary.py -- this alias keeps old imports working."""

import sys

from gmr_agents.catalog import schema_dictionary as _moved

sys.modules[__name__] = _moved
