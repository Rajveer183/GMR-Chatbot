"""Moved to gmr_agents/catalog/schema_brief.py -- this alias keeps old imports working."""

import sys

from gmr_agents.catalog import schema_brief as _moved

sys.modules[__name__] = _moved
