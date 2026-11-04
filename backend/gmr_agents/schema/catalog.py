"""Moved to gmr_agents/catalog/catalog.py -- this alias keeps old imports working."""

import sys

from gmr_agents.catalog import catalog as _moved

sys.modules[__name__] = _moved
