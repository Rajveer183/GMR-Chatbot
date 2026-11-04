"""Moved to gmr_agents/guards/sql_guard.py -- this alias keeps old imports working."""

import sys

from gmr_agents.guards import sql_guard as _moved

sys.modules[__name__] = _moved
