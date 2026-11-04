"""Moved to gmr_agents/execution/connections.py -- this alias keeps old imports working."""

import sys

from gmr_agents.execution import connections as _moved

sys.modules[__name__] = _moved
