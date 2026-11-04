"""Moved to gmr_agents/execution/hierarchy.py -- this alias keeps old imports working."""

import sys

from gmr_agents.execution import hierarchy as _moved

sys.modules[__name__] = _moved
