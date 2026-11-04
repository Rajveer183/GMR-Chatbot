"""Moved to gmr_agents/supervisor/graph.py -- this alias keeps old imports working."""

import sys

from gmr_agents.supervisor import graph as _moved

sys.modules[__name__] = _moved
