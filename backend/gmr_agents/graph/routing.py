"""Moved to gmr_agents/supervisor/routing.py -- this alias keeps old imports working."""

import sys

from gmr_agents.supervisor import routing as _moved

sys.modules[__name__] = _moved
