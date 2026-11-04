"""Moved to gmr_agents/supervisor/state.py -- this alias keeps old imports working."""

import sys

from gmr_agents.supervisor import state as _moved

sys.modules[__name__] = _moved
