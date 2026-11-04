"""Moved to gmr_agents/execution/tools/calling.py -- this alias keeps old imports working."""

import sys

from gmr_agents.execution.tools import calling as _moved

sys.modules[__name__] = _moved
