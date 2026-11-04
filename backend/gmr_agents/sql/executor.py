"""Moved to gmr_agents/execution/executor.py -- this alias keeps old imports working."""

import sys

from gmr_agents.execution import executor as _moved

sys.modules[__name__] = _moved
