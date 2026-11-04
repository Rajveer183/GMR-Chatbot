"""Moved to gmr_agents/execution/dialect.py -- this alias keeps old imports working."""

import sys

from gmr_agents.execution import dialect as _moved

sys.modules[__name__] = _moved
