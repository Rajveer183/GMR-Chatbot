"""Moved to gmr_agents/execution/tools/calculator.py -- this alias keeps old imports working."""

import sys

from gmr_agents.execution.tools import calculator as _moved

sys.modules[__name__] = _moved
