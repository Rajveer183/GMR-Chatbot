"""Moved to gmr_agents/checks/extreme_check.py -- this alias keeps old imports working."""

import sys

from gmr_agents.checks import extreme_check as _moved

sys.modules[__name__] = _moved
