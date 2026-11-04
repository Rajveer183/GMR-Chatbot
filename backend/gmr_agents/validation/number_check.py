"""Moved to gmr_agents/checks/number_check.py -- this alias keeps old imports working."""

import sys

from gmr_agents.checks import number_check as _moved

sys.modules[__name__] = _moved
