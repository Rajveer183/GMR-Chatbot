"""Moved to gmr_agents/checks/grounding.py -- this alias keeps old imports working."""

import sys

from gmr_agents.checks import grounding as _moved

sys.modules[__name__] = _moved
