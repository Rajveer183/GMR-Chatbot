"""Moved to gmr_agents/checks/entity_check.py -- this alias keeps old imports working."""

import sys

from gmr_agents.checks import entity_check as _moved

sys.modules[__name__] = _moved
