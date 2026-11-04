"""Moved to gmr_agents/ops/context_manager.py -- this alias keeps old imports working."""

import sys

from gmr_agents.ops import context_manager as _moved

sys.modules[__name__] = _moved
