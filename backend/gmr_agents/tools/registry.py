"""Moved to gmr_agents/execution/tools/registry.py -- this alias keeps old imports working."""

import sys

from gmr_agents.execution.tools import registry as _moved

sys.modules[__name__] = _moved
