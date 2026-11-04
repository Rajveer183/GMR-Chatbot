"""Moved to gmr_agents/supervisor/context.py -- this alias keeps old imports working."""

import sys

from gmr_agents.supervisor import context as _moved

sys.modules[__name__] = _moved
