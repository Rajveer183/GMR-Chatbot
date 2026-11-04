"""Moved to gmr_agents/ops/prewarm.py -- this alias keeps old imports working."""

import sys

from gmr_agents.ops import prewarm as _moved

sys.modules[__name__] = _moved
