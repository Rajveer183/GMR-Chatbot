"""Moved to gmr_agents/ops/answer_cache.py -- this alias keeps old imports working."""

import sys

from gmr_agents.ops import answer_cache as _moved

sys.modules[__name__] = _moved
