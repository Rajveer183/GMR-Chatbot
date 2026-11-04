"""Moved to gmr_agents/resolvers/query_encoder.py -- this alias keeps old imports working."""

import sys

from gmr_agents.resolvers import query_encoder as _moved

sys.modules[__name__] = _moved
