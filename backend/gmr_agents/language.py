"""Moved to gmr_agents/resolvers/language.py -- this alias keeps old imports working."""

import sys

from gmr_agents.resolvers import language as _moved

sys.modules[__name__] = _moved
