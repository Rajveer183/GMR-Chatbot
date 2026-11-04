"""Moved to gmr_agents/resolvers/scope.py -- this alias keeps old imports working."""

import sys

from gmr_agents.resolvers import scope as _moved

sys.modules[__name__] = _moved
