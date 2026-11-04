"""Moved to gmr_agents/retrieval/retrieval.py -- this alias keeps old imports working."""

import sys

from gmr_agents.retrieval import retrieval as _moved

sys.modules[__name__] = _moved
