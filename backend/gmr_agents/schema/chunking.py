"""Moved to gmr_agents/retrieval/chunking.py -- this alias keeps old imports working."""

import sys

from gmr_agents.retrieval import chunking as _moved

sys.modules[__name__] = _moved
