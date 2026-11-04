"""Moved to gmr_agents/retrieval/data_skeleton.py -- this alias keeps old imports working."""

import sys

from gmr_agents.retrieval import data_skeleton as _moved

sys.modules[__name__] = _moved
