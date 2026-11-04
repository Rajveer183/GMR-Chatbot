"""Moved to gmr_agents/retrieval/topk.py -- this alias keeps old imports working."""

import sys

from gmr_agents.retrieval import topk as _moved

sys.modules[__name__] = _moved
