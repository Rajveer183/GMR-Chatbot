"""Moved to gmr_agents/retrieval/knowledge_store.py -- this alias keeps old imports working."""

import sys

from gmr_agents.retrieval import knowledge_store as _moved

sys.modules[__name__] = _moved
