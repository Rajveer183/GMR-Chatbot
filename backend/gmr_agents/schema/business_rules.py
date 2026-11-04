"""Moved to gmr_agents/catalog/business_rules.py -- this alias keeps old imports working."""

import sys

from gmr_agents.catalog import business_rules as _moved

sys.modules[__name__] = _moved
