"""Moved to gmr_agents/presentation/dashboard_charts.py -- this alias keeps old imports working."""

import sys

from gmr_agents.presentation import dashboard_charts as _moved

sys.modules[__name__] = _moved
