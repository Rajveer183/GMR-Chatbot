"""Moved to gmr_agents/execution/tools/sql_executor.py -- this alias keeps old imports working."""

import sys

from gmr_agents.execution.tools import sql_executor as _moved

sys.modules[__name__] = _moved
