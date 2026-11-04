"""Moved to gmr_agents/ops/metrics.py -- this alias keeps old imports and
`python -m gmr_agents.metrics` working."""

import runpy
import sys

if __name__ == "__main__":
    runpy.run_module("gmr_agents.ops.metrics", run_name="__main__")
else:
    from gmr_agents.ops import metrics as _moved

    sys.modules[__name__] = _moved
