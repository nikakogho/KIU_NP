from __future__ import annotations
from typing import Any, Dict
import numpy as np

# Keys we consider "the run artifact"
RUN_KEYS = (
    "times",
    "X",
    "V",
    "targets_assigned",
    "targets_raw",
    "x0",
    "v0",
    "perm",
    "metrics",
)

def save_run_npz(path: str, run: Dict[str, Any]) -> None:
    payload = {}
    for k in RUN_KEYS:
        if k in run:
            payload[k] = run[k]

    # store metrics dict in an object array so np.savez can preserve it
    if "metrics" in payload and isinstance(payload["metrics"], dict):
        payload["metrics"] = np.array([payload["metrics"]], dtype=object)

    np.savez_compressed(path, **payload)
