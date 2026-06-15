from __future__ import annotations

from typing import Dict, List
import numpy as np

from .troubleshooting_env import STATUS_VALUES, LOG_VALUES


def one_hot(value: str, choices: List[str]) -> List[float]:
    return [1.0 if value == c else 0.0 for c in choices]


def encode_obs(obs: Dict) -> np.ndarray:
    """Encode observable state into a fixed vector.

    Hidden fault_type is intentionally excluded.
    """
    x: List[float] = []
    x += one_hot(obs["network_status"], STATUS_VALUES)
    x += one_hot(obs["disk_status"], STATUS_VALUES)
    x += one_hot(obs["db_status"], STATUS_VALUES)
    x += one_hot(obs["permission_status"], STATUS_VALUES)
    x += one_hot(obs["log_status"], LOG_VALUES)
    x += [
        float(obs["logs_corrupted"]),
        float(obs["network_misconfigured"]),
        float(obs["db_misconfigured"]),
        float(obs["permission_misconfigured"]),
        float(obs["step_count"]) / max(1.0, float(obs["max_steps"])),
    ]
    return np.asarray(x, dtype=np.float32)


def obs_dim() -> int:
    dummy = {
        "network_status": "unknown",
        "disk_status": "unknown",
        "db_status": "unknown",
        "permission_status": "unknown",
        "log_status": "unknown",
        "logs_corrupted": False,
        "network_misconfigured": False,
        "db_misconfigured": False,
        "permission_misconfigured": False,
        "step_count": 0,
        "max_steps": 6,
    }
    return int(encode_obs(dummy).shape[0])
