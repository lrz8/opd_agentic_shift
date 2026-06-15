from __future__ import annotations

import numpy as np
from typing import Dict

from opd_agentic_shift.envs.troubleshooting_env import (
    ACTIONS,
    ACTION_TO_ID,
    FAULT_TO_TOOL,
    LOG_CLUE_TO_FAULT,
)


class RuleTeacher:
    """Oracle teacher used for expert data and OPD labels.

    The teacher can inspect the hidden fault through the environment snapshot.
    This mimics a strong teacher model/API that labels student-visited states.
    """

    def __init__(self, label_smoothing: float = 0.02):
        self.label_smoothing = label_smoothing

    def act_from_snapshot(self, snapshot: Dict) -> int:
        fault = snapshot["fault"]

        # First read logs when possible; this creates a multi-turn diagnostic path.
        if snapshot["log_status"] == "unknown" and not snapshot["logs_corrupted"]:
            return ACTION_TO_ID["check_service_log"]

        # If log clue is available, use it. If corrupted, fall back to oracle fault.
        if snapshot["log_status"] in LOG_CLUE_TO_FAULT:
            inferred_fault = LOG_CLUE_TO_FAULT[snapshot["log_status"]]
        else:
            inferred_fault = fault

        # Service crash can be diagnosed directly from log.
        if inferred_fault == "service_crash":
            return ACTION_TO_ID["submit_service_crash"]

        needed_tool = FAULT_TO_TOOL[inferred_fault]
        status_field = {
            "check_network": "network_status",
            "check_disk": "disk_status",
            "check_db": "db_status",
            "check_permission": "permission_status",
        }[needed_tool]

        if snapshot[status_field] == "unknown":
            return ACTION_TO_ID[needed_tool]

        return ACTION_TO_ID[f"submit_{inferred_fault}"]

    def probs_from_snapshot(self, snapshot: Dict) -> np.ndarray:
        n = len(ACTIONS)
        probs = np.full(n, self.label_smoothing / max(1, n - 1), dtype=np.float32)
        probs[self.act_from_snapshot(snapshot)] = 1.0 - self.label_smoothing
        probs /= probs.sum()
        return probs
