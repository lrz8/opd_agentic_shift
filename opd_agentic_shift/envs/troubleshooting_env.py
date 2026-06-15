from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import random

FAULTS = [
    "network_dns_error",
    "disk_full",
    "database_connection_error",
    "permission_denied",
    "service_crash",
]

CHECK_ACTIONS = [
    "check_service_log",
    "check_network",
    "check_disk",
    "check_db",
    "check_permission",
]

REPAIR_ACTIONS = [
    "restart_service",
    "clean_disk",
    "fix_dns",
    "fix_db_config",
    "fix_permission",
]

SUBMIT_ACTIONS = [f"submit_{f}" for f in FAULTS]

ACTIONS = CHECK_ACTIONS + REPAIR_ACTIONS + SUBMIT_ACTIONS
ACTION_TO_ID = {a: i for i, a in enumerate(ACTIONS)}
ID_TO_ACTION = {i: a for a, i in ACTION_TO_ID.items()}

FAULT_TO_TOOL = {
    "network_dns_error": "check_network",
    "disk_full": "check_disk",
    "database_connection_error": "check_db",
    "permission_denied": "check_permission",
    "service_crash": "check_service_log",
}

FAULT_TO_LOG_CLUE = {
    "network_dns_error": "network_timeout",
    "disk_full": "no_space_left",
    "database_connection_error": "db_refused",
    "permission_denied": "permission_error",
    "service_crash": "segfault",
}

LOG_CLUE_TO_FAULT = {v: k for k, v in FAULT_TO_LOG_CLUE.items()}

STATUS_VALUES = ["unknown", "ok", "bad", "corrupted", "misconfigured"]
LOG_VALUES = ["unknown", "network_timeout", "no_space_left", "db_refused", "permission_error", "segfault", "corrupted"]


@dataclass
class EnvSnapshot:
    fault: str
    step_count: int
    network_status: str
    disk_status: str
    db_status: str
    permission_status: str
    log_status: str
    logs_corrupted: bool
    network_misconfigured: bool
    db_misconfigured: bool
    permission_misconfigured: bool
    done: bool = False


class TroubleshootingEnv:
    """CPU-only multi-turn agentic toy task.

    The policy does not observe the hidden fault directly. It can call tools,
    take repair actions, and submit a final diagnosis. Wrong early repair actions
    create side effects, so the state distribution depends on the agent's own
    previous decisions.
    """

    def __init__(self, max_steps: int = 6, seed: int = 0):
        self.max_steps = max_steps
        self.rng = random.Random(seed)
        self.snapshot: Optional[EnvSnapshot] = None

    def reset(self, fault: Optional[str] = None) -> Dict:
        if fault is None:
            fault = self.rng.choice(FAULTS)
        self.snapshot = EnvSnapshot(
            fault=fault,
            step_count=0,
            network_status="unknown",
            disk_status="unknown",
            db_status="unknown",
            permission_status="unknown",
            log_status="unknown",
            logs_corrupted=False,
            network_misconfigured=False,
            db_misconfigured=False,
            permission_misconfigured=False,
            done=False,
        )
        return self.observe()

    def set_snapshot(self, data: Dict) -> Dict:
        self.snapshot = EnvSnapshot(**data)
        return self.observe()

    def get_snapshot(self) -> Dict:
        assert self.snapshot is not None
        return asdict(self.snapshot)

    def observe(self) -> Dict:
        s = self._s
        return {
            "network_status": s.network_status,
            "disk_status": s.disk_status,
            "db_status": s.db_status,
            "permission_status": s.permission_status,
            "log_status": s.log_status,
            "logs_corrupted": s.logs_corrupted,
            "network_misconfigured": s.network_misconfigured,
            "db_misconfigured": s.db_misconfigured,
            "permission_misconfigured": s.permission_misconfigured,
            "step_count": s.step_count,
            "max_steps": self.max_steps,
        }

    @property
    def _s(self) -> EnvSnapshot:
        if self.snapshot is None:
            raise RuntimeError("Call reset() before using the environment.")
        return self.snapshot

    def get_state_key(self) -> Tuple:
        """Observable state key used for support/off-support statistics."""
        obs = self.observe()
        return (
            obs["network_status"],
            obs["disk_status"],
            obs["db_status"],
            obs["permission_status"],
            obs["log_status"],
            int(obs["logs_corrupted"]),
            int(obs["network_misconfigured"]),
            int(obs["db_misconfigured"]),
            int(obs["permission_misconfigured"]),
            obs["step_count"],
        )

    def step(self, action_id: int):
        action = ID_TO_ACTION[int(action_id)]
        s = self._s
        if s.done:
            return self.observe(), 0.0, True, {"error": "step_after_done"}

        reward = -0.05
        info = {"action": action, "fault": s.fault, "success": False, "early_error": False}

        if action in CHECK_ACTIONS:
            self._apply_check(action)
        elif action in REPAIR_ACTIONS:
            reward += self._apply_repair(action)
            info["early_error"] = True
        elif action in SUBMIT_ACTIONS:
            predicted = action.replace("submit_", "")
            s.done = True
            if predicted == s.fault:
                reward += 1.0
                info["success"] = True
            else:
                reward -= 1.0
        else:
            raise ValueError(f"Unknown action: {action}")

        s.step_count += 1
        if s.step_count >= self.max_steps and not s.done:
            s.done = True
            reward -= 0.5

        return self.observe(), reward, s.done, info

    def _apply_check(self, action: str) -> None:
        s = self._s
        if action == "check_service_log":
            if s.logs_corrupted:
                s.log_status = "corrupted"
            else:
                s.log_status = FAULT_TO_LOG_CLUE[s.fault]
            return

        if action == "check_network":
            if s.network_misconfigured:
                s.network_status = "misconfigured"
            elif s.fault == "network_dns_error":
                s.network_status = "bad"
            else:
                s.network_status = "ok"
            return

        if action == "check_disk":
            s.disk_status = "bad" if s.fault == "disk_full" else "ok"
            return

        if action == "check_db":
            if s.db_misconfigured:
                s.db_status = "misconfigured"
            elif s.fault == "database_connection_error":
                s.db_status = "bad"
            else:
                s.db_status = "ok"
            return

        if action == "check_permission":
            if s.permission_misconfigured:
                s.permission_status = "misconfigured"
            elif s.fault == "permission_denied":
                s.permission_status = "bad"
            else:
                s.permission_status = "ok"
            return

    def _apply_repair(self, action: str) -> float:
        """Repair actions create side effects if called before diagnosis.

        They are intentionally risky; this makes early mistakes influence future
        observations and creates off-support states.
        """
        s = self._s
        penalty = -0.20
        if action == "restart_service":
            s.logs_corrupted = True
            s.log_status = "corrupted"
        elif action == "clean_disk" and s.fault != "disk_full":
            s.logs_corrupted = True
        elif action == "fix_dns" and s.fault != "network_dns_error":
            s.network_misconfigured = True
        elif action == "fix_db_config" and s.fault != "database_connection_error":
            s.db_misconfigured = True
        elif action == "fix_permission" and s.fault != "permission_denied":
            s.permission_misconfigured = True
        else:
            # Correct repair is mildly useful but diagnosis still needs submit.
            penalty = 0.0
        return penalty
