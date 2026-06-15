from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, Optional, Set, Tuple

import numpy as np

from opd_agentic_shift.envs.troubleshooting_env import (
    ACTION_TO_ID,
    FAULTS,
    ID_TO_ACTION,
    REPAIR_ACTIONS,
    TroubleshootingEnv,
)
from opd_agentic_shift.envs.encoder import encode_obs
from opd_agentic_shift.policies.mlp_policy import load_policy
from opd_agentic_shift.utils.io import read_jsonl

FAULT_TO_REPAIR = {
    "network_dns_error": "fix_dns",
    "disk_full": "clean_disk",
    "database_connection_error": "fix_db_config",
    "permission_denied": "fix_permission",
    "service_crash": "restart_service",
}


def load_state_keys(path: Optional[str]) -> Set[Tuple]:
    if not path:
        return set()
    return {tuple(r["state_key"]) for r in read_jsonl(path)}


def evaluate(
    policy_path: str,
    output: str,
    num_episodes: int = 500,
    seed: int = 10,
    max_steps: int = 6,
    expert_data: Optional[str] = None,
    opd_data: Optional[str] = None,
    deterministic: bool = True,
    temperature: float = 1.0,
    initial_wrong_repair_prob: float = 0.0,
):
    rng = random.Random(seed)
    env = TroubleshootingEnv(max_steps=max_steps, seed=seed)
    policy = load_policy(policy_path)
    expert_keys = load_state_keys(expert_data)
    opd_keys = load_state_keys(opd_data)

    episodes = []
    totals = []
    successes = []
    lengths = []
    off_exp_counts, off_opd_counts, state_counts = [], [], []

    for ep in range(num_episodes):
        fault = rng.choice(FAULTS)
        obs = env.reset(fault=fault)
        injected_actions = []
        injected_reward = 0.0
        if rng.random() < initial_wrong_repair_prob:
            wrong_repairs = [a for a in REPAIR_ACTIONS if a != FAULT_TO_REPAIR[fault]]
            action = rng.choice(wrong_repairs)
            obs, reward, done, _ = env.step(ACTION_TO_ID[action])
            injected_actions.append(action)
            injected_reward += reward

        done = False
        total_reward = injected_reward
        states = []
        actions = []
        rewards = []
        success = False
        for t in range(max_steps):
            key = tuple(env.get_state_key())
            states.append(list(key))
            obs_vec = encode_obs(obs)
            action_id, _ = policy.act(obs_vec, deterministic=deterministic, temperature=temperature)
            actions.append(ID_TO_ACTION[action_id])
            obs, reward, done, info = env.step(action_id)
            rewards.append(reward)
            total_reward += reward
            if info.get("success", False):
                success = True
            if done:
                break
        totals.append(total_reward)
        successes.append(float(success))
        lengths.append(len(actions))
        state_counts.append(len(states))
        off_exp_counts.append(sum(1 for s in states if tuple(s) not in expert_keys) if expert_keys else 0)
        off_opd_counts.append(sum(1 for s in states if tuple(s) not in opd_keys) if opd_keys else 0)
        episodes.append({
            "episode_id": ep,
            "fault": fault,
            "success": success,
            "total_reward": total_reward,
            "length": len(actions),
            "injected_actions": injected_actions,
            "actions": actions,
            "rewards": rewards,
            "states": states,
        })

    total_states = max(1, sum(state_counts))
    metrics = {
        "policy_path": policy_path,
        "num_episodes": num_episodes,
        "success_rate": float(np.mean(successes)),
        "avg_reward": float(np.mean(totals)),
        "avg_length": float(np.mean(lengths)),
        "off_support_vs_expert": float(sum(off_exp_counts) / total_states) if expert_keys else None,
        "off_support_vs_offline_opd": float(sum(off_opd_counts) / total_states) if opd_keys else None,
        "initial_wrong_repair_prob": initial_wrong_repair_prob,
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "episodes": episodes}, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"wrote eval details to {output}")
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--policy", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--num_episodes", type=int, default=500)
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--max_steps", type=int, default=6)
    p.add_argument("--expert_data", type=str, default="runs/data/expert.jsonl")
    p.add_argument("--opd_data", type=str, default="runs/data/offline_opd.jsonl")
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--initial_wrong_repair_prob", type=float, default=0.0)
    args = p.parse_args()
    evaluate(
        args.policy,
        args.output,
        args.num_episodes,
        args.seed,
        args.max_steps,
        args.expert_data,
        args.opd_data,
        not args.stochastic,
        args.temperature,
        args.initial_wrong_repair_prob,
    )
