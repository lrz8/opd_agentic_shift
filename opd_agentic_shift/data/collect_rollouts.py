from __future__ import annotations

import argparse
import random

from opd_agentic_shift.envs.troubleshooting_env import TroubleshootingEnv, ID_TO_ACTION, FAULTS
from opd_agentic_shift.envs.encoder import encode_obs
from opd_agentic_shift.policies.mlp_policy import load_policy
from opd_agentic_shift.utils.io import write_jsonl


def collect(policy_path: str, output: str, num_episodes: int = 1000, seed: int = 1, max_steps: int = 6, temperature: float = 1.0, deterministic: bool = False):
    rng = random.Random(seed)
    env = TroubleshootingEnv(max_steps=max_steps, seed=seed)
    policy = load_policy(policy_path)
    rows = []
    for ep in range(num_episodes):
        fault = rng.choice(FAULTS)
        obs = env.reset(fault=fault)
        done = False
        total_reward = 0.0
        t = 0
        while not done:
            obs_vec = encode_obs(obs)
            snapshot = env.get_snapshot()
            state_key = list(env.get_state_key())
            action_id, logprob = policy.act(obs_vec, deterministic=deterministic, temperature=temperature)
            next_obs, reward, done, info = env.step(action_id)
            total_reward += reward
            rows.append({
                "episode_id": ep,
                "t": t,
                "fault": fault,
                "obs_vec": obs_vec.tolist(),
                "state_key": state_key,
                "snapshot": snapshot,
                "policy_action_id": action_id,
                "policy_action": ID_TO_ACTION[action_id],
                "policy_logprob": logprob,
                "reward": reward,
                "done": done,
                "success": bool(info.get("success", False)),
                "episode_reward_so_far": total_reward,
            })
            obs = next_obs
            t += 1
            if t >= max_steps:
                break
    write_jsonl(output, rows)
    print(f"wrote {len(rows)} rollout transitions to {output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--policy", type=str, default="runs/ckpts/sft.pt")
    p.add_argument("--output", type=str, default="runs/data/sft_rollouts.jsonl")
    p.add_argument("--num_episodes", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=6)
    p.add_argument("--temperature", type=float, default=1.2)
    p.add_argument("--deterministic", action="store_true")
    args = p.parse_args()
    collect(args.policy, args.output, args.num_episodes, args.seed, args.max_steps, args.temperature, args.deterministic)
