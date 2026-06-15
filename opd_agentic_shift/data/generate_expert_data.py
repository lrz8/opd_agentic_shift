from __future__ import annotations

import argparse
import random

from opd_agentic_shift.envs.troubleshooting_env import TroubleshootingEnv, ID_TO_ACTION, FAULTS
from opd_agentic_shift.envs.encoder import encode_obs
from opd_agentic_shift.teachers.rule_teacher import RuleTeacher
from opd_agentic_shift.utils.io import write_jsonl


def generate(num_episodes: int, output: str, seed: int = 0, max_steps: int = 6):
    rng = random.Random(seed)
    env = TroubleshootingEnv(max_steps=max_steps, seed=seed)
    teacher = RuleTeacher(label_smoothing=0.0)
    rows = []
    for ep in range(num_episodes):
        fault = rng.choice(FAULTS)
        obs = env.reset(fault=fault)
        done = False
        t = 0
        while not done:
            snapshot = env.get_snapshot()
            action_id = teacher.act_from_snapshot(snapshot)
            state_key = list(env.get_state_key())
            obs_vec = encode_obs(obs).tolist()
            next_obs, reward, done, info = env.step(action_id)
            rows.append({
                "episode_id": ep,
                "t": t,
                "fault": fault,
                "obs_vec": obs_vec,
                "state_key": state_key,
                "snapshot": snapshot,
                "action_id": action_id,
                "action": ID_TO_ACTION[action_id],
                "reward": reward,
                "done": done,
                "success": bool(info.get("success", False)),
            })
            obs = next_obs
            t += 1
            if t >= max_steps:
                break
    write_jsonl(output, rows)
    print(f"wrote {len(rows)} transitions to {output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--num_episodes", type=int, default=1000)
    p.add_argument("--output", type=str, default="runs/data/expert.jsonl")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_steps", type=int, default=6)
    args = p.parse_args()
    generate(args.num_episodes, args.output, args.seed, args.max_steps)
