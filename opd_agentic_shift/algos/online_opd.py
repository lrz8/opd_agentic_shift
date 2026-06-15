from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from opd_agentic_shift.envs.troubleshooting_env import TroubleshootingEnv, FAULTS
from opd_agentic_shift.envs.encoder import encode_obs
from opd_agentic_shift.teachers.rule_teacher import RuleTeacher
from opd_agentic_shift.policies.mlp_policy import MLPPolicy, load_policy, save_policy


def train_online_opd(
    output: str,
    init_ckpt: str | None = None,
    episodes: int = 1200,
    lr: float = 2e-3,
    seed: int = 3,
    max_steps: int = 6,
    rollout_temperature: float = 1.0,
):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    env = TroubleshootingEnv(max_steps=max_steps, seed=seed)
    teacher = RuleTeacher(label_smoothing=0.02)
    policy = load_policy(init_ckpt) if init_ckpt else MLPPolicy()
    policy.train()
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)

    for ep in range(episodes):
        obs = env.reset(fault=rng.choice(FAULTS))
        xs, ts = [], []
        done = False
        for _ in range(max_steps):
            obs_vec = encode_obs(obs)
            xs.append(obs_vec)
            ts.append(teacher.probs_from_snapshot(env.get_snapshot()))
            action_id, _ = policy.act(obs_vec, deterministic=False, temperature=rollout_temperature)
            obs, _, done, _ = env.step(action_id)
            if done:
                break

        bx = torch.tensor(np.asarray(xs), dtype=torch.float32)
        bt = torch.tensor(np.asarray(ts), dtype=torch.float32)
        logits = policy(bx)
        loss = -(bt * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

        if ep % max(1, episodes // 10) == 0 or ep == episodes - 1:
            print(f"ep={ep:05d} online_opd_loss={float(loss.item()):.4f}")

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    save_policy(policy, output)
    print(f"saved online OPD policy to {output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=str, default="runs/ckpts/online_opd.pt")
    p.add_argument("--init_ckpt", type=str, default="runs/ckpts/sft.pt")
    p.add_argument("--episodes", type=int, default=1200)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--max_steps", type=int, default=6)
    p.add_argument("--rollout_temperature", type=float, default=1.0)
    args = p.parse_args()
    train_online_opd(**vars(args))
