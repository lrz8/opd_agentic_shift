from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from opd_agentic_shift.envs.troubleshooting_env import TroubleshootingEnv, FAULTS
from opd_agentic_shift.envs.encoder import encode_obs
from opd_agentic_shift.policies.mlp_policy import MLPPolicy, load_policy, save_policy


def train_online_rl(
    output: str,
    init_ckpt: str | None = None,
    episodes: int = 4000,
    lr: float = 1e-3,
    gamma: float = 0.98,
    entropy_coef: float = 0.01,
    seed: int = 2,
    max_steps: int = 6,
):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    env = TroubleshootingEnv(max_steps=max_steps, seed=seed)
    policy = load_policy(init_ckpt) if init_ckpt else MLPPolicy()
    policy.train()
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)

    ema_reward = None
    for ep in range(episodes):
        obs = env.reset(fault=rng.choice(FAULTS))
        logps, rewards, entropies = [], [], []
        done = False
        for _ in range(max_steps):
            x = torch.tensor(encode_obs(obs), dtype=torch.float32).unsqueeze(0)
            logits = policy(x).squeeze(0)
            dist = Categorical(logits=logits)
            action = dist.sample()
            obs, reward, done, _ = env.step(int(action.item()))
            logps.append(dist.log_prob(action))
            entropies.append(dist.entropy())
            rewards.append(float(reward))
            if done:
                break
        returns = []
        g = 0.0
        for r in reversed(rewards):
            g = r + gamma * g
            returns.append(g)
        returns = list(reversed(returns))
        ret_t = torch.tensor(returns, dtype=torch.float32)
        if len(ret_t) > 1:
            ret_t = (ret_t - ret_t.mean()) / (ret_t.std() + 1e-6)
        loss = -(torch.stack(logps) * ret_t).sum() - entropy_coef * torch.stack(entropies).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()

        ep_reward = sum(rewards)
        ema_reward = ep_reward if ema_reward is None else 0.98 * ema_reward + 0.02 * ep_reward
        if ep % max(1, episodes // 10) == 0 or ep == episodes - 1:
            print(f"ep={ep:05d} reward={ep_reward:.3f} ema_reward={ema_reward:.3f}")

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    save_policy(policy, output)
    print(f"saved online RL policy to {output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=str, default="runs/ckpts/online_rl.pt")
    p.add_argument("--init_ckpt", type=str, default="runs/ckpts/sft.pt")
    p.add_argument("--episodes", type=int, default=4000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.98)
    p.add_argument("--entropy_coef", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--max_steps", type=int, default=6)
    args = p.parse_args()
    train_online_rl(**vars(args))
