from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from opd_agentic_shift.policies.mlp_policy import MLPPolicy, save_policy
from opd_agentic_shift.utils.io import read_jsonl


def train_sft(train_path: str, output: str, epochs: int = 25, batch_size: int = 128, lr: float = 3e-3, seed: int = 0):
    torch.manual_seed(seed)
    rows = read_jsonl(train_path)
    x = torch.tensor([r["obs_vec"] for r in rows], dtype=torch.float32)
    y = torch.tensor([r["action_id"] for r in rows], dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)

    policy = MLPPolicy()
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    for epoch in range(epochs):
        total = 0.0
        correct = 0
        n = 0
        for bx, by in loader:
            logits = policy(bx)
            loss = F.cross_entropy(logits, by)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * bx.size(0)
            correct += int((logits.argmax(-1) == by).sum().item())
            n += bx.size(0)
        if epoch % max(1, epochs // 5) == 0 or epoch == epochs - 1:
            print(f"epoch={epoch:03d} loss={total/n:.4f} acc={correct/n:.3f}")

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    save_policy(policy, output)
    print(f"saved SFT policy to {output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=str, default="runs/data/expert.jsonl")
    p.add_argument("--output", type=str, default="runs/ckpts/sft.pt")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    train_sft(args.train, args.output, args.epochs, args.batch_size, args.lr, args.seed)
