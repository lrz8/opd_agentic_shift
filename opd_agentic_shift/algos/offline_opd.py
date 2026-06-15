from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from opd_agentic_shift.policies.mlp_policy import MLPPolicy, load_policy, save_policy
from opd_agentic_shift.utils.io import read_jsonl


def train_offline_opd(
    train_path: str,
    output: str,
    init_ckpt: str | None = None,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 2e-3,
    seed: int = 0,
    support_aware: bool = False,
    support_threshold: int = 5,
    anchor_coef: float = 0.0,
):
    torch.manual_seed(seed)
    rows = read_jsonl(train_path)
    x = torch.tensor([r["obs_vec"] for r in rows], dtype=torch.float32)
    teacher_probs = torch.tensor([r["teacher_probs"] for r in rows], dtype=torch.float32)

    if support_aware:
        counts = Counter(tuple(r["state_key"]) for r in rows)
        weights = [min(1.0, counts[tuple(r["state_key"])] / float(support_threshold)) ** 0.5 for r in rows]
    else:
        weights = [1.0 for _ in rows]
    w = torch.tensor(weights, dtype=torch.float32)

    loader = DataLoader(TensorDataset(x, teacher_probs, w), batch_size=batch_size, shuffle=True)

    policy = load_policy(init_ckpt) if init_ckpt else MLPPolicy()
    policy.train()
    anchor = load_policy(init_ckpt) if (init_ckpt and anchor_coef > 0) else None
    if anchor:
        anchor.eval()

    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    for epoch in range(epochs):
        total = 0.0
        n = 0
        for bx, bt, bw in loader:
            logits = policy(bx)
            logp = F.log_softmax(logits, dim=-1)
            # KL(teacher || student) up to a constant H(teacher).
            per_sample = -(bt * logp).sum(dim=-1)
            loss = (per_sample * bw).mean()
            if anchor is not None:
                with torch.no_grad():
                    anchor_probs = F.softmax(anchor(bx), dim=-1)
                loss = loss + anchor_coef * F.kl_div(logp, anchor_probs, reduction="batchmean")
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * bx.size(0)
            n += bx.size(0)
        if epoch % max(1, epochs // 5) == 0 or epoch == epochs - 1:
            tag = "support-aware" if support_aware else "plain"
            print(f"epoch={epoch:03d} loss={total/n:.4f} mode={tag}")

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    save_policy(policy, output)
    print(f"saved offline OPD policy to {output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=str, default="runs/data/offline_opd.jsonl")
    p.add_argument("--output", type=str, default="runs/ckpts/offline_opd.pt")
    p.add_argument("--init_ckpt", type=str, default="runs/ckpts/sft.pt")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--support_aware", action="store_true")
    p.add_argument("--support_threshold", type=int, default=5)
    p.add_argument("--anchor_coef", type=float, default=0.0)
    args = p.parse_args()
    train_offline_opd(
        train_path=args.train,
        output=args.output,
        init_ckpt=args.init_ckpt,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        support_aware=args.support_aware,
        support_threshold=args.support_threshold,
        anchor_coef=args.anchor_coef,
    )
