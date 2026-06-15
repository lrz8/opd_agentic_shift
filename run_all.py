from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"


def sh(cmd):
    print("\n$", " ".join(cmd))
    subprocess.check_call(cmd, cwd=ROOT)


def main():
    py = sys.executable
    sh([py, "-m", "opd_agentic_shift.data.generate_expert_data", "--num_episodes", "1000", "--output", "runs/data/expert.jsonl"])
    sh([py, "-m", "opd_agentic_shift.algos.sft", "--train", "runs/data/expert.jsonl", "--output", "runs/ckpts/sft.pt"])
    sh([py, "-m", "opd_agentic_shift.data.collect_rollouts", "--policy", "runs/ckpts/sft.pt", "--num_episodes", "1000", "--temperature", "1.4", "--output", "runs/data/sft_rollouts.jsonl"])
    sh([py, "-m", "opd_agentic_shift.data.build_offline_opd_data", "--rollouts", "runs/data/sft_rollouts.jsonl", "--output", "runs/data/offline_opd.jsonl"])
    sh([py, "-m", "opd_agentic_shift.algos.offline_opd", "--train", "runs/data/offline_opd.jsonl", "--init_ckpt", "runs/ckpts/sft.pt", "--output", "runs/ckpts/offline_opd.pt"])
    sh([py, "-m", "opd_agentic_shift.algos.offline_opd", "--train", "runs/data/offline_opd.jsonl", "--init_ckpt", "runs/ckpts/sft.pt", "--support_aware", "--support_threshold", "5", "--anchor_coef", "0.01", "--output", "runs/ckpts/offline_opd_support.pt"])
    sh([py, "-m", "opd_agentic_shift.algos.online_rl", "--init_ckpt", "runs/ckpts/sft.pt", "--episodes", "2000", "--output", "runs/ckpts/online_rl.pt"])
    sh([py, "-m", "opd_agentic_shift.algos.online_opd", "--init_ckpt", "runs/ckpts/sft.pt", "--episodes", "800", "--output", "runs/ckpts/online_opd.pt"])

    policies = {
        "sft": "runs/ckpts/sft.pt",
        "online_rl": "runs/ckpts/online_rl.pt",
        "offline_opd": "runs/ckpts/offline_opd.pt",
        "offline_opd_support": "runs/ckpts/offline_opd_support.pt",
        "online_opd": "runs/ckpts/online_opd.pt",
    }
    metrics = {}
    for name, ckpt in policies.items():
        out = f"runs/eval/{name}.json"
        sh([py, "-m", "opd_agentic_shift.eval.evaluate", "--policy", ckpt, "--output", out, "--num_episodes", "500", "--expert_data", "runs/data/expert.jsonl", "--opd_data", "runs/data/offline_opd.jsonl"])
        with open(ROOT / out, "r", encoding="utf-8") as f:
            metrics[name] = json.load(f)["metrics"]

    (RUNS / "eval").mkdir(parents=True, exist_ok=True)
    with open(RUNS / "eval" / "summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print("\nSUMMARY")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
