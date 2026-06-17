from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"
LATEX_RESULTS = ROOT / "latex_repo" / "results"

PROFILES = {
    "quick": {
        "expert_episodes": 300,
        "rollout_episodes": 300,
        "sft_epochs": 12,
        "offline_epochs": 15,
        "online_rl_episodes": 600,
        "online_opd_episodes": 300,
        "eval_episodes": 200,
    },
    "full": {
        "expert_episodes": 1000,
        "rollout_episodes": 1000,
        "sft_epochs": 25,
        "offline_epochs": 30,
        "online_rl_episodes": 2000,
        "online_opd_episodes": 800,
        "eval_episodes": 500,
    },
}


def sh(cmd):
    print("\n$", " ".join(cmd))
    sys.stdout.flush()
    subprocess.check_call(cmd, cwd=ROOT)


def load_eval_case(path: Path, want_success: bool):
    with open(path, "r", encoding="utf-8") as f:
        episodes = json.load(f)["episodes"]
    for ep in episodes:
        if bool(ep["success"]) == want_success:
            return ep
    return None


def fmt_metric(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def result_table(metrics):
    lines = [
        "| method | success_rate | avg_reward | avg_length | off_support_vs_expert | off_support_vs_offline_opd |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in metrics.items():
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    fmt_metric(row["success_rate"]),
                    fmt_metric(row["avg_reward"]),
                    fmt_metric(row["avg_length"]),
                    fmt_metric(row["off_support_vs_expert"]),
                    fmt_metric(row["off_support_vs_offline_opd"]),
                ]
            )
            + " |"
        )
    return lines


def write_report(metrics, shifted_metrics, eval_paths, shifted_eval_paths, profile: str, cfg) -> None:
    lines = [
        "# Offline OPD under Agentic Shift - Report",
        "",
        "## Task Choice",
        "",
        "I chose case 2 from the online assessment: Offline OPD under Agentic Shift. The implementation is a CPU-only multi-turn troubleshooting task where an agent must call diagnostic tools, avoid harmful premature repairs, and submit a final root-cause answer.",
        "",
        "The task directly targets a boundary of Lightning OPD: the paper argues offline OPD can reuse teacher log-probabilities over SFT rollouts when teacher consistency holds, while its reported setting is math/code generation rather than multi-turn tool-use. This toy task probes whether the same assumptions survive when early actions change later states. The comparison paper, Revealing the Power of Post-Training for Small Language Models via Knowledge Distillation, also uses curriculum SFT followed by offline on-policy knowledge distillation, but frames the pipeline as practical KD for small language models rather than Lightning OPD's teacher-consistency analysis.",
        "",
        "Paper links:",
        "",
        "- Lightning OPD: https://arxiv.org/pdf/2604.13010",
        "- Revealing the Power of Post-Training for Small Language Models via Knowledge Distillation: https://arxiv.org/pdf/2509.26497",
        "",
        "## Toy Task",
        "",
        "The environment hides one of five faults: DNS failure, disk full, DB connection error, permission denial, or service crash. The policy observes only tool results and side-effect flags. It can inspect logs/network/disk/DB/permissions, attempt repairs, or submit a diagnosis. Wrong early repair actions corrupt logs or misconfigure services, so later states depend on the agent's own mistakes.",
        "",
        "This gives the required agentic ingredients: multi-turn decisions, tool-style actions, branching error states, final-answer reward, and early mistakes that change future observations.",
        "",
        "## Methods",
        "",
        "- `SFT`: imitates expert trajectories from the rule teacher.",
        "- `online_rl`: REINFORCE from environment reward, initialized from SFT.",
        "- `offline_opd`: collects SFT rollouts once, precomputes teacher probabilities, and trains on that fixed dataset.",
        "- `offline_opd_support`: the proposed patch, a support-aware OPD loss plus optional SFT-anchor KL regularization.",
        "- `online_opd`: upper bound that queries the teacher on current student rollouts during training.",
        "",
        "## Run Settings",
        "",
        f"Profile: `{profile}`",
        "",
        "| setting | value |",
        "|---|---:|",
    ]
    for key, value in cfg.items():
        lines.append(f"| {key} | {value} |")

    lines.extend([
        "",
        "Command:",
        "",
        f"```bash\npython run_all.py --profile {profile}\n```",
        "",
        "## In-Distribution Results",
        "",
    ])
    lines.extend(result_table(metrics))

    lines.extend([
        "",
        "## Agentic-Shift Stress Results",
        "",
        "Each stress episode starts after one injected wrong repair action. This simulates the core failure mode of multi-turn agentic systems: an early bad tool call changes the later observable state.",
        "",
    ])
    lines.extend(result_table(shifted_metrics))

    lines.extend([
        "",
        "## In-Distribution Case Notes",
        "",
    ])
    for name, path in eval_paths.items():
        success = load_eval_case(path, True)
        failure = load_eval_case(path, False)
        lines.append(f"### {name}")
        if success:
            lines.append(
                f"- Success: fault `{success['fault']}`, reward {success['total_reward']:.2f}, actions {success['actions']}"
            )
        else:
            lines.append("- Success: no successful case found in this evaluation sample.")
        if failure:
            lines.append(
                f"- Failure: fault `{failure['fault']}`, reward {failure['total_reward']:.2f}, actions {failure['actions']}"
            )
        else:
            lines.append("- Failure: no failed case found in this evaluation sample.")
        lines.append("")

    lines.extend([
        "## Shifted Failure Notes",
        "",
    ])
    for name, path in shifted_eval_paths.items():
        failure = load_eval_case(path, False)
        if failure:
            lines.append(
                f"- `{name}` shifted failure: fault `{failure['fault']}`, injected {failure['injected_actions']}, policy actions {failure['actions']}, reward {failure['total_reward']:.2f}"
            )
        else:
            lines.append(f"- `{name}` shifted failure: no failed case found in this evaluation sample.")

    lines.extend([
        "",
        "## Limitation And Patch",
        "",
        "Naive offline OPD assumes the fixed SFT rollout dataset remains a good proxy for the states visited after OPD updates. In this toy task that assumption is fragile: bad repair actions create corrupted or misconfigured states that may be rare or absent in expert data, and a policy trained on fixed labels can still drift into those states at inference time.",
        "",
        "The patch is `offline_opd_support`: each offline state receives a weight based on how often its observable state key appears in the rollout dataset, and an optional KL anchor keeps the policy near the SFT initialization. This is intentionally conservative: it downweights sparse branch states where the offline teacher labels may be less representative, while still learning from common on-policy states.",
        "",
        "The ablation is the row comparison between `offline_opd` and `offline_opd_support`. When the support-aware row improves success/reward or lowers off-support ratio, it supports the patch. If it underperforms, that is also informative: overly conservative weighting can slow useful correction when the SFT rollout distribution already has enough coverage.",
        "",
        "## Takeaways",
        "",
        "Offline OPD is effective when the SFT rollout distribution has enough coverage and teacher consistency holds. It becomes brittle in multi-turn agentic settings when early actions alter the future state distribution, because a fixed offline dataset cannot query the teacher on newly induced branches. Online OPD is the cleaner upper bound because it labels the current student distribution; support-aware weighting is a cheap offline patch, but not a substitute for refreshed rollouts when drift is large.",
    ])

    with open(ROOT / "REPORT.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_result_bundle(base: Path, metrics, shifted_metrics) -> None:
    (base / "eval").mkdir(parents=True, exist_ok=True)
    (base / "eval_shift").mkdir(parents=True, exist_ok=True)
    (base / "tables").mkdir(parents=True, exist_ok=True)

    with open(base / "eval" / "summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(base / "eval_shift" / "summary.json", "w", encoding="utf-8") as f:
        json.dump(shifted_metrics, f, indent=2, ensure_ascii=False)

    readme = (
        "# 实验结果归档说明\n\n"
        "本目录只保留报告需要引用的精简结果。完整训练数据、checkpoint 和逐 episode 评估详情保留在项目根目录的 `runs/` 下。\n\n"
        "## 文件说明\n\n"
        "- `eval/summary.json`：正常分布评估的汇总指标。\n"
        "- `eval_shift/summary.json`：注入一次早期错误 repair action 后的 agentic-shift 压力测试汇总指标。\n"
        "- `tables/summary_tables_cn.csv`：正常评估与 shifted 评估的主结果表格汇总。\n"
    )
    with open(base / "README_cn.md", "w", encoding="utf-8") as f:
        f.write(readme)

    columns = [
        "setting",
        "method",
        "success_rate",
        "avg_reward",
        "avg_length",
        "off_support_vs_expert",
        "off_support_vs_offline_opd",
    ]
    with open(base / "tables" / "summary_tables_cn.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for setting, table in [("in_distribution", metrics), ("agentic_shift", shifted_metrics)]:
            for method, row in table.items():
                writer.writerow({
                    "setting": setting,
                    "method": method,
                    "success_rate": f"{row['success_rate']:.3f}",
                    "avg_reward": f"{row['avg_reward']:.3f}",
                    "avg_length": f"{row['avg_length']:.3f}",
                    "off_support_vs_expert": f"{row['off_support_vs_expert']:.3f}",
                    "off_support_vs_offline_opd": f"{row['off_support_vs_offline_opd']:.3f}",
                })


def write_submission_results(metrics, shifted_metrics) -> None:
    write_result_bundle(RESULTS, metrics, shifted_metrics)
    write_result_bundle(LATEX_RESULTS, metrics, shifted_metrics)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(PROFILES), default="quick")
    args = parser.parse_args()
    cfg = PROFILES[args.profile]

    py = sys.executable
    sh([py, "-m", "opd_agentic_shift.data.generate_expert_data", "--num_episodes", str(cfg["expert_episodes"]), "--output", "runs/data/expert.jsonl"])
    sh([py, "-m", "opd_agentic_shift.algos.sft", "--train", "runs/data/expert.jsonl", "--epochs", str(cfg["sft_epochs"]), "--output", "runs/ckpts/sft.pt"])
    sh([py, "-m", "opd_agentic_shift.data.collect_rollouts", "--policy", "runs/ckpts/sft.pt", "--num_episodes", str(cfg["rollout_episodes"]), "--temperature", "1.4", "--output", "runs/data/sft_rollouts.jsonl"])
    sh([py, "-m", "opd_agentic_shift.data.build_offline_opd_data", "--rollouts", "runs/data/sft_rollouts.jsonl", "--output", "runs/data/offline_opd.jsonl"])
    sh([py, "-m", "opd_agentic_shift.algos.offline_opd", "--train", "runs/data/offline_opd.jsonl", "--init_ckpt", "runs/ckpts/sft.pt", "--epochs", str(cfg["offline_epochs"]), "--output", "runs/ckpts/offline_opd.pt"])
    sh([py, "-m", "opd_agentic_shift.algos.offline_opd", "--train", "runs/data/offline_opd.jsonl", "--init_ckpt", "runs/ckpts/sft.pt", "--epochs", str(cfg["offline_epochs"]), "--support_aware", "--support_threshold", "5", "--anchor_coef", "0.01", "--output", "runs/ckpts/offline_opd_support.pt"])
    sh([py, "-m", "opd_agentic_shift.algos.online_rl", "--init_ckpt", "runs/ckpts/sft.pt", "--episodes", str(cfg["online_rl_episodes"]), "--output", "runs/ckpts/online_rl.pt"])
    sh([py, "-m", "opd_agentic_shift.algos.online_opd", "--init_ckpt", "runs/ckpts/sft.pt", "--episodes", str(cfg["online_opd_episodes"]), "--output", "runs/ckpts/online_opd.pt"])

    policies = {
        "sft": "runs/ckpts/sft.pt",
        "online_rl": "runs/ckpts/online_rl.pt",
        "offline_opd": "runs/ckpts/offline_opd.pt",
        "offline_opd_support": "runs/ckpts/offline_opd_support.pt",
        "online_opd": "runs/ckpts/online_opd.pt",
    }
    metrics = {}
    eval_paths = {}
    for name, ckpt in policies.items():
        out = f"runs/eval/{name}.json"
        sh([py, "-m", "opd_agentic_shift.eval.evaluate", "--policy", ckpt, "--output", out, "--num_episodes", str(cfg["eval_episodes"]), "--expert_data", "runs/data/expert.jsonl", "--opd_data", "runs/data/offline_opd.jsonl"])
        eval_paths[name] = ROOT / out
        with open(ROOT / out, "r", encoding="utf-8") as f:
            metrics[name] = json.load(f)["metrics"]

    shifted_metrics = {}
    shifted_eval_paths = {}
    for name, ckpt in policies.items():
        out = f"runs/eval_shift/{name}.json"
        sh([
            py,
            "-m",
            "opd_agentic_shift.eval.evaluate",
            "--policy",
            ckpt,
            "--output",
            out,
            "--num_episodes",
            str(cfg["eval_episodes"]),
            "--expert_data",
            "runs/data/expert.jsonl",
            "--opd_data",
            "runs/data/offline_opd.jsonl",
            "--initial_wrong_repair_prob",
            "1.0",
        ])
        shifted_eval_paths[name] = ROOT / out
        with open(ROOT / out, "r", encoding="utf-8") as f:
            shifted_metrics[name] = json.load(f)["metrics"]

    (RUNS / "eval").mkdir(parents=True, exist_ok=True)
    with open(RUNS / "eval" / "summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    (RUNS / "eval_shift").mkdir(parents=True, exist_ok=True)
    with open(RUNS / "eval_shift" / "summary.json", "w", encoding="utf-8") as f:
        json.dump(shifted_metrics, f, indent=2, ensure_ascii=False)
    write_submission_results(metrics, shifted_metrics)
    write_report(metrics, shifted_metrics, eval_paths, shifted_eval_paths, args.profile, cfg)
    print("\nSUMMARY")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print("\nSHIFTED SUMMARY")
    print(json.dumps(shifted_metrics, indent=2, ensure_ascii=False))
    print("\nWROTE results/")
    print("\nWROTE REPORT.md")


if __name__ == "__main__":
    main()
