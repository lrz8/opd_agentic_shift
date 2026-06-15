# Offline OPD under Agentic Shift - CPU-only Starter

This is a lightweight toy framework for the online assessment topic: Offline OPD under Agentic Shift.

## Goal

Study whether offline on-policy distillation works in a multi-turn agentic task where early wrong actions change later states.

## Toy task

A troubleshooting agent must diagnose one hidden fault by calling tools and then submitting a root cause. Wrong early repair actions corrupt logs or misconfigure components, creating off-support states.

## Methods

- SFT: imitate expert trajectories from a rule teacher.
- Online RL: REINFORCE baseline on environment reward.
- Offline OPD: collect SFT rollouts, precompute teacher probabilities, and train offline.
- Online OPD: label current student rollouts online, used as an upper bound.
- Patch: support-aware Offline OPD with optional anchor regularization.

## Quick start

```bash
pip install -r requirements.txt
python run_all.py --profile quick
```

This runs the full toy pipeline on CPU with reduced episode counts, writes
metrics to `runs/eval/summary.json`, and generates `REPORT.md` for handoff.
For a larger rerun that matches the original demo scale:

```bash
python run_all.py --profile full
```

## Manual pipeline

```bash
python -m opd_agentic_shift.data.generate_expert_data --num_episodes 1000 --output runs/data/expert.jsonl
python -m opd_agentic_shift.algos.sft --train runs/data/expert.jsonl --output runs/ckpts/sft.pt
python -m opd_agentic_shift.data.collect_rollouts --policy runs/ckpts/sft.pt --num_episodes 1000 --temperature 1.4 --output runs/data/sft_rollouts.jsonl
python -m opd_agentic_shift.data.build_offline_opd_data --rollouts runs/data/sft_rollouts.jsonl --output runs/data/offline_opd.jsonl
python -m opd_agentic_shift.algos.offline_opd --train runs/data/offline_opd.jsonl --init_ckpt runs/ckpts/sft.pt --output runs/ckpts/offline_opd.pt
python -m opd_agentic_shift.algos.offline_opd --train runs/data/offline_opd.jsonl --init_ckpt runs/ckpts/sft.pt --support_aware --anchor_coef 0.01 --output runs/ckpts/offline_opd_support.pt
python -m opd_agentic_shift.algos.online_rl --init_ckpt runs/ckpts/sft.pt --episodes 2000 --output runs/ckpts/online_rl.pt
python -m opd_agentic_shift.algos.online_opd --init_ckpt runs/ckpts/sft.pt --episodes 800 --output runs/ckpts/online_opd.pt
```

Evaluate one policy:

```bash
python -m opd_agentic_shift.eval.evaluate \
  --policy runs/ckpts/offline_opd_support.pt \
  --output runs/eval/offline_opd_support.json \
  --expert_data runs/data/expert.jsonl \
  --opd_data runs/data/offline_opd.jsonl

python -m opd_agentic_shift.eval.case_analysis --eval_json runs/eval/offline_opd_support.json
```

## Key metrics

- success_rate
- avg_reward
- avg_length
- off_support_vs_expert
- off_support_vs_offline_opd

## Main files

```text
opd_agentic_shift/envs/troubleshooting_env.py   # toy environment
opd_agentic_shift/envs/encoder.py               # symbolic observation encoder
opd_agentic_shift/teachers/rule_teacher.py      # oracle teacher/expert
opd_agentic_shift/policies/mlp_policy.py        # CPU MLP policy
opd_agentic_shift/data/generate_expert_data.py  # expert trajectories
opd_agentic_shift/data/collect_rollouts.py      # SFT/student rollouts
opd_agentic_shift/data/build_offline_opd_data.py# teacher pre-labeling
opd_agentic_shift/algos/sft.py                  # SFT baseline
opd_agentic_shift/algos/offline_opd.py          # offline OPD + support-aware patch
opd_agentic_shift/algos/online_rl.py            # REINFORCE baseline
opd_agentic_shift/algos/online_opd.py           # online OPD upper bound
opd_agentic_shift/eval/evaluate.py              # metrics
opd_agentic_shift/eval/case_analysis.py         # success/failure case printing
```
