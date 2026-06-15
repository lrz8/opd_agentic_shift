from __future__ import annotations

import argparse

from opd_agentic_shift.teachers.rule_teacher import RuleTeacher
from opd_agentic_shift.envs.troubleshooting_env import ID_TO_ACTION
from opd_agentic_shift.utils.io import read_jsonl, write_jsonl


def build(rollouts: str, output: str, label_smoothing: float = 0.02):
    rows = read_jsonl(rollouts)
    teacher = RuleTeacher(label_smoothing=label_smoothing)
    out = []
    for r in rows:
        probs = teacher.probs_from_snapshot(r["snapshot"])
        teacher_action_id = int(probs.argmax())
        new_r = dict(r)
        new_r.update({
            "teacher_action_id": teacher_action_id,
            "teacher_action": ID_TO_ACTION[teacher_action_id],
            "teacher_probs": probs.tolist(),
        })
        out.append(new_r)
    write_jsonl(output, out)
    print(f"wrote {len(out)} OPD-labeled transitions to {output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts", type=str, default="runs/data/sft_rollouts.jsonl")
    p.add_argument("--output", type=str, default="runs/data/offline_opd.jsonl")
    p.add_argument("--label_smoothing", type=float, default=0.02)
    args = p.parse_args()
    build(args.rollouts, args.output, args.label_smoothing)
