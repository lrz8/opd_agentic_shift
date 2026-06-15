from __future__ import annotations

import argparse
import json


def summarize(eval_json: str, max_cases: int = 5):
    with open(eval_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    episodes = data["episodes"]
    success = [e for e in episodes if e["success"]]
    fail = [e for e in episodes if not e["success"]]
    print("METRICS")
    print(json.dumps(data["metrics"], indent=2, ensure_ascii=False))
    print("\nSUCCESS CASES")
    for e in success[:max_cases]:
        print(f"fault={e['fault']} reward={e['total_reward']:.2f} len={e['length']} actions={e['actions']}")
    print("\nFAILURE CASES")
    for e in fail[:max_cases]:
        print(f"fault={e['fault']} reward={e['total_reward']:.2f} len={e['length']} actions={e['actions']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--eval_json", type=str, required=True)
    p.add_argument("--max_cases", type=int, default=5)
    args = p.parse_args()
    summarize(args.eval_json, args.max_cases)
