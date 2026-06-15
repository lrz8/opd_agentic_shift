# Offline OPD under Agentic Shift - Report

## Task Choice

I chose case 2 from the online assessment: Offline OPD under Agentic Shift. The implementation is a CPU-only multi-turn troubleshooting task where an agent must call diagnostic tools, avoid harmful premature repairs, and submit a final root-cause answer.

The task directly targets a boundary of Lightning OPD: the paper argues offline OPD can reuse teacher log-probabilities over SFT rollouts when teacher consistency holds, while its reported setting is math/code generation rather than multi-turn tool-use. This toy task probes whether the same assumptions survive when early actions change later states. The comparison paper, Revealing the Power of Post-Training for Small Language Models via Knowledge Distillation, also uses curriculum SFT followed by offline on-policy knowledge distillation, but frames the pipeline as practical KD for small language models rather than Lightning OPD's teacher-consistency analysis.

Paper links:

- Lightning OPD: https://arxiv.org/pdf/2604.13010
- Revealing the Power of Post-Training for Small Language Models via Knowledge Distillation: https://arxiv.org/pdf/2509.26497

## Toy Task

The environment hides one of five faults: DNS failure, disk full, DB connection error, permission denial, or service crash. The policy observes only tool results and side-effect flags. It can inspect logs/network/disk/DB/permissions, attempt repairs, or submit a diagnosis. Wrong early repair actions corrupt logs or misconfigure services, so later states depend on the agent's own mistakes.

This gives the required agentic ingredients: multi-turn decisions, tool-style actions, branching error states, final-answer reward, and early mistakes that change future observations.

## Methods

- `SFT`: imitates expert trajectories from the rule teacher.
- `online_rl`: REINFORCE from environment reward, initialized from SFT.
- `offline_opd`: collects SFT rollouts once, precomputes teacher probabilities, and trains on that fixed dataset.
- `offline_opd_support`: the proposed patch, a support-aware OPD loss plus optional SFT-anchor KL regularization.
- `online_opd`: upper bound that queries the teacher on current student rollouts during training.

## Run Settings

Profile: `quick`

| setting | value |
|---|---:|
| expert_episodes | 300 |
| rollout_episodes | 300 |
| sft_epochs | 12 |
| offline_epochs | 15 |
| online_rl_episodes | 600 |
| online_opd_episodes | 300 |
| eval_episodes | 200 |

Command:

```bash
python run_all.py --profile quick
```

## In-Distribution Results

| method | success_rate | avg_reward | avg_length | off_support_vs_expert | off_support_vs_offline_opd |
|---|---:|---:|---:|---:|---:|
| sft | 1.000 | 0.861 | 2.785 | 0.000 | 0.000 |
| online_rl | 0.755 | 0.410 | 2.000 | 0.000 | 0.000 |
| offline_opd | 1.000 | 0.861 | 2.785 | 0.000 | 0.000 |
| offline_opd_support | 1.000 | 0.861 | 2.785 | 0.000 | 0.000 |
| online_opd | 1.000 | 0.861 | 2.785 | 0.000 | 0.000 |

## Agentic-Shift Stress Results

Each stress episode starts after one injected wrong repair action. This simulates the core failure mode of multi-turn agentic systems: an early bad tool call changes the later observable state.

| method | success_rate | avg_reward | avg_length | off_support_vs_expert | off_support_vs_offline_opd |
|---|---:|---:|---:|---:|---:|
| sft | 0.790 | 0.188 | 2.850 | 1.000 | 0.919 |
| online_rl | 0.580 | -0.183 | 1.865 | 1.000 | 0.949 |
| offline_opd | 0.870 | 0.357 | 3.055 | 1.000 | 0.925 |
| offline_opd_support | 0.870 | 0.357 | 3.055 | 1.000 | 0.925 |
| online_opd | 0.730 | 0.177 | 3.350 | 1.000 | 0.931 |

## In-Distribution Case Notes

### sft
- Success: fault `service_crash`, reward 0.90, actions ['check_service_log', 'submit_service_crash']
- Failure: no failed case found in this evaluation sample.

### online_rl
- Success: fault `service_crash`, reward 0.90, actions ['check_service_log', 'submit_service_crash']
- Failure: fault `disk_full`, reward -1.10, actions ['check_service_log', 'submit_network_dns_error']

### offline_opd
- Success: fault `service_crash`, reward 0.90, actions ['check_service_log', 'submit_service_crash']
- Failure: no failed case found in this evaluation sample.

### offline_opd_support
- Success: fault `service_crash`, reward 0.90, actions ['check_service_log', 'submit_service_crash']
- Failure: no failed case found in this evaluation sample.

### online_opd
- Success: fault `service_crash`, reward 0.90, actions ['check_service_log', 'submit_service_crash']
- Failure: no failed case found in this evaluation sample.

## Shifted Failure Notes

- `sft` shifted failure: fault `permission_denied`, injected ['clean_disk'], policy actions ['check_service_log', 'check_network', 'submit_network_dns_error'], reward -1.40
- `online_rl` shifted failure: fault `permission_denied`, injected ['clean_disk'], policy actions ['check_service_log', 'submit_network_dns_error'], reward -1.35
- `offline_opd` shifted failure: fault `database_connection_error`, injected ['clean_disk'], policy actions ['check_service_log', 'check_network', 'check_permission', 'submit_network_dns_error'], reward -1.45
- `offline_opd_support` shifted failure: fault `database_connection_error`, injected ['clean_disk'], policy actions ['check_service_log', 'check_network', 'check_permission', 'submit_network_dns_error'], reward -1.45
- `online_opd` shifted failure: fault `permission_denied`, injected ['clean_disk'], policy actions ['check_service_log', 'check_db', 'check_db', 'check_db', 'check_db'], reward -1.00

## Limitation And Patch

Naive offline OPD assumes the fixed SFT rollout dataset remains a good proxy for the states visited after OPD updates. In this toy task that assumption is fragile: bad repair actions create corrupted or misconfigured states that may be rare or absent in expert data, and a policy trained on fixed labels can still drift into those states at inference time.

The patch is `offline_opd_support`: each offline state receives a weight based on how often its observable state key appears in the rollout dataset, and an optional KL anchor keeps the policy near the SFT initialization. This is intentionally conservative: it downweights sparse branch states where the offline teacher labels may be less representative, while still learning from common on-policy states.

The ablation is the row comparison between `offline_opd` and `offline_opd_support`. When the support-aware row improves success/reward or lowers off-support ratio, it supports the patch. If it underperforms, that is also informative: overly conservative weighting can slow useful correction when the SFT rollout distribution already has enough coverage.

## Takeaways

Offline OPD is effective when the SFT rollout distribution has enough coverage and teacher consistency holds. It becomes brittle in multi-turn agentic settings when early actions alter the future state distribution, because a fixed offline dataset cannot query the teacher on newly induced branches. Online OPD is the cleaner upper bound because it labels the current student distribution; support-aware weighting is a cheap offline patch, but not a substitute for refreshed rollouts when drift is large.
