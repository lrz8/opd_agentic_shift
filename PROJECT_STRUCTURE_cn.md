# 项目结构说明

本项目用于完成题目二：Offline OPD under Agentic Shift。
代码主体实现一个 CPU-only 的多轮故障排查 toy task，并比较 SFT、Online RL、Offline OPD、Support-aware Offline OPD 和 Online OPD。

## 顶层目录

```text
Offline_OPD/
├── README.md
├── REPORT.md
├── PROJECT_STRUCTURE_cn.md
├── run_all.py
├── requirements.txt
├── pyproject.toml
├── opd_agentic_shift/
├── latex_repo/
├── results/
└── runs/
```

## 关键入口

- `run_all.py`：一键运行完整实验流程，支持 `--profile quick` 和 `--profile full`。
- `README.md`：项目快速开始说明。
- `REPORT.md`：由 `run_all.py` 自动生成的 Markdown 版实验报告。
- `latex_repo/main_cn.tex`：中文 LaTeX 报告入口。
- `results/`：建议打包提交给 mentor 的精简实验结果。
- `runs/`：完整运行产物和中间文件，主要用于复现与调试。

## 代码主体：opd_agentic_shift/

```text
opd_agentic_shift/
├── envs/
│   ├── troubleshooting_env.py
│   └── encoder.py
├── teachers/
│   └── rule_teacher.py
├── policies/
│   └── mlp_policy.py
├── data/
│   ├── generate_expert_data.py
│   ├── collect_rollouts.py
│   └── build_offline_opd_data.py
├── algos/
│   ├── sft.py
│   ├── online_rl.py
│   ├── offline_opd.py
│   └── online_opd.py
├── eval/
│   ├── evaluate.py
│   └── case_analysis.py
└── utils/
    └── io.py
```

### envs/

- `troubleshooting_env.py`：定义多轮故障排查环境、动作空间、隐藏故障、错误修复动作的副作用。
- `encoder.py`：把符号观测编码成 MLP policy 可用的向量。

### teachers/

- `rule_teacher.py`：规则 teacher / oracle。它能读取隐藏故障，用于生成专家轨迹和 OPD teacher probabilities。

### policies/

- `mlp_policy.py`：所有方法共用的小型 MLP 策略网络。

### data/

- `generate_expert_data.py`：用 rule teacher 生成专家数据。
- `collect_rollouts.py`：用 SFT policy 采样 student rollouts。
- `build_offline_opd_data.py`：对 student rollout 中的状态预计算 teacher distribution，形成 Offline OPD 数据。

### algos/

- `sft.py`：行为克隆 baseline。
- `online_rl.py`：REINFORCE baseline。
- `offline_opd.py`：普通 Offline OPD，以及 support-aware loss + anchor regularization patch。
- `online_opd.py`：在线查询 teacher 的 OPD 上界方法。

### eval/

- `evaluate.py`：评估 success rate、average reward、trajectory length、off-support ratio；支持 `--initial_wrong_repair_prob` 压力测试。
- `case_analysis.py`：打印成功和失败 case。

## latex_repo/

`latex_repo/` 是可以直接上传到 Overleaf 的中文报告包。
它保留报告源码、LaTeX 配置和必要的汇总结果数据。

```text
latex_repo/
├── README_cn.md
├── main_cn.tex
├── setup.tex
└── results/
    ├── README_cn.md
    ├── eval/
    │   └── summary.json
    ├── eval_shift/
    │   └── summary.json
    └── tables/
        └── summary_tables_cn.csv
```

- `main_cn.tex`：中文完整报告正文，也是 Overleaf 主文件。
- `setup.tex`：LaTeX 宏包与命令定义。
- `results/`：报告表格对应的 summary JSON 和 CSV 数据。

## results/

`results/` 是建议随项目打包提交的精简结果目录。

```text
results/
├── README_cn.md
├── eval/
│   └── summary.json
├── eval_shift/
│   └── summary.json
└── tables/
    └── summary_tables_cn.csv
```

- `eval/summary.json`：正常分布评估的汇总结果。
- `eval_shift/summary.json`：注入早期错误 repair action 后的 agentic-shift 压力测试汇总结果。
- `tables/summary_tables_cn.csv`：两张主结果表的 CSV 汇总。

## runs/

`runs/` 是完整实验运行目录，包含可复现和调试用的中间产物。

```text
runs/
├── data/
├── ckpts/
├── eval/
└── eval_shift/
```

- `runs/data/`：训练数据与 OPD 标注数据。
- `runs/ckpts/`：训练得到的 policy checkpoint。
- `runs/eval/`：正常评估的逐方法详细 JSON 和 summary。
- `runs/eval_shift/`：压力测试的逐方法详细 JSON 和 summary。

如果只提交给 mentor，重点看 `results/`、`latex_repo/`、`opd_agentic_shift/` 和 `run_all.py`。
如果需要完整复现实验或查看每个 episode 的动作轨迹，再看 `runs/`。

## 推荐运行命令

```bash
python run_all.py --profile quick
```

该命令会重新生成：

- `runs/` 下的完整实验产物；
- `results/` 下的精简提交结果；
- 根目录的 `REPORT.md`。
