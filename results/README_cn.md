# 实验结果归档说明

本目录只保留中文 LaTeX 报告需要引用的精简结果。
完整训练数据、checkpoint 和逐 episode 评估详情保留在项目根目录的 `runs/` 下，可通过复现实验重新生成。

## 当前结构

```text
results/
├── eval/
│   └── summary.json
├── eval_shift/
│   └── summary.json
└── tables/
    └── summary_tables_cn.csv
```

## 文件说明

- `eval/summary.json`：正常分布评估的汇总指标。
- `eval_shift/summary.json`：注入一次早期错误 repair action 后的 agentic-shift 压力测试汇总指标。
- `tables/summary_tables_cn.csv`：正常评估与 shifted 评估的主结果表格汇总，便于检查和二次制表。

## 复现命令

在项目根目录运行：

```bash
python run_all.py --profile quick
```

该命令会重新生成 `runs/` 下的完整数据与评估详情。
