# LaTeX / Overleaf 提交目录说明

本目录是可以直接上传到 Overleaf 的中文报告包。
`main_cn.tex` 是唯一主入口，内容已经直接展开，不再通过 `\input{src/...}` 跳转到其他正文文件。

## 目录结构

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

## 文件说明

- `main_cn.tex`：中文完整报告正文，包含论文理解、toy task、代码框架、方法、实验、case 分析和结论。
- `setup.tex`：LaTeX 宏包、格式和自定义命令。
- `results/eval/summary.json`：正常分布评估汇总结果。
- `results/eval_shift/summary.json`：注入早期错误 repair action 后的 agentic-shift 压力测试汇总结果。
- `results/tables/summary_tables_cn.csv`：中文报告中主结果表的 CSV 数据。

## Overleaf 使用方式

上传整个 `latex_repo/` 目录后，将主文件设置为：

```text
main_cn.tex
```

中文编译建议使用 XeLaTeX。

## 复现实验

如需重新生成实验结果，在项目根目录运行：

```bash
python run_all.py --profile quick
```

该命令会刷新根目录的 `runs/`、`results/` 和 `REPORT.md`。
如果要更新 Overleaf 包里的结果数据，将根目录 `results/` 同步复制到 `latex_repo/results/` 即可。
