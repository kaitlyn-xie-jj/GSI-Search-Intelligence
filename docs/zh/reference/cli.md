# CLI 与脚本

本页列出常用入口。完整参数以 `--help` 输出为准。

## Benchmark

```bash
python run/run_exp_multi_method.py --help
```

用途：

- 运行 SGI 或 baseline planner。
- 设置任务类型、难度筛选和样本数。
- 启用 new case 评估。
- 指定输出目录。

## Replan 数据采集

```bash
python run/run_collect_replan_dataset.py --help
```

用途：

- 批量运行任务。
- 注入 new case。
- 捕获 replan prompt、response 和 state。
- 生成后续训练数据来源。

## Validator Server

```bash
python run/plan_validation_server.py
```

训练容器中推荐使用 wrapper：

```bash
./scripts/runtime/serve_validator.sh start
./scripts/runtime/serve_validator.sh status
./scripts/runtime/serve_validator.sh logs
./scripts/runtime/serve_validator.sh stop
```

## SFT / RLVR

```bash
./scripts/runtime/train_sft_unsloth.sh
./scripts/runtime/train_rlvr.sh
```

详见 [SFT 训练](../training/sft.md) 和 [RLVR 训练](../training/rlvr.md)。
