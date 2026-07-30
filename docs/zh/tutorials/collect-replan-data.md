# 采集 Replan 数据

Replan 数据采集用于构建后续 SFT/RLVR 数据。采集入口为：

```bash
python run/run_collect_replan_dataset.py --help
```

## 基础命令

```bash
python run/run_collect_replan_dataset.py \
  --max-count 100 \
  --max-workers 10 \
  --max-newcases-per-run 1 \
  --dataset-root "$GSI_DATASET_ROOT" \
  --output-root results/replan_batch_runs
```

## 常用过滤参数

```bash
--capture-min-newcases
--capture-min-replan-index
--capture-goal-types
--capture-event-types
--capture-event-reasons
--capture-max-records-per-run
--capture-save-llm-io
--capture-data-tag
```

示例：

```bash
python run/run_collect_replan_dataset.py \
  --max-count 200 \
  --max-newcases-per-run 2 \
  --capture-goal-types transport assembly \
  --capture-data-tag train_replan_v1 \
  --dataset-root "$GSI_DATASET_ROOT"
```

## 输出用途

采集结果可整理为训练用 parquet 和 state shard。RLVR 期望的数据结构见 [RLVR 训练](../training/rlvr.md)。
