# 数据格式

GSI 的主要数据包括 benchmark task、replan 采集记录、训练数据和评估输出。不同流程使用的数据目录不同，应避免混用。

## Benchmark 数据

Benchmark 数据集来自：

```text
WindyLab/GSI
```

运行时通过 `GSI_DATASET_ROOT` 或 `--dataset-root` 指定本地 snapshot。数据中通常包含 scenario、goal 和 task 相关文件。

## Task / Goal / Scenario

- Scenario：描述环境、机器人、位置、实体和关系。
- Goal：描述任务目标、成功条件和难度标签。
- Task：将 scenario 与 goal 组合为可运行样本。

概念说明见 [Task 与 State](../concepts/task-and-state.md)。

## Replan / RLVR 数据

RLVR 数据目录通常包含：

```text
train.parquet
val.parquet
states.index.json
states/*.jsonl
```

`train.parquet` 和 `val.parquet` 保存训练样本；`states.index.json` 与 `states/*.jsonl` 保存 validator 所需的 replan state。

## 评估输出

Benchmark 输出通常包含：

```text
summary.jsonl
aggregate_full.json
0001_<task_id>/
```

字段说明见 [输出目录](../training/outputs.md)。
