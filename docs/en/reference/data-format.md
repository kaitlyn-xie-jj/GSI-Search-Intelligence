# Data Format

GSI uses benchmark tasks, replan collection records, training data, and evaluation outputs. These workflows use different data directories and should not be mixed.

## Benchmark Data

Benchmark data comes from:

```text
WindyLab/GSI
```

At runtime, specify the local snapshot through `GSI_DATASET_ROOT` or `--dataset-root`. The dataset usually contains scenario, goal, and task files.

## Task / Goal / Scenario

- Scenario: environment, robots, locations, entities, and relations.
- Goal: task objective, success condition, and difficulty tags.
- Task: runnable sample combining scenario and goal.

Concepts are explained in [Task and State](../concepts/task-and-state.md).

## Replan / RLVR Data

RLVR data usually contains:

```text
train.parquet
val.parquet
states.index.json
states/*.jsonl
```

`train.parquet` and `val.parquet` store training samples. `states.index.json` and `states/*.jsonl` store replan state required by the validator.

## Evaluation Outputs

Benchmark output usually contains:

```text
summary.jsonl
aggregate_full.json
0001_<task_id>/
```

Fields are described in [Output Directory](../training/outputs.md).
