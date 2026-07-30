# Task 与 State

GSI 的运行对象由 scenario、goal、task 和 state 组成。它们共同决定任务输入、执行条件、验证依据和训练样本上下文。

## Scenario

Scenario 描述环境初始状态，通常包括：

- 机器人类型、位置和能力。
- 区域、道路、建筑和兴趣点。
- 物体、事件和可交互实体。
- 实体之间的空间关系。

相关路径：

```text
modules/dataset_builder/generate_scenarios.py
modules/dataset_builder/scene_utils/
modules/dataset_loader/
```

## Goal

Goal 描述任务目标和成功条件。常见类型包括 `search`、`transport`、`assembly`、`patrol`、`guidance`、`verbal broadcast` 和 `traffic enforcement`。

常见字段：

- `instruction`：给模型或 planner 的自然语言指令。
- `goal_details`：结构化目标描述。
- `success_condition`：成功判定条件。
- `meta`：规划、协同和语言难度标签。

## Task

Task 是 scenario 与 goal 的组合。Benchmark 会从数据集中筛选 task id，并将同一批任务交给一个或多个方法运行。

入口：

```bash
python run/run_exp_multi_method.py --help
```

## State

State 表示运行过程中的世界状态和 replan 上下文。RLVR 数据中的 state shard 会被 validator 加载，用于复现计划验证和 reward 计算。

典型 RLVR 数据结构：

```text
train.parquet
val.parquet
states.index.json
states/*.jsonl
```

训练说明见 [RLVR 训练](../training/rlvr.md)。
