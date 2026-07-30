# 新增任务

新增任务通常涉及 scenario、goal 和 task 三类数据。目标是让任务能被数据加载器发现，并能被 planner、validator 和 goal monitor 正确处理。

## 关键路径

```text
modules/dataset_builder/generate_scenarios.py
modules/dataset_builder/generate_goals.py
modules/dataset_builder/generate_tasks.py
modules/dataset_builder/goal_utils/
modules/dataset_builder/scene_utils/
```

## 接入步骤

1. 定义任务目标类型和成功条件。
2. 确认 scenario 中包含所需实体、区域和机器人。
3. 生成或手写 goal。
4. 将 scenario 与 goal 组合为 task。
5. 使用小批量 benchmark 验证运行结果。

## 成功条件

成功条件应能被 `GoalProgressMonitor` 或 validator 判断。仅保留自然语言描述不足以支撑自动验证；需要提供结构化字段或可检查状态。

相关概念见 [Task 与 State](../concepts/task-and-state.md)。

## 验证

```bash
python run/run_exp_multi_method.py \
  --methods sgi \
  --max-count 5 \
  --dataset-root "$GSI_DATASET_ROOT"
```
