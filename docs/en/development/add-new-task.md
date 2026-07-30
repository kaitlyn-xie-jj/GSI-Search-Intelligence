# Add a New Task

Adding a task usually involves scenario, goal, and task data. The task should be discoverable by the data loader and correctly handled by the planner, validator, and goal monitor.

## Key Paths

```text
modules/dataset_builder/generate_scenarios.py
modules/dataset_builder/generate_goals.py
modules/dataset_builder/generate_tasks.py
modules/dataset_builder/goal_utils/
modules/dataset_builder/scene_utils/
```

## Integration Steps

1. Define the goal type and success condition.
2. Confirm that the scenario contains required entities, regions, and robots.
3. Generate or write the goal.
4. Combine scenario and goal into a task.
5. Validate the task with a small benchmark run.

## Success Condition

The success condition should be checkable by `GoalProgressMonitor` or the validator. A natural-language description alone is insufficient for automatic validation; provide structured fields or checkable state.

See [Task and State](../concepts/task-and-state.md) for concepts.

## Validation

```bash
python run/run_exp_multi_method.py \
  --methods sgi \
  --max-count 5 \
  --dataset-root "$GSI_DATASET_ROOT"
```
