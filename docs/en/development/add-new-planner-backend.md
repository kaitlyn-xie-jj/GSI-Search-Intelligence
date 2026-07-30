# Add a Planner Backend

Add a planner backend when integrating a new LLM planner, rule-based planner, or baseline method. New methods should reuse the shared task lifecycle for fair comparison.

## Key Paths

```text
modules/task_solver/solver_factory.py
modules/task_solver/sgi_task_solver.py
modules/task_solver/baseline_planners/
modules/task_solver/sgi_planner/
```

## Integration Requirements

- Reuse the same task, world model, and robot skill inputs.
- Produce output that can be converted into platform-executable skill sequences.
- Return format and constraint errors to the validator or retry logic.
- Make the method selectable through `--methods`.

## Validation

```bash
python run/run_exp_multi_method.py \
  --methods <new_method> \
  --max-count 5 \
  --dataset-root "$GSI_DATASET_ROOT"
```

If the planner depends on a new LLM endpoint, update:

```text
modules/task_solver/llm_framework/config/llm_config.yaml
```
