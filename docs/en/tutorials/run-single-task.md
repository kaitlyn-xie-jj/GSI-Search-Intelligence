# Run a Single Task

Single-task execution is useful for debugging planner behavior, configuration, and execution feedback. Batch evaluation should use [Run Benchmark](run-benchmark.md).

## Entry Point

```bash
python run/run_exp.py
```

The task id, configuration, and model parameters in the single-task script may need direct edits in the script or config files. Before running, confirm:

- `solver_type`, `platform_type`, and `enable_replanning` in `config/default.json`.
- The LLM endpoint is configured in `modules/task_solver/llm_framework/config/llm_config.yaml` or environment variables.
- The dataset path exists, or the script can read the default `dataset/` directory.

## Debugging Guidance

- Use the semantic platform for fast validation.
- Enable `enable_detailed_print` for more detailed execution logs.
- To reproduce batch results, use `run/run_exp_multi_method.py` instead of temporary task ids in the single-task script.

## Related Pages

- Batch execution: [Run Benchmark](run-benchmark.md).
- Configuration fields: [Configuration](../reference/config.md).
