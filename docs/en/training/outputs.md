# Output Directory

This page explains outputs produced by `run/run_exp_multi_method.py`. Typical path:

```text
outputs/eval_multi_method/qwen3_0_6b_cybertown/<timestamp>/newcase_0/batch_sgi/
```

With Docker compose, the host path may be:

```text
outputs/docker_runtime_train/eval_multi_method/qwen3_0_6b_cybertown/<timestamp>/newcase_0/batch_sgi/
```

## Directory Structure

```text
outputs/eval_multi_method/<eval_name>/
  <timestamp>/
    newcase_0/
      batch_sgi/
        aggregate_full.json
        summary.jsonl
        0001_<task_id>/
          metrics.json
          single_run_summary.json
          temp_vars.jsonl
          log.md
          alloc_config/
            planner_param.yaml
            scene.yaml
            task_param.yaml
            vehicle_param.yaml
          alloc_results.yaml
    newcase_4/
      batch_sgi/
        aggregate_full.json
        summary.jsonl
```

Layer meanings:

- `<eval_name>`: experiment directory under `--output-root`.
- `<timestamp>`: one run timestamp.
- `newcase_N`: at most N active new-case injections; standard benchmark may not have this layer.
- `batch_<method>`: batch result for one planner method.
- `0001_<task_id>`: single-task run directory.

## Key Files

### `aggregate_full.json`

Batch aggregate metrics for tables, method comparison, and new-case intensity comparison.

Common fields:

- `overall`: overall metrics.
- `by_goal_type`: metrics grouped by task type.
- `input_params`: runtime parameter snapshot.

### `summary.jsonl`

One line per task run. Useful for filtering failed samples, locating task ids, and finding run directories.

Common fields:

- `goal_type`
- `goal_id`
- `run_dir`
- `metrics`
- `algorithm_specific`

### `single_run_summary.json`

Single-task summary, usually including runtime parameters, goal information, success status, elapsed time, and key metrics.

### `metrics.json`

Single-task metrics file. Use it to inspect success status, elapsed time, replan counts, energy, and token statistics.

### `log.md`

Human-readable run log. Use it to inspect planner output, execution steps, feedback, and failure reasons.

### `temp_vars.jsonl`

Debug variable snapshots, one JSON object per line. Common records:

- `run_input_default`
- `goal`
- `prompt`
- `response`
- `dispatcher_result`
- `skills_by_timestep`
- `execution_result`
- `feedback_event`
- `new_case_event`

Use this file when `log.md` is too long or script-based filtering is needed.

### `alloc_config/`

Configuration snapshot passed to the TANGO allocator:

- `planner_param.yaml`
- `task_param.yaml`
- `vehicle_param.yaml`
- `scene.yaml`

Used to reproduce or debug task allocation.

### `alloc_results.yaml`

TANGO allocator output. Common fields:

- `result.flagSuccess`
- `energyConstraintCheck`
- `energyCost`
- `finalAllCost`
- `optStatus`
- `objVal`
- `MIPGap`
- `solverTime`
- `team`
- `vehicle_paths`

If this file is missing, the run usually did not enter allocation or failed before allocation completed.

## Aggregate Metrics

Common fields in `aggregate_full.json`:

| Field | Meaning |
| --- | --- |
| `count_runs` | Number of runs included. |
| `success_rate` | Fraction of successful runs. |
| `elapsed_sec` | Total runtime per run. |
| `llm_calls` | Number of LLM calls. |
| `planning_duration` | Planning latency. |
| `allocation_duration` | Allocation latency. |
| `replans_full` | Full replan count. |
| `replans_partial` | Partial replan count. |
| `replans_total` | Total replan count. |
| `total_energy` | Cumulative energy or movement cost. |
| `newcase_total_orig` | Actively injected new cases. |
| `newcase_total` | New cases recorded during execution. |
| `newcase_top_types` | Frequent new-case types. |
| `completed_tasks` | Completed atomic task count. |
| `total_tasks` | Total atomic task count. |
| `atomic_task_completion_rate` | Atomic task completion ratio. |
| `prompt_tokens_total` | Total prompt tokens. |
| `response_tokens_total` | Total response tokens. |

Token statistics depend on model service support. With `GSI_DISABLE_TOKEN_STATS=1`, token fields are usually unavailable or unreliable.

## Compare Results

Compare new-case intensity:

```text
newcase_0/batch_sgi/aggregate_full.json
newcase_4/batch_sgi/aggregate_full.json
```

Focus on:

- `success_rate`
- `replans_total`
- `elapsed_sec`
- `llm_calls`
- `atomic_task_completion_rate`
- `by_goal_type`

Compare methods:

```text
newcase_0/batch_sgi/
newcase_0/batch_spine/
newcase_0/batch_smartllm/
newcase_0/batch_lipllm/
```

For SFT/RLVR model evaluation, inspect `batch_sgi` first. Baseline methods are more useful for base-model comparisons.

## Quick Inspection

View aggregate files:

```bash
RUN=outputs/eval_multi_method/qwen3_0_6b_cybertown/2026-06-04_19-02-36

python -m json.tool "$RUN/newcase_0/batch_sgi/aggregate_full.json" | less
python -m json.tool "$RUN/newcase_4/batch_sgi/aggregate_full.json" | less
```

View the first summary row:

```bash
head -1 "$RUN/newcase_0/batch_sgi/summary.jsonl" | python -m json.tool
```

List failed tasks:

```bash
python - <<'PY'
import json
from pathlib import Path

summary = Path("outputs/eval_multi_method/qwen3_0_6b_cybertown/2026-06-04_19-02-36/newcase_0/batch_sgi/summary.jsonl")
for line in summary.read_text().splitlines():
    row = json.loads(line)
    if not row.get("metrics", {}).get("success", False):
        print(row["goal_type"], row["goal_id"], row["run_dir"])
PY
```

Summarize by task type:

```bash
python - <<'PY'
import json
from collections import defaultdict
from pathlib import Path

summary = Path("outputs/eval_multi_method/qwen3_0_6b_cybertown/2026-06-04_19-02-36/newcase_0/batch_sgi/summary.jsonl")
stats = defaultdict(lambda: [0, 0])
for line in summary.read_text().splitlines():
    row = json.loads(line)
    key = row.get("goal_type", "unknown")
    stats[key][1] += 1
    stats[key][0] += int(bool(row.get("metrics", {}).get("success", False)))
for key, (ok, total) in sorted(stats.items()):
    print(f"{key}: {ok}/{total} = {ok / total:.2%}")
PY
```

Inspect a failed run:

```bash
CASE=outputs/eval_multi_method/qwen3_0_6b_cybertown/2026-06-04_19-02-36/newcase_0/batch_sgi/0001_cybertown_scenario_1_g_19075
python -m json.tool "$CASE/single_run_summary.json" | less
less "$CASE/log.md"
```

## Common Misreadings

- `ok=true` is not task success; use `metrics.success` or `success_rate`.
- `summary.jsonl` order may follow completion order under concurrency, not task id order.
- Under `newcase_0`, `newcase_total_orig` should be 0. Nonzero `newcase_total` means events were recorded during execution.
- `aggregate_full.json` is complete only after the batch finishes.
- Missing `alloc_config/` or `alloc_results.yaml` usually means the run did not reach allocation.
- Token fields are empty or unreliable when token statistics are disabled.
