# Configuration

`config/default.json` defines GSI runtime defaults, including planner, platform, replan, new case, HITL, and part of solver configuration. Model service address, Hugging Face cache, benchmark dataset path, and TANGO backend are usually set through CLI arguments or environment variables.

## Configuration Sources

Common priority:

1. CLI arguments.
2. Explicit environment variables.
3. `config/default.json`.
4. Script or module defaults.

Common entry points:

- `python run/run_exp_multi_method.py ...`
- `python run/run_collect_replan_dataset.py ...`
- `config/llm_finetune.json`

## Recommended Defaults

```json
{
  "planner_mode": "full",
  "platform_type": "semantic",
  "solver_type": "sgi",
  "enable_replanning": true,
  "enable_new_case_generation": false,
  "collect_replan_dataset": false
}
```

The default path uses SGI, semantic platform, and replanning. Standard benchmarks do not inject new cases. New-case evaluation and replan data collection should enable them through script arguments.

## Planner and LLM

- `planner_mode=full`: recommended mode; generate a complete task plan.
- `planner_mode=phase`: legacy phase mode; not the default recommended path.
- `llm_temperature`: usually `0.0` for evaluation and data collection.
- `default_robot_types`: robot type allowlist for planning.

Default robot types:

```json
["UAV", "UGV", "Quadruped", "Humanoid"]
```

## Runtime and Debugging

- `enable_detailed_print`: detailed runtime logs for single-case debugging.
- `simulate_time_delay`: simulated skill wait time; usually disabled in batch evaluation.
- `enable_visualization`: visualization switch.
- `enable_video_recording`: video recording switch.
- `fine_grained_simulation`: fine-grained semantic simulation.
- `max_concurrency`: default concurrency limit; batch scripts usually use `--max-workers`.
- `enable_logging`, `enable_checkpointing`: logs and intermediate state snapshots.

## Replan and New Case

- `enable_replanning`: global replanning switch.
- `enable_new_case_generation`: new-case injection switch.
- `collect_replan_dataset`: whether to collect replan training data.
- `max_newcases_per_run`: maximum new cases in one run.
- `new_case_mode`: new-case handling strategy.
- `newcase_spacing_factor`: injection spacing control.
- `newcase_cooldown_rounds`: cooldown rounds after injection.
- `newcase_similarity_threshold`, `newcase_similarity_damping`: repeated injection suppression under similar plans.

New-case benchmark example:

```bash
python run/run_exp_multi_method.py \
  --methods sgi \
  --enable-newcase \
  --newcase-counts 4 \
  --dataset-root "$GSI_DATASET_ROOT" \
  --output-root outputs/eval_multi_method/qwen3_0_6b_cybertown
```

## Dataset and Platform

`repo_id` is the default remote task dataset repository:

```text
WindyLab/GSI
```

Formal runs should explicitly set the dataset path:

```bash
export GSI_DATASET_ROOT=$(
python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download(
    repo_id="WindyLab/GSI",
    repo_type="dataset",
    revision="small",
))
PY
)
```

Platform types:

- `semantic`: default platform for training, validator, benchmark, and batch evaluation.
- `unreal`: connects to an Unreal Engine service for UE5 integration.

When `platform_type=unreal`, `base_url`, `timeout`, and `polling_interval` under `unreal_platform` take effect.

## Replay Mode

Replay mode replays existing traces instead of requesting new LLM plans:

- `enabled`: global switch.
- `trace_root`: trace root directory.
- `trace_tag`: tag under the trace directory.

Standard evaluation and post-training validation should keep `replay_mode.enabled=false`.

## Human-In-The-Loop

HITL is for manual intervention, mainly in Unreal or interactive flows. Non-interactive benchmarks usually keep it disabled:

- `enabled`
- `instruction_enabled`
- `review_enabled`
- `decision_enabled`
- `instruction_timeout`
- `review_timeout`
- `decision_timeout`
- `server_port`
- `retry_count`
- `retry_delay`

## Planner Methods

`solver_type` selects the planner method, not the TANGO optimization backend. Common values:

- `sgi`
- `llamar`
- `spine`
- `lipllm`
- `smartllm`

`solver_config` stores method-specific parameters. Common fields:

- `max_steps`
- `model_family`
- `model_name_override`
- `validate_plan`
- `use_few_shot`
- `n_attempts`
- `max_iterations`
- `alpha`

Multi-method benchmark should use CLI:

```bash
python run/run_exp_multi_method.py \
  --methods sgi spine smartllm lipllm \
  --dataset-root "$GSI_DATASET_ROOT" \
  --output-root outputs/eval_multi_method/qwen3_0_6b_cybertown_baselines
```

## Configuration Outside JSON

LLM endpoint:

```bash
export GSI_LLM_API_BASE=http://127.0.0.1:8001/v1
export GSI_LLM_API_KEY=EMPTY
export GSI_LLM_MODEL=qwen3_0_6b_cybertown_sft
export GSI_DISABLE_TOKEN_STATS=1
```

TANGO backend:

```bash
export GSI_TANGO_SOLVER_BACKEND=scip
export GSI_TANGO_SOLVER_MAX_TIME=120
```

Dataset and offline cache:

```bash
export GSI_DATASET_ROOT=/GSI/dataset
export HF_HUB_OFFLINE=1
```

## Common Recipes

Evaluate a trained model:

```bash
export GSI_LLM_API_BASE=http://127.0.0.1:8001/v1
export GSI_LLM_API_KEY=EMPTY
export GSI_LLM_MODEL=qwen3_0_6b_cybertown_sft
export GSI_TANGO_SOLVER_BACKEND=scip
export GSI_TANGO_SOLVER_MAX_TIME=120

python run/run_exp_multi_method.py \
  --methods sgi \
  --enable-newcase \
  --newcase-counts 4 \
  --max-count 100 \
  --max-workers 10 \
  --dataset-root "$GSI_DATASET_ROOT" \
  --output-root outputs/eval_multi_method/qwen3_0_6b_cybertown
```

Collect replan data:

```bash
python run/run_collect_replan_dataset.py \
  --batch-name qwen3_0_6b_cybertown_nc4 \
  --output-root outputs/replan_collect \
  --dataset-root "$GSI_DATASET_ROOT" \
  --max-count 100 \
  --max-workers 10 \
  --max-newcases-per-run 4 \
  --capture-save-llm-io
```

## Troubleshooting Checklist

- `GSI_LLM_MODEL` must match the vLLM served name.
- Non-`null` `model_name_override` overrides the environment model name.
- `solver_type` selects the planner; `GSI_TANGO_SOLVER_BACKEND` selects the allocator solver.
- New-case parameters do not affect standard execution when `enable_new_case_generation=false`.
- `collect_replan_dataset=true` changes output and collection behavior.
- `platform_type=unreal` requires a reachable UE service.
- Excessive `--max-workers` can cause model timeouts, truncated outputs, or abnormal failure rates.
