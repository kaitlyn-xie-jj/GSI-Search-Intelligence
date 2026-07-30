# Reproduce Results

GSI results depend on the code version, dataset, model checkpoint, vLLM served model name, task filters, new-case settings, solver backend, and concurrency. Formal reproduction should fix these variables and record the full command.

## Reproduction Paths

1. Benchmark reproduction: evaluate SGI with a released model.
2. Multi-method comparison: compare SGI with baseline planners on the same task set.
3. New Case / Replan evaluation: evaluate robustness under incidents and replanning.
4. Training reproduction: rerun SFT or RLVR and evaluate the resulting checkpoint.

The first three are evaluation workflows. Training reproduction is a model training workflow. Keep their output directories separate.

## Variables to Fix

| Variable | Recommended record | Notes |
| --- | --- | --- |
| Code version | `git rev-parse HEAD` | Runtime logic, prompts, and aggregation may change. |
| Dataset | `GSI_DATASET_ROOT` or snapshot path | Keeps task discovery and filters consistent. |
| Model | repo id or checkpoint path | Distinguishes base, SFT, and RLVR models. |
| Served model name | id from `/v1/models` | Must equal `GSI_LLM_MODEL`. |
| Command | full shell command | Includes methods, counts, filters, and new-case settings. |
| Solver backend | `GSI_TANGO_SOLVER_BACKEND` | Affects allocation results and timeouts. |
| Concurrency | `--max-workers`, `--max-num-seqs` | Affects model service stability. |
| Output directory | `--output-root` and timestamp | Required for later comparison. |

## 1. Evaluate the Released RLVR Model

Set the model:

```bash
export SERVED_MODEL_NAME=qwen3_0_6b_cybertown_rlvr
export MODEL_PATH=WindyLab/Qwen3-0.6B-cybertown-RLVR
```

After starting vLLM:

```bash
curl -fsS http://127.0.0.1:8001/v1/models | python -m json.tool
```

Configure GSI:

```bash
export GSI_LLM_API_BASE=http://127.0.0.1:8001/v1
export GSI_LLM_API_KEY=EMPTY
export GSI_LLM_MODEL=qwen3_0_6b_cybertown_rlvr
export GSI_DISABLE_TOKEN_STATS=1
export GSI_TANGO_SOLVER_BACKEND=scip
export GSI_TANGO_SOLVER_MAX_TIME=120
```

Prepare the dataset:

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

Run a smoke test:

```bash
python run/run_exp_multi_method.py \
  --methods sgi \
  --task-mix transport=1.0 \
  --max-count 5 \
  --max-workers 1 \
  --dataset-root "$GSI_DATASET_ROOT" \
  --output-root results/reproduce/smoke_rlvr
```

Run the formal benchmark:

```bash
python run/run_exp_multi_method.py \
  --methods sgi \
  --max-count 200 \
  --max-workers 10 \
  --dataset-root "$GSI_DATASET_ROOT" \
  --output-root results/reproduce/sgi_rlvr
```

High concurrency can increase request timeouts and malformed outputs. Confirm stability before raising `--max-workers`.

## 2. Multi-Method Comparison

`run_exp_multi_method.py` fixes the task set before running each method:

```bash
python run/run_exp_multi_method.py \
  --methods sgi spine smartllm lipllm \
  --max-count 200 \
  --max-workers 10 \
  --dataset-root "$GSI_DATASET_ROOT" \
  --output-root results/reproduce/multi_method_rlvr
```

Output structure:

```text
results/reproduce/multi_method_rlvr/<timestamp>/
  batch_sgi/
  batch_spine/
  batch_smartllm/
  batch_lipllm/
```

Record at least:

- `success_rate`
- `elapsed_sec`
- `llm_calls`
- `replans_total`
- `total_energy`
- `prompt_tokens_total` / `response_tokens_total`

Metric definitions are in [Evaluation Metrics](../concepts/evaluation-metrics.md).

## 3. New Case / Replan Evaluation

Standard benchmarks do not inject new cases. Enable them explicitly:

```bash
python run/run_exp_multi_method.py \
  --methods sgi \
  --enable-newcase \
  --newcase-counts 1 2 3 4 \
  --max-count 200 \
  --max-workers 10 \
  --dataset-root "$GSI_DATASET_ROOT" \
  --output-root results/reproduce/newcase_rlvr
```

`newcase_0` is the baseline without active injection:

```text
results/reproduce/newcase_rlvr/<timestamp>/
  newcase_0/batch_sgi/
  newcase_1/batch_sgi/
  newcase_2/batch_sgi/
  newcase_3/batch_sgi/
  newcase_4/batch_sgi/
```

Compare:

- `success_rate`
- `replans_full` / `replans_partial`
- `newcase_total`
- `newcase_top_types`
- `elapsed_sec`
- `llm_calls`

## 4. Training Reproduction

Training reproduction has two stages:

1. SFT: learn the task planning format, constraints, and output structure from a base model.
2. RLVR: continue from the SFT model with validator/reward signals.

Entry points:

- [SFT Training](../training/sft.md)
- [RLVR Training](../training/rlvr.md)
- [Hugging Face Preparation](../training/huggingface-prepare.md)

Training logs are not benchmark results. After training, start the model through vLLM and evaluate it with the benchmark flow above.

## Inspect Outputs

Important files under each `batch_<method>/`:

- `summary.jsonl`: one row per task, useful for failure analysis.
- `aggregate_full.json`: aggregate metrics for tables and comparisons.
- `0001_<task_id>/`: single-run directory with logs, temporary variables, LLM outputs, and execution state.

View the latest aggregate:

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("results/reproduce/sgi_rlvr")
latest = max(root.iterdir(), key=lambda p: p.stat().st_mtime)
agg = latest / "batch_sgi" / "aggregate_full.json"
data = json.loads(agg.read_text())
print(agg)
print(json.dumps(data["overall"], ensure_ascii=False, indent=2))
PY
```

See [Output Directory](../training/outputs.md) for details.

## Check Order for Mismatched Results

1. The model id from `/v1/models` equals `GSI_LLM_MODEL`.
2. Base, SFT, and RLVR models are not mixed.
3. `GSI_DATASET_ROOT` points to the same snapshot.
4. `--max-count`, `--task-mix`, `--plan-level`, `--coor-level`, and `--lang-level` match.
5. `--enable-newcase` is consistently set.
6. `GSI_TANGO_SOLVER_BACKEND` and `GSI_TANGO_SOLVER_MAX_TIME` match.
7. `--max-workers` does not overload vLLM.
8. Output directories do not mix different experiments.
