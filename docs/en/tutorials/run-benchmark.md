# Run Benchmark

Batch benchmark runs one or more planner methods on the same task set. Entry point:

```bash
python run/run_exp_multi_method.py
```

## Basic SGI Run

```bash
python run/run_exp_multi_method.py \
  --methods sgi \
  --max-count 100 \
  --max-workers 10 \
  --dataset-root "$GSI_DATASET_ROOT" \
  --output-root results/batch_runs
```

## Multi-Method Comparison

```bash
python run/run_exp_multi_method.py \
  --methods sgi spine smartllm lipllm \
  --max-count 200 \
  --max-workers 10 \
  --dataset-root "$GSI_DATASET_ROOT" \
  --output-root results/batch_runs
```

Supported methods:

```text
sgi
spine
smartllm
lipllm
```

## Task Filters

Common parameters:

```bash
--task-mix
--plan-level
--coor-level
--lang-level
--max-count
```

Example:

```bash
python run/run_exp_multi_method.py \
  --methods sgi \
  --task-mix transport=0.5 assembly=0.5 \
  --max-count 20 \
  --dataset-root "$GSI_DATASET_ROOT"
```

## New Case

```bash
python run/run_exp_multi_method.py \
  --methods sgi \
  --enable-newcase \
  --newcase-counts 1 2 3 4 \
  --dataset-root "$GSI_DATASET_ROOT"
```

## Output Structure

Standard benchmark:

```text
results/batch_runs/<timestamp>/
  batch_sgi/
  batch_spine/
```

New-case benchmark:

```text
results/batch_runs/<timestamp>/
  newcase_0/
  newcase_1/
```

Output files are described in [Output Directory](../training/outputs.md).
