# Collect Replan Data

Replan data collection is used to build later SFT/RLVR datasets. Entry point:

```bash
python run/run_collect_replan_dataset.py --help
```

## Basic Command

```bash
python run/run_collect_replan_dataset.py \
  --max-count 100 \
  --max-workers 10 \
  --max-newcases-per-run 1 \
  --dataset-root "$GSI_DATASET_ROOT" \
  --output-root results/replan_batch_runs
```

## Common Capture Filters

```bash
--capture-min-newcases
--capture-min-replan-index
--capture-goal-types
--capture-event-types
--capture-event-reasons
--capture-max-records-per-run
--capture-save-llm-io
--capture-data-tag
```

Example:

```bash
python run/run_collect_replan_dataset.py \
  --max-count 200 \
  --max-newcases-per-run 2 \
  --capture-goal-types transport assembly \
  --capture-data-tag train_replan_v1 \
  --dataset-root "$GSI_DATASET_ROOT"
```

## Output Usage

Collected results can be converted into training parquet files and state shards. RLVR expects the structure described in [RLVR Training](../training/rlvr.md).
