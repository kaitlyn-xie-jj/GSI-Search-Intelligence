# Ray and Checkpoint

RLVR training depends on Ray, VeRL, and vLLM. After abnormal termination, old processes and old checkpoints are common failure sources.

## Stop Old Ray Runtime

Enable preflight before training:

```bash
GSI_PREFLIGHT=1
GSI_PREFLIGHT_STOP_RAY=1
```

`train_rlvr.sh` will attempt to stop old Ray runtime during preflight.

## Automatic Checkpoint Resume

VeRL uses `trainer.resume_mode=auto` by default. If `RLVR_OUTPUT_DIR` contains `global_step_*`, training resumes automatically.

Log example:

```text
Found checkpoint: .../global_step_1800
Load from checkpoint folder: .../global_step_1800
Setting global step to 1800
Resuming from .../global_step_1800
```

## Use a New Directory for New Experiments

After changing batch, rollout, epoch, or dataset settings, use a new output directory:

```bash
export RLVR_OUTPUT_DIR=/GSI/outputs/rlvr_gsi/qwen3_0_6b_cybertown_rlvr_bsz8
```

Force ignoring old checkpoints:

```bash
trainer.resume_mode=disable
```

See [RLVR Training](../training/rlvr.md) for details.
