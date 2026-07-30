# Replan and New Case

Replan and new case evaluate how the system recovers under dynamic task conditions. Standard benchmarks can run without new cases; robustness evaluation must enable them explicitly.

## New Case

A new case is an incident introduced during execution, such as a goal change, robot failure, blocked route, or environment change.

Related paths:

```text
modules/platform/semantic_platform/new_case_generator.py
modules/platform/semantic_platform/new_case_injector.py
modules/platform/semantic_platform/new_case_controller.py
```

Enable in benchmark:

```bash
python run/run_exp_multi_method.py \
  --methods sgi \
  --enable-newcase \
  --newcase-counts 1 2 3 4
```

## Replan

Replan means regenerating or adjusting the plan based on execution feedback. Common triggers:

- Main goal is not completed.
- A skill execution fails.
- A new case changes task conditions.
- Validator or goal monitor marks the current plan as invalid.

Configuration example:

```json
{
  "enable_replanning": true,
  "enable_new_case_generation": false
}
```

## Data Collection

Collect replan data:

```bash
python run/run_collect_replan_dataset.py --help
```

Collected results can be converted into SFT/RLVR data. Training-side details are in [Hugging Face Preparation](../training/huggingface-prepare.md) and [RLVR Training](../training/rlvr.md).
