# Repository Layout

The GSI repository can be understood through the runtime flow: configuration, data, core modules, runtime entry points, and training scripts.

```text
config/        Default runtime configuration
docs/          Documentation site content
modules/       Planner, validator, solver, world model, and platform
run/           Benchmark, evaluation, and data collection entry points
llm_finetune/  SFT, RLVR, vLLM, VeRL, and training container scripts
```

## Core Modules

```text
modules/task_solver/
```

Main task-solving path, including the SGI planner, baseline planners, solver factory, and world model.

```text
modules/plan_validator/
```

Plan validation, replan state loading, and RLVR reward-related logic.

```text
modules/platform/
```

Execution abstraction for the semantic platform and Unreal platform.

```text
modules/dataset_builder/
modules/dataset_loader/
```

Data construction, loading, and filtering logic.

## Runtime Entry Points

```text
run/run_exp_multi_method.py
run/run_collect_replan_dataset.py
run/plan_validation_server.py
```

## Training Entry Points

```text
llm_finetune/scripts/runtime/train_sft_unsloth.sh
llm_finetune/scripts/runtime/train_rlvr.sh
llm_finetune/scripts/runtime/serve_validator.sh
```

Training workflow is covered in [Training Overview](../training/overview.md).

## Change Guidelines

- Add new planners through the shared lifecycle in `modules/task_solver/`.
- Add validator rules while checking both benchmark and RLVR reward paths.
- Add task data with consistent goal, scenario, success condition, and filters.
- Update documentation when public CLI arguments, data formats, or model repositories change.
