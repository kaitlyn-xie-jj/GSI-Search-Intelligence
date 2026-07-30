# Task and State

GSI runtime objects consist of scenario, goal, task, and state. Together they define task input, execution conditions, validation context, and training sample context.

## Scenario

Scenario describes the initial environment state, usually including:

- Robot types, locations, and capabilities.
- Regions, roads, buildings, and points of interest.
- Objects, events, and interactive entities.
- Spatial relations between entities.

Related paths:

```text
modules/dataset_builder/generate_scenarios.py
modules/dataset_builder/scene_utils/
modules/dataset_loader/
```

## Goal

Goal describes the task objective and success conditions. Common types include `search`, `transport`, `assembly`, `patrol`, `guidance`, `verbal broadcast`, and `traffic enforcement`.

Common fields:

- `instruction`: natural-language instruction for the model or planner.
- `goal_details`: structured goal description.
- `success_condition`: condition used to determine success.
- `meta`: planning, coordination, and language difficulty tags.

## Task

Task combines scenario and goal. The benchmark filters task ids from the dataset and runs the same task set with one or more methods.

Entry point:

```bash
python run/run_exp_multi_method.py --help
```

## State

State represents runtime world state and replan context. RLVR state shards are loaded by the validator to reproduce plan validation and reward computation.

Typical RLVR structure:

```text
train.parquet
val.parquet
states.index.json
states/*.jsonl
```

Training details are in [RLVR Training](../training/rlvr.md).
