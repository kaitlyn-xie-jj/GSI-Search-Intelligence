# Runtime Configuration Overview

This document briefly describes the core configurable parameters for system runtime.

For complete configuration behavior, common runtime recipes, and CLI/env override rules, see [`../docs/runtime_config.md`](../docs/runtime_config.md).

---

## 🌐 Planner Settings

### `planner_mode`
- `"full"`: Generate complete plan at once, including tasks with unclear parameters.
- `"phase"`: Generate phase plan with only explicit parameter tasks (deprecated).

### `default_robot_types`
List of robot types supported by the system by default.

---

## 🖥 Runtime / Debug Settings

### `enable_detailed_print`
Enable more detailed log output (for debugging).

### `simulate_time_delay`
Whether to simulate skill execution time delays. Set to `false` for batch testing.

### `enable_visualization`
Whether to enable visualization window.

### `enable_video_recording`
Whether to record execution video. Visualization must be enabled to record video.

### `fine_grained_simulation`
Enable fine-grained simulation steps. This determines whether lightweight physics simulation is enabled. Recommended to enable.

### `max_concurrency`
Maximum number of concurrent executions allowed.

### `enable_logging`
Whether to save runtime logs.

### `enable_checkpointing`
Whether to save state snapshots at key steps (for recovery/replay).

---

## 🔁 Replanning & Dataset

### `enable_replanning`
Whether to enable replanning functionality. When enabled, the system will attempt to replan when unexpected events occur or execution fails.

### `collect_replan_dataset`
Whether to collect replanning scenario datasets for fine-tuning. Disabled by default. If enabled, `enable_new_case_generation` must also be enabled, and `max_newcases_per_run` should be set to at least 1.

---

## ⚠️ New Case Generation

### `enable_new_case_generation`
Whether to enable new case (unexpected event) generation.

### `max_newcases_per_run`
Maximum number of new cases allowed per task execution. Only effective when `enable_new_case_generation` is enabled.

### `new_case_mode`
- Only effective when `enable_new_case_generation` is enabled.
- `"aggregate"`: Some strategies become "non-rollback" aggregated processing, i.e., no restoration of previous graph operations. Prechecks validate all conditions before reporting failed events collectively.
- `"immediate"`: Execute with original template rollback behavior, i.e., restore previous graph operations. Prechecks report events immediately when any condition fails.

### `newcase_spacing_factor`
Geometric distribution parameter controlling new case injection intervals. Higher values result in sparser injection. System calculates dynamic window via `window = remaining_injectable × spacing_factor`, then uses `p = remaining / window` as current round injection probability. Minimum value is `1.0`, which gives highest injection probability per round. Only effective when `enable_new_case_generation` is enabled.

### `newcase_cooldown_rounds`
Cooldown rounds after each successful new case injection. During cooldown, injection probability is forced to `0` to prevent consecutive dense injections. Set to `0` for no cooldown. Only effective when `enable_new_case_generation` is enabled.

### `newcase_similarity_threshold`
Jaccard similarity threshold between adjacent round plans. When skill fingerprint similarity between consecutive plans exceeds this threshold, the system considers plan changes minimal and applies damping (multiplies by `newcase_similarity_damping`) to injection probability, avoiding repeated injection in highly similar plans. Range `[0, 1]`, lower values trigger damping more easily. Only effective when `enable_new_case_generation` is enabled.

### `newcase_similarity_damping`
Damping coefficient for injection probability when plan similarity exceeds `newcase_similarity_threshold`. For example, setting to `0.3` reduces injection probability to 30% of original. Range `[0, 1]`, lower values mean stronger damping. Only effective when `enable_new_case_generation` is enabled.

---

## 📦 Dataset & Platform

### `repo_id`
Dataset repository ID used to identify remote repository address when uploading/downloading datasets.

### `platform_type`
Runtime platform type. `"semantic"` for semantic simulation platform (lightweight), `"unreal"` for Unreal Engine simulation platform.

### `unreal_platform`
Unreal Engine platform connection configuration. Only effective when `platform_type` is `"unreal"`.
- `base_url`: HTTP address of UE5 simulation service.
- `timeout`: Single request timeout (seconds).
- `polling_interval`: Polling interval for execution status (seconds).

---

## 📼 Replay Mode

Used to replay existing trajectories without executing actual planning process.

### `replay_mode.enabled`
Whether to enable replay mode. When enabled, allows complete reproduction of a previous experiment.

### `replay_mode.trace_root`
Root directory of replay trajectory files, i.e., directory containing `temp_var.json`. Only effective when `replay_mode.enabled` is enabled.

### `replay_mode.trace_tag`
Tag used for replay (to distinguish multiple trajectories). Only effective when `replay_mode.enabled` is enabled.

---

## 🧑‍💻 Human-in-the-Loop (HITL)

Human-in-the-loop interaction configuration for introducing human operator instruction input, plan review, and decision intervention during execution.

### `human_in_loop.enabled`
HITL master switch. When set to `false`, all sub-features below are disabled and the system runs fully autonomously.

### `human_in_loop.instruction_enabled`
Whether to enable instruction input functionality. When enabled, the system requests operator instruction input from UE5 before task starts. Requires `enabled` to be `true`.

### `human_in_loop.review_enabled`
Whether to enable plan review functionality. When enabled, system-generated plans are sent to operator for review and modification before continuing execution. Requires `enabled` to be `true`.

### `human_in_loop.decision_enabled`
Whether to enable decision request functionality. When enabled, the system requests operator decisions when ambiguity or human judgment is needed (e.g., search target not found). Requires `enabled` to be `true`.

### `human_in_loop.instruction_timeout`
Timeout for waiting for operator instruction input (seconds). After timeout, system continues with preloaded instruction.

### `human_in_loop.review_timeout`
Timeout for waiting for operator to complete plan review (seconds). After timeout, system continues with original plan.

### `human_in_loop.decision_timeout`
Timeout for waiting for operator decision (seconds). After timeout, system continues with default decision (`end_task`).

### `human_in_loop.server_port`
Python-side HTTP server port for HITL message communication with UE5.

### `human_in_loop.retry_count`
Number of retries when HITL operation fails.

### `human_in_loop.retry_delay`
Wait interval between retries (seconds).

---

## 🧩 Solver Configuration

### `solver_type`
Current solver type in use. Options: `"sgi"`, `"llamar"`, `"spine"`, `"lipllm"`, `"smartllm"`. System reads corresponding configuration block from `solver_config` based on this value.

### `solver_config`
Independent configuration for each solver, indexed by solver name. Below are common and solver-specific parameter descriptions.

#### Common Parameters (supported by all solvers)

| Parameter | Description |
|------|------|
| `max_steps` | Maximum execution steps (upper limit of plan-execute loop rounds). |
| `model_family` | LLM model family to use. Uses system default model when set to `null`. |
| `model_name_override` | Override model name. Uses default model for `model_family` when set to `null`. |

#### `sgi` Specific Parameters

| Parameter | Description |
|------|------|
| `validate_plan` | Whether to validate generated plans. |

#### `llamar` Specific Parameters

| Parameter | Description |
|------|------|
| `use_few_shot` | Whether to use few-shot examples in prompt to guide LLM generation. |

#### `spine` Specific Parameters

| Parameter | Description |
|------|------|
| `use_few_shot` | Whether to use few-shot examples. |
| `n_attempts` | Maximum attempts per planning round. Retries on generation failure. |

#### `lipllm` Specific Parameters

| Parameter | Description |
|------|------|
| `use_few_shot` | Whether to use few-shot examples. |
| `n_attempts` | Maximum attempts per planning round. |
| `max_iterations` | Maximum rounds of LipLLM internal iterative optimization. |
| `alpha` | Step size/weight coefficient in iterative optimization, controlling update magnitude per round. |

#### `smartllm` Specific Parameters

| Parameter | Description |
|------|------|
| `use_few_shot` | Whether to use few-shot examples. |

---

For extensions or specific usage details, refer to the implementation documentation of each module.
