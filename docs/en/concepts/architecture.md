# Architecture

GSI places natural-language tasks into an executable, verifiable, and comparable multi-robot system. The system handles scene state, robot capabilities, task dependencies, plan format, execution feedback, new cases, and replanning.

## Execution Flow

```text
task id
  -> DatasetLoader reads goal / scenario / metadata
  -> UnifiedTaskSolver initializes world model, platform, and planner
  -> Planner calls the LLM to generate a task graph or baseline plan
  -> TaskGraphManager tracks dependencies and ready tasks
  -> TANGO allocator assigns robots
  -> plan translator converts the plan into timestep skills
  -> semantic platform executes skills
  -> world model updates state
  -> feedback processor decides success, failure, or replan
  -> batch runner writes summary and aggregate outputs
```

Main entry points:

- `run/run_exp_multi_method.py`: multi-method benchmark and new-case evaluation.
- `run/utils/case_runner.py`: single-task runtime wrapper.
- `run/utils/batch_runner.py`: batch execution and aggregate output.

## UnifiedTaskSolver

`UnifiedTaskSolver` manages the lifecycle of a single task. It does not implement a specific planner or execute low-level skills directly. It coordinates components in order:

1. Initialize context, world model, platform executor, and planner.
2. Call the planner to generate a plan.
3. Convert the plan into executable skill sequences.
4. Execute skills and collect outcomes.
5. Update world model and metrics.
6. Decide whether to continue, finish, or replan.

Key paths:

```text
modules/task_solver/unified_task_solver.py
run/utils/case_runner.py
run/utils/batch_runner.py
```

The shared lifecycle allows SGI, SPINE, SmartLLM, LipLLM, and other methods to be compared under the same task set and metric system.

## Planner Layer

`solver_type` or `--methods` selects the planner method. Common values:

```text
sgi
spine
smartllm
lipllm
```

The SGI planner asks the LLM to generate a structured task graph rather than each robot's low-level action sequence. The task graph usually contains task names, semantic goals, dependencies, capability requirements, and structured fields consumed by later modules.

Related paths:

```text
modules/task_solver/sgi_planner/planning_layer.py
modules/task_solver/sgi_planner/plan_module.py
modules/task_solver/sgi_planner/actions/plan_task.py
modules/task_solver/sgi_planner/task_graph_manager.py
modules/task_solver/baseline_planners/
```

## Allocator / Solver Layer

The TANGO allocator assigns ready tasks to robots. The problem includes capability constraints, spatial cost, and resource limits, so it is not handled purely through LLM text generation.

```text
ready tasks + robot states + capability constraints
  -> TANGO allocator
  -> allocation result
```

Related paths:

```text
modules/task_solver/sgi_planner/alloc_module.py
modules/task_solver/sgi_planner/allocator/
modules/task_solver/sgi_planner/allocator/tango/
```

Naming boundary:

- `--methods sgi` selects the planner method.
- `GSI_TANGO_SOLVER_BACKEND=scip/gurobi` selects the allocator optimization backend.

## World Model Layer

The world model maintains the system's current understanding of the environment and converts execution outcomes into state for later planning.

Responsibilities:

- Initialize robot, goal, and object state from the scene graph.
- Receive skill execution outcomes.
- Update local world state.
- Determine whether goals are completed.
- Construct feedback and state for replanning.

Related paths:

```text
modules/task_solver/world_model/world_model_layer.py
modules/task_solver/world_model/world_model_manager.py
modules/task_solver/world_model/goal_progress_monitor.py
modules/task_solver/world_model/status_tracker.py
```

## Platform Layer

The platform executor receives skill sequences and executes them. Large-scale benchmarks use the semantic platform by default:

- It does not require UE.
- It executes quickly.
- It produces structured outcomes, newcase events, and metrics.

Related paths:

```text
modules/platform/semantic_platform/platform_executor.py
modules/platform/semantic_platform/skill_executor.py
modules/platform/semantic_platform/new_case_controller.py
modules/platform/semantic_platform/new_case_generator.py
modules/platform/semantic_platform/new_case_injector.py
modules/platform/unreal_platform/
```

The Unreal platform is mainly for UE5 integration or demonstrations. It is not the default training or benchmark path.

## Feedback / Replan Layer

Execution may produce skill failures, partial goal completion, pending tasks, new cases, or external feedback. The feedback processor converts these signals into replanning strategies:

- `NONE`: no replanning.
- `PARTIAL`: keep part of the task graph and adjust affected parts.
- `FULL`: rebuild the prompt and generate a complete task graph.

Related paths:

```text
modules/task_solver/sgi_planner/feedback_processor.py
modules/task_solver/sgi_planner/base_feedback_processor.py
modules/utils/replan_recorder.py
```

RLVR replan samples also come from this flow and include state, newcase, feedback, and validator/reward information.

## Validator / Reward Layer

RLVR training uses the validator/reward manager to check whether model outputs satisfy format, task constraints, and executability requirements.

Related paths:

```text
modules/plan_validator/plan_validator.py
modules/plan_validator/replan_state_store.py
llm_finetune/rlvr/gsi_reward_manager.py
llm_finetune/rlvr/build_replan_state_store.py
```

Benchmark execution evaluates how a planner performs in tasks. RLVR reward evaluates whether model output is verifiable, executable, and scoreable under a given state. They share task and state semantics but serve different stages.

## Metrics and Outputs

Batch execution is managed by `ParallelBatchRunner`. Each task produces an individual run directory. At the end of a batch, the runner writes:

- `summary.jsonl`
- `aggregate_full.json`

Aggregation logic:

```text
run/utils/batch_runner.py
run/utils/analysis.py
```

Common metrics include `success_rate`, `llm_calls`, `replans_total`, `total_energy`, `newcase_total`, `planning_duration`, `allocation_duration`, and token statistics. See [Output Directory](../training/outputs.md) for details.

## Recommended Code Reading Order

1. `run/run_exp_multi_method.py`
2. `run/utils/batch_runner.py`
3. `run/utils/case_runner.py`
4. `modules/task_solver/unified_task_solver.py`
5. `modules/task_solver/sgi_planner/planning_layer.py`
6. `modules/task_solver/sgi_planner/alloc_module.py`
7. `modules/platform/semantic_platform/skill_executor.py`
8. `modules/task_solver/sgi_planner/feedback_processor.py`
