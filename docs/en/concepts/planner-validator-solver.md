# Planner / Validator / Solver

GSI separates plan generation, plan validation, and task allocation. This avoids placing format generation, constraint checking, and optimization on the LLM alone.

## Planner

The planner converts task instructions, environment state, skill libraries, and execution feedback into a task graph or candidate plan.

Main paths:

```text
modules/task_solver/sgi_planner/
modules/task_solver/baseline_planners/
```

The SGI planner generates a structured task graph. Baseline planners use their own prompts and strategies while sharing the same task lifecycle.

## Validator

The validator checks plan format, skill validity, state constraints, and reward conditions. RLVR training also reuses validator logic.

Main paths:

```text
modules/plan_validator/
run/plan_validation_server.py
llm_finetune/rlvr/gsi_reward_manager.py
```

In the training container:

```bash
./scripts/runtime/serve_validator.sh start
```

## Solver

In GSI, solver usually refers to the optimization backend used by the TANGO allocator. It assigns ready tasks to concrete robots.

Main paths:

```text
modules/task_solver/sgi_planner/allocator/tango/
modules/task_solver/sgi_planner/allocator/tango/algorithm/solver_backend.py
```

Common configuration:

```bash
export GSI_TANGO_SOLVER_BACKEND=scip
```

## Naming Boundaries

- `--methods sgi` or `solver_type=sgi` selects the planner method.
- `GSI_TANGO_SOLVER_BACKEND=scip/gurobi` selects the task allocation optimization backend.

Solver troubleshooting is in [Solver](../troubleshooting/solver.md).
