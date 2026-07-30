# Add a Solver Backend

GSI task allocation calls optimization solvers through the TANGO allocator. Common backends are SCIP and Gurobi.

## Key Paths

```text
modules/task_solver/sgi_planner/allocator/tango/
modules/task_solver/sgi_planner/allocator/tango/algorithm/solver_backend.py
```

## Integration Steps

1. Add the backend name or enum in the solver backend.
2. Implement a solver interface compatible with the existing allocator.
3. Preserve the semantics of input variables, constraints, and objectives.
4. Expose the backend through an environment variable or configuration switch.
5. Compare the new backend against SCIP/Gurobi on small tasks.

## Runtime Switch

```bash
export GSI_TANGO_SOLVER_BACKEND=scip
```

New backends should reuse this environment variable so benchmark, validator, and training scripts share one configuration path.
