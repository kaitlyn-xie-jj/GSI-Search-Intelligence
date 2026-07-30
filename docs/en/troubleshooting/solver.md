# Solver

The GSI TANGO allocator supports SCIP and Gurobi backends. SCIP is the default recommendation. Gurobi can reduce solve time but requires a valid license.

## Select Backend

```bash
export GSI_TANGO_SOLVER_BACKEND=scip
```

Use Gurobi:

```bash
export GSI_TANGO_SOLVER_BACKEND=gurobi
```

## Gurobi License

Gurobi is commercial software. Academic users may apply for an academic license. Non-academic or production use must follow commercial licensing. Use Gurobi official documentation for current license rules:

- [Gurobi Academic License Program](https://www.gurobi.com/academics)
- [How do I obtain a free academic license?](https://support.gurobi.com/hc/en-us/articles/360040541251-How-do-I-obtain-a-free-academic-license)
- [How do I retrieve an Academic Named-User license?](https://support.gurobi.com/hc/en-us/articles/13207658935185-How-do-I-retrieve-an-Academic-Named-User-license)
- [How do I use Gurobi on multiple computers for my academic research?](https://support.gurobi.com/hc/en-us/articles/360040995151-How-do-I-use-Gurobi-on-multiple-computers-for-my-academic-research)

Common choices:

- Fixed single workstation: Named-User license.
- Docker, multi-machine, or cloud environment: WLS license.
- Minimal GSI run: SCIP.

`gurobi.lic`, WLS credentials, and license keys must not be committed to the repository.

## Use Gurobi in GSI

```bash
export GRB_LICENSE_FILE=/path/to/gurobi.lic
export GSI_TANGO_SOLVER_BACKEND=gurobi
```

For containers, mount the license as read-only:

```bash
docker run --rm \
  -v /host/path/gurobi.lic:/licenses/gurobi.lic:ro \
  -e GRB_LICENSE_FILE=/licenses/gurobi.lic \
  ...
```

If the license is unavailable, keep:

```bash
export GSI_TANGO_SOLVER_BACKEND=scip
```

## Quick Self-Check

Check import:

```bash
python - <<'PY'
import gurobipy as gp
print(gp.gurobi.version())
PY
```

Check license:

```bash
python - <<'PY'
import gurobipy as gp
m = gp.Model()
x = m.addVar(lb=0.0, name="x")
m.setObjective(x, gp.GRB.MAXIMIZE)
m.addConstr(x <= 1.0)
m.optimize()
print("status", m.Status, "obj", m.ObjVal if m.SolCount else None)
PY
```

If self-check fails, fix Gurobi installation or license before running GSI benchmark.

## Solver Timeout

When validator timeout is frequent:

1. Confirm the solver backend is usable.
2. Lower training rollout concurrency.
3. Increase validator/reward HTTP timeout.
4. Reproduce one sampled state to rule out task-specific infeasibility.

Training parameters are in [RLVR Training](../training/rlvr.md).
