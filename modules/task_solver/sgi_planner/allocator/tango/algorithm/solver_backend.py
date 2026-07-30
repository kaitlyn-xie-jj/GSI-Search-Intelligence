import os
import importlib.util
from pathlib import Path


def backend_name() -> str:
    return os.environ.get("GSI_TANGO_SOLVER_BACKEND", "scip").strip().lower() or "scip"


if backend_name() == "scip":
    try:
        from . import scip_compat as gp
    except ImportError:
        module_path = Path(__file__).with_name("scip_compat.py")
        spec = importlib.util.spec_from_file_location("gsi_tango_scip_compat", module_path)
        if spec is None or spec.loader is None:
            raise
        gp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gp)
else:
    import gurobipy as gp


GRB = gp.GRB
Model = gp.Model
Env = gp.Env
quicksum = gp.quicksum
Var = getattr(gp, "Var", object)
