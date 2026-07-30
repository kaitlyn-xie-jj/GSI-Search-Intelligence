from __future__ import annotations

from itertools import product
from typing import Any

from pyscipopt import Model as ScipModel
from pyscipopt import quicksum as scip_quicksum


class _GRB:
    BINARY = "B"
    CONTINUOUS = "C"
    INTEGER = "I"
    MINIMIZE = 1
    MAXIMIZE = -1
    OPTIMAL = 2
    INFEASIBLE = 3
    INF_OR_UNBD = 4
    TIME_LIMIT = 9

    class Attr:
        NumConstrs = "NumConstrs"
        SolCount = "SolCount"


GRB = _GRB()


class Env:
    def __init__(self):
        self.params: dict[str, Any] = {}

    def setParam(self, name, value):
        self.params[name] = value

    def dispose(self):
        return None


def _unwrap(value):
    if isinstance(value, (Expr, Var)):
        return value._raw
    return value


class Expr:
    def __init__(self, raw):
        self._raw = raw

    def __add__(self, other):
        return Expr(self._raw + _unwrap(other))

    def __radd__(self, other):
        return Expr(_unwrap(other) + self._raw)

    def __sub__(self, other):
        return Expr(self._raw - _unwrap(other))

    def __rsub__(self, other):
        return Expr(_unwrap(other) - self._raw)

    def __mul__(self, other):
        return Expr(self._raw * _unwrap(other))

    def __rmul__(self, other):
        return Expr(_unwrap(other) * self._raw)

    def __neg__(self):
        return Expr(-self._raw)

    def __le__(self, other):
        return self._raw <= _unwrap(other)

    def __ge__(self, other):
        return self._raw >= _unwrap(other)

    def __eq__(self, other):
        return self._raw == _unwrap(other)


def LinExpr(value=0.0):
    return Expr(float(value))


class Var:
    def __init__(self, model: Model, raw_var, name: str, obj: float = 0.0):
        self._model = model
        self._raw = raw_var
        self._obj = float(obj)
        self._name = name

    @property
    def X(self) -> float:
        if self._model._solution is None:
            return 0.0
        return float(self._model._raw.getSolVal(self._model._solution, self._raw))

    @property
    def Obj(self) -> float:
        return float(self._obj)

    @property
    def VarName(self) -> str:
        return self._name

    def setAttr(self, name: str, value):
        if name == "Obj":
            self._obj = float(value)
            self._model._objective_dirty = True
            return
        if name == "LB":
            self._model._raw.chgVarLb(self._raw, float(value))
            return
        if name == "UB":
            self._model._raw.chgVarUb(self._raw, float(value))
            return
        if name == "VarName":
            self._name = str(value)
            return
        raise AttributeError(name)

    def __add__(self, other):
        return Expr(self._raw + _unwrap(other))

    def __radd__(self, other):
        return Expr(_unwrap(other) + self._raw)

    def __sub__(self, other):
        return Expr(self._raw - _unwrap(other))

    def __rsub__(self, other):
        return Expr(_unwrap(other) - self._raw)

    def __mul__(self, other):
        return Expr(self._raw * _unwrap(other))

    def __rmul__(self, other):
        return Expr(_unwrap(other) * self._raw)

    def __neg__(self):
        return Expr(-self._raw)

    def __le__(self, other):
        return self._raw <= _unwrap(other)

    def __ge__(self, other):
        return self._raw >= _unwrap(other)

    def __eq__(self, other):
        return self._raw == _unwrap(other)


class VarDict(dict):
    pass


class Model:
    def __init__(self, name: str | None = None, env: Env | None = None):
        self._raw = ScipModel(name or "model")
        self._vars: list[Var] = []
        self._constraints = []
        self._solution = None
        self._objective_dirty = True
        self._model_sense = GRB.MINIMIZE
        self._runtime = 0.0
        if env and int(env.params.get("OutputFlag", 1)) == 0:
            self._raw.hideOutput()

    def setParam(self, name, value):
        if name == "TimeLimit":
            self._raw.setParam("limits/time", float(value))
        elif name == "OutputFlag" and int(value) == 0:
            self._raw.hideOutput()
        elif name == "LazyConstraints":
            return None

    def getParam(self, name):
        if name == "LazyConstraints":
            return 0
        return None

    def addVar(self, lb=0.0, ub=None, obj=0.0, vtype=GRB.CONTINUOUS, name=""):
        raw = self._raw.addVar(
            name=name,
            lb=float(lb),
            ub=None if ub is None else float(ub),
            vtype="B" if vtype == GRB.BINARY else ("I" if vtype == GRB.INTEGER else "C"),
        )
        var = Var(self, raw, name, obj)
        self._vars.append(var)
        self._objective_dirty = True
        return var

    def addVars(self, *dims, vtype=GRB.CONTINUOUS, lb=0.0, ub=None, name=""):
        result = VarDict()
        if len(dims) == 1:
            keys = range(int(dims[0]))
        else:
            keys = product(*[range(int(dim)) for dim in dims])
        for key in keys:
            display_key = key if isinstance(key, tuple) else (key,)
            var_name = f"{name}[{','.join(map(str, display_key))}]"
            result[key] = self.addVar(lb=lb, ub=ub, vtype=vtype, name=var_name)
        return result

    def addConstr(self, expr, name=""):
        cons = self._raw.addCons(expr, name=name)
        self._constraints.append(cons)
        return cons

    def setObjective(self, expr, sense=GRB.MINIMIZE):
        self._raw.setObjective(_unwrap(expr), "minimize" if sense == GRB.MINIMIZE else "maximize")
        self._objective_dirty = False

    def setAttr(self, name, value):
        if name == "ModelSense":
            self._model_sense = value
            self._objective_dirty = True
            return
        raise AttributeError(name)

    def _sync_objective(self):
        if not self._objective_dirty:
            return
        expr = scip_quicksum(var.Obj * var._raw for var in self._vars)
        self._raw.setObjective(expr, "minimize" if self._model_sense == GRB.MINIMIZE else "maximize")
        self._objective_dirty = False

    def optimize(self):
        self._sync_objective()
        self._raw.optimize()
        self._solution = self._raw.getBestSol()
        try:
            self._runtime = float(self._raw.getSolvingTime())
        except Exception:
            self._runtime = 0.0

    def update(self):
        return None

    def getAttr(self, attr):
        if attr == GRB.Attr.NumConstrs:
            return len(self._constraints)
        if attr == GRB.Attr.SolCount:
            return 1 if self._solution is not None else 0
        raise AttributeError(attr)

    @property
    def Status(self):
        status = str(self._raw.getStatus()).lower()
        if status == "optimal":
            return GRB.OPTIMAL
        if status == "infeasible":
            return GRB.INFEASIBLE
        if status == "timelimit":
            return GRB.TIME_LIMIT
        return GRB.INFEASIBLE

    @property
    def SolCount(self):
        return 1 if self._solution is not None else 0

    @property
    def ObjVal(self):
        return float(self._raw.getObjVal()) if self._solution is not None else 0.0

    @property
    def MIPGap(self):
        if self._solution is None:
            return float("inf")
        try:
            return float(self._raw.getGap())
        except Exception:
            return 0.0

    @property
    def ObjBound(self):
        if self._solution is None:
            return 0.0
        try:
            return float(self._raw.getDualbound())
        except Exception:
            return self.ObjVal

    @property
    def NumVars(self):
        return len(self._vars)

    @property
    def NumConstrs(self):
        return len(self._constraints)

    @property
    def NodeCount(self):
        try:
            return int(self._raw.getNNodes())
        except Exception:
            return 0

    @property
    def IterCount(self):
        try:
            return int(self._raw.getNLPIterations())
        except Exception:
            return 0

    @property
    def Runtime(self):
        return self._runtime

    def dispose(self):
        try:
            self._raw.freeProb()
        except Exception:
            return None


def quicksum(items):
    return Expr(scip_quicksum(_unwrap(item) for item in items))
