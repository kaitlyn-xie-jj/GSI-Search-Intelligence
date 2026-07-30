# Solver

GSI 的 TANGO allocator 支持 SCIP 和 Gurobi 后端。默认推荐使用 SCIP；Gurobi 可用于对求解耗时敏感的评估，但需要有效 license。

## 选择后端

```bash
export GSI_TANGO_SOLVER_BACKEND=scip
```

使用 Gurobi：

```bash
export GSI_TANGO_SOLVER_BACKEND=gurobi
```

## Gurobi License

Gurobi 是商业优化器。学术用户可申请 academic license；非学术或生产用途应遵循商业授权。申请和 license 类型以 Gurobi 官方说明为准：

- [Gurobi Academic License Program](https://www.gurobi.com/academics)
- [How do I obtain a free academic license?](https://support.gurobi.com/hc/en-us/articles/360040541251-How-do-I-obtain-a-free-academic-license)
- [How do I retrieve an Academic Named-User license?](https://support.gurobi.com/hc/en-us/articles/13207658935185-How-do-I-retrieve-an-Academic-Named-User-license)
- [How do I use Gurobi on multiple computers for my academic research?](https://support.gurobi.com/hc/en-us/articles/360040995151-How-do-I-use-Gurobi-on-multiple-computers-for-my-academic-research)

常见选择：

- 单机固定工作站：Named-User license。
- Docker、多机器或云环境：WLS license。
- 只需跑通 GSI：使用 SCIP。

`gurobi.lic`、WLS credentials 和 license key 不应提交到仓库。

## 在 GSI 中使用 Gurobi

```bash
export GRB_LICENSE_FILE=/path/to/gurobi.lic
export GSI_TANGO_SOLVER_BACKEND=gurobi
```

容器中可将 license 挂载为只读文件：

```bash
docker run --rm \
  -v /host/path/gurobi.lic:/licenses/gurobi.lic:ro \
  -e GRB_LICENSE_FILE=/licenses/gurobi.lic \
  ...
```

如果 license 不可用，保持：

```bash
export GSI_TANGO_SOLVER_BACKEND=scip
```

## 快速自检

确认 Python 包可导入：

```bash
python - <<'PY'
import gurobipy as gp
print(gp.gurobi.version())
PY
```

确认 license 可用：

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

如果自检失败，应先修复 Gurobi 安装或 license，再运行 GSI benchmark。

## 求解超时

Validator timeout 频繁出现时：

1. 确认 solver 后端可用。
2. 降低训练 rollout 并发。
3. 提高 validator/reward HTTP timeout。
4. 抽样复现单条 state，确认不是具体任务不可解。

训练参数见 [RLVR 训练](../training/rlvr.md)。
