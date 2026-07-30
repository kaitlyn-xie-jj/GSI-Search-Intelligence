# 新增 Solver 后端

GSI 的任务分配通过 TANGO allocator 调用优化求解器。当前常用后端为 SCIP 和 Gurobi。

## 关键路径

```text
modules/task_solver/sgi_planner/allocator/tango/
modules/task_solver/sgi_planner/allocator/tango/algorithm/solver_backend.py
```

## 接入步骤

1. 在 solver backend 中新增后端名称或枚举。
2. 实现与现有 allocator 兼容的求解接口。
3. 保持输入变量、约束和目标函数语义不变。
4. 暴露环境变量或配置开关。
5. 使用小任务对比新后端与 SCIP/Gurobi 的结果。

## 运行切换

```bash
export GSI_TANGO_SOLVER_BACKEND=scip
```

新增后端应复用该环境变量入口，避免 benchmark、validator 和训练脚本出现不同配置路径。
