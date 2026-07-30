# Planner / Validator / Solver

GSI 将计划生成、计划验证和任务分配拆分为三个职责，避免让 LLM 同时承担格式生成、约束检查和优化求解。

## Planner

Planner 将任务指令、环境状态、技能库和执行反馈转换为任务图或计划候选。

主要路径：

```text
modules/task_solver/sgi_planner/
modules/task_solver/baseline_planners/
```

SGI planner 主要生成结构化任务图；baseline planner 采用各自的 prompt 和规划策略，但共享统一的运行生命周期。

## Validator

Validator 检查计划格式、技能合法性、状态约束和 reward 所需条件。RLVR 训练也复用 validator 逻辑。

主要路径：

```text
modules/plan_validator/
run/plan_validation_server.py
llm_finetune/rlvr/gsi_reward_manager.py
```

训练容器中常用启动方式：

```bash
./scripts/runtime/serve_validator.sh start
```

## Solver

Solver 在 GSI 中主要指 TANGO allocator 使用的底层优化后端。它负责把 ready tasks 分配给具体机器人。

主要路径：

```text
modules/task_solver/sgi_planner/allocator/tango/
modules/task_solver/sgi_planner/allocator/tango/algorithm/solver_backend.py
```

常用配置：

```bash
export GSI_TANGO_SOLVER_BACKEND=scip
```

## 容易混淆的名称

- `--methods sgi` 或 `solver_type=sgi`：选择 planner 方法。
- `GSI_TANGO_SOLVER_BACKEND=scip/gurobi`：选择任务分配优化后端。

Solver 排查见 [Solver](../troubleshooting/solver.md)。
