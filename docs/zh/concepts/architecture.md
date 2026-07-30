# 系统架构

GSI 将自然语言任务放入可执行、可验证、可比较的多机器人系统中。系统需要同时处理场景状态、机器人能力、任务依赖、计划格式、执行反馈、new case 和重规划。

## 执行链路

```text
task id
  -> DatasetLoader 读取 goal / scenario / metadata
  -> UnifiedTaskSolver 初始化 world model、platform 和 planner
  -> Planner 调用 LLM 生成任务图或 baseline plan
  -> TaskGraphManager 跟踪依赖和 ready tasks
  -> TANGO allocator 分配机器人
  -> plan translator 转换为 timestep skills
  -> semantic platform 执行技能
  -> world model 更新状态
  -> feedback processor 判断成功、失败或 replan
  -> batch runner 写出 summary 和 aggregate
```

主要入口：

- `run/run_exp_multi_method.py`：多方法 benchmark 与 newcase 评估。
- `run/utils/case_runner.py`：单任务运行封装。
- `run/utils/batch_runner.py`：批量运行和聚合输出。

## UnifiedTaskSolver

`UnifiedTaskSolver` 管理单个 task 的生命周期。它不直接实现具体 planner，也不执行底层技能；它负责按顺序协调组件：

1. 初始化 context、world model、platform executor 和 planner。
2. 调用 planner 生成计划。
3. 将计划转换为可执行技能序列。
4. 执行技能并收集 outcomes。
5. 更新 world model 和 metrics。
6. 根据反馈决定继续、结束或重规划。

关键路径：

```text
modules/task_solver/unified_task_solver.py
run/utils/case_runner.py
run/utils/batch_runner.py
```

统一生命周期使 SGI、SPINE、SmartLLM、LipLLM 等方法能够在同一任务集合和指标体系下比较。

## Planner 层

`solver_type` 或 `--methods` 选择 planner 方法，常见值为：

```text
sgi
spine
smartllm
lipllm
```

SGI planner 让 LLM 生成结构化任务图，而不是直接生成每个机器人每一步的底层动作。任务图通常包含任务名称、语义目标、依赖关系、能力需求和后续模块可消费的结构化字段。

相关路径：

```text
modules/task_solver/sgi_planner/planning_layer.py
modules/task_solver/sgi_planner/plan_module.py
modules/task_solver/sgi_planner/actions/plan_task.py
modules/task_solver/sgi_planner/task_graph_manager.py
modules/task_solver/baseline_planners/
```

## Allocator / Solver 层

TANGO allocator 负责将 ready tasks 分配给机器人。该问题涉及能力约束、空间代价和资源限制，不适合完全交由 LLM 文本生成。

```text
ready tasks + robot states + capability constraints
  -> TANGO allocator
  -> allocation result
```

相关路径：

```text
modules/task_solver/sgi_planner/alloc_module.py
modules/task_solver/sgi_planner/allocator/
modules/task_solver/sgi_planner/allocator/tango/
```

名称边界：

- `--methods sgi`：选择 planner 方法。
- `GSI_TANGO_SOLVER_BACKEND=scip/gurobi`：选择 allocator 优化后端。

## World Model 层

World model 维护系统对环境的当前理解，并将执行结果转换为下一轮规划可用的状态。

职责：

- 从 scene graph 初始化机器人、目标和物体状态。
- 接收技能执行 outcomes。
- 更新局部世界状态。
- 判断目标是否完成。
- 为 replan 构造反馈和状态。

相关路径：

```text
modules/task_solver/world_model/world_model_layer.py
modules/task_solver/world_model/world_model_manager.py
modules/task_solver/world_model/goal_progress_monitor.py
modules/task_solver/world_model/status_tracker.py
```

## Platform 层

Platform executor 接收技能序列并执行。大规模 benchmark 默认使用 semantic platform：

- 不依赖 UE。
- 执行速度快。
- 能产生结构化 outcomes、newcase events 和 metrics。

相关路径：

```text
modules/platform/semantic_platform/platform_executor.py
modules/platform/semantic_platform/skill_executor.py
modules/platform/semantic_platform/new_case_controller.py
modules/platform/semantic_platform/new_case_generator.py
modules/platform/semantic_platform/new_case_injector.py
modules/platform/unreal_platform/
```

Unreal platform 主要用于 UE5 联动或展示，不是当前训练和 benchmark 的默认路径。

## Feedback / Replan 层

执行过程中可能出现技能失败、部分目标完成、pending tasks、new case 或外部反馈。Feedback processor 将这些信息转换为重规划策略：

- `NONE`：无需重规划。
- `PARTIAL`：保留部分任务图，只调整受影响部分。
- `FULL`：重新构造 prompt 并生成完整任务图。

相关路径：

```text
modules/task_solver/sgi_planner/feedback_processor.py
modules/task_solver/sgi_planner/base_feedback_processor.py
modules/utils/replan_recorder.py
```

RLVR 数据中的 replan 样本也来自该链路，包含 state、newcase、feedback 和 validator/reward 所需信息。

## Validator / Reward 层

RLVR 训练使用 validator/reward manager 检查模型输出是否满足格式、任务约束和可执行性要求。

相关路径：

```text
modules/plan_validator/plan_validator.py
modules/plan_validator/replan_state_store.py
llm_finetune/rlvr/gsi_reward_manager.py
llm_finetune/rlvr/build_replan_state_store.py
```

Benchmark 关注 planner 在任务中的运行结果；RLVR reward 关注模型输出在给定 state 下是否可验证、可执行、可得分。两者共享任务和状态语义，但服务于不同阶段。

## Metrics 与输出

批量运行由 `ParallelBatchRunner` 管理。每个 task 生成独立 run 目录，batch 结束后写出：

- `summary.jsonl`
- `aggregate_full.json`

聚合逻辑：

```text
run/utils/batch_runner.py
run/utils/analysis.py
```

常见指标包括 `success_rate`、`llm_calls`、`replans_total`、`total_energy`、`newcase_total`、`planning_duration`、`allocation_duration` 和 token 统计。详细说明见 [输出目录](../training/outputs.md)。

## 推荐阅读代码顺序

1. `run/run_exp_multi_method.py`
2. `run/utils/batch_runner.py`
3. `run/utils/case_runner.py`
4. `modules/task_solver/unified_task_solver.py`
5. `modules/task_solver/sgi_planner/planning_layer.py`
6. `modules/task_solver/sgi_planner/alloc_module.py`
7. `modules/platform/semantic_platform/skill_executor.py`
8. `modules/task_solver/sgi_planner/feedback_processor.py`
