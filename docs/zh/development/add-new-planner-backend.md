# 新增 Planner 后端

新增 planner 后端适用于接入新的 LLM planner、规则 planner 或 baseline 方法。新方法应复用统一任务生命周期，以便与现有方法比较。

## 关键路径

```text
modules/task_solver/solver_factory.py
modules/task_solver/sgi_task_solver.py
modules/task_solver/baseline_planners/
modules/task_solver/sgi_planner/
```

## 接入要求

- 输入复用相同 task、world model 和 robot skill 信息。
- 输出可转换为平台可执行技能序列。
- 格式错误和约束问题应能反馈给 validator 或上层重试逻辑。
- Benchmark 入口可通过 `--methods` 选择新方法。

## 验证

```bash
python run/run_exp_multi_method.py \
  --methods <new_method> \
  --max-count 5 \
  --dataset-root "$GSI_DATASET_ROOT"
```

如果新 planner 依赖新的 LLM endpoint，需要同步更新：

```text
modules/task_solver/llm_framework/config/llm_config.yaml
```
