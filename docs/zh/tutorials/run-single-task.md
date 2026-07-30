# 运行单个任务

单任务运行适合调试 planner、配置和执行反馈。批量评估应使用 [运行 Benchmark](run-benchmark.md)。

## 入口

```bash
python run/run_exp.py
```

当前单任务脚本中的 task id、配置和模型参数通常需要在脚本或配置文件中调整。运行前确认：

- `config/default.json` 中的 `solver_type`、`platform_type` 和 `enable_replanning`。
- LLM endpoint 已在 `modules/task_solver/llm_framework/config/llm_config.yaml` 或环境变量中配置。
- 数据集路径存在，或脚本能从默认 `dataset/` 读取。

## 调试建议

- 使用 semantic platform 做快速验证。
- 打开 `enable_detailed_print` 查看更详细的执行过程。
- 复现批量结果时，应使用 `run/run_exp_multi_method.py`，不依赖单任务脚本中的临时 task id。

## 相关文档

- 批量运行：见 [运行 Benchmark](run-benchmark.md)。
- 配置字段：见 [配置说明](../reference/config.md)。
