# GSI 中文文档

GSI 是面向具身多智能体任务规划、执行验证和重规划的系统与训练仓库。本文档以中文为主要维护版本，覆盖运行评估、结果复现、模型训练、系统扩展和故障排查。

## 系统主线

```text
任务数据
  -> LLM planner 生成任务计划
  -> validator 检查计划约束
  -> TANGO allocator 分配机器人
  -> semantic platform 执行技能
  -> world model 更新状态
  -> feedback processor 判断是否重规划
  -> batch runner 汇总指标
```

## 阅读入口

- 初次运行：阅读 [快速跑通](getting-started/quickstart.md)。
- 复现实验：阅读 [复现实验结果](getting-started/reproduce-results.md)。
- 理解系统：阅读 [系统架构](concepts/architecture.md)。
- 训练模型：阅读 [训练与模型概览](training/overview.md)。
- 扩展代码：阅读 [仓库结构](development/repo-layout.md)。
- 排查问题：阅读 [故障排查](troubleshooting/gpu-and-vllm.md)。

## 关键边界

- `solver_type` 或 `--methods` 选择 planner 方法，例如 `sgi`、`spine`、`smartllm`、`lipllm`。
- `GSI_TANGO_SOLVER_BACKEND` 选择 TANGO allocator 的优化后端，例如 `scip` 或 `gurobi`。
- vLLM 负责提供 OpenAI-compatible 模型服务；`GSI_LLM_MODEL` 必须等于服务暴露的模型名。
- `WindyLab/GSI` 是 benchmark 数据集；SFT/RLVR 训练数据使用独立数据集。
- 普通 benchmark 默认不注入 new case；newcase 评估需要显式设置 `--enable-newcase` 和 `--newcase-counts`。

## 建议路径

先完成一个小规模 benchmark，再阅读输出目录和系统架构。训练相关内容依赖数据、模型、validator、vLLM 和 GPU 环境，建议在评估链路稳定后再进入 SFT 或 RLVR。
