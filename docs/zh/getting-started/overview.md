# 项目概览

GSI 用于研究多机器人场景中的任务规划、分配、执行验证和重规划。系统将自然语言任务、场景状态、机器人能力和执行反馈连接为可评估的 benchmark 与训练流程。

## 核心能力

- 将任务指令和环境状态转换为结构化计划。
- 使用 validator 检查计划格式、技能约束和状态约束。
- 使用 TANGO allocator 将 ready tasks 分配给机器人。
- 在 semantic platform 中执行技能并记录结果。
- 在执行失败或 new case 出现后触发重规划。
- 采集 SFT/RLVR 数据并评估训练后的模型。

## 典型流程

```text
数据集任务
  -> SGI planner
  -> validator / allocator
  -> semantic platform
  -> feedback / replan
  -> metrics / outputs
```

## 下一步

- 运行最小示例：阅读 [快速跑通](quickstart.md)。
- 复现实验：阅读 [复现实验结果](reproduce-results.md)。
- 理解代码结构：阅读 [仓库结构](../development/repo-layout.md)。
