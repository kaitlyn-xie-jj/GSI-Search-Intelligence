# 评估指标

GSI benchmark 会记录任务成功率、耗时、LLM 调用、重规划、new case 和资源使用等指标。不同实验的字段可能略有差异，但核心含义保持一致。

## 核心指标

| 指标 | 含义 |
| --- | --- |
| `success_rate` | 成功任务数占总任务数的比例。 |
| `elapsed_sec` | 单个任务从开始到结束的总耗时。 |
| `planning_duration` | planner 生成或更新计划的耗时。 |
| `allocation_duration` | TANGO allocator 分配任务的耗时。 |
| `llm_calls` | 单次任务运行中的 LLM 调用次数。 |
| `prompt_tokens_total` | prompt token 总数；需启用 token 统计。 |
| `response_tokens_total` | response token 总数；需启用 token 统计。 |
| `total_energy` | 语义执行阶段估计的总能耗或移动代价。 |
| `replans_total` | 完整重规划和局部重规划次数之和。 |
| `newcase_total` | 执行过程中记录到的 new case 数量。 |
| `atomic_task_completion_rate` | 已完成 atomic tasks 占总 atomic tasks 的比例。 |

## 使用方式

多方法对比时，不应只看 `success_rate`。建议同时比较耗时、LLM 调用、重规划次数和能耗。New case 评估还应比较 `newcase_total`、`newcase_top_types` 和不同 `newcase_N` 目录下的变化。

## 输出位置

聚合指标通常位于：

```text
<output-root>/<timestamp>/batch_<method>/aggregate_full.json
<output-root>/<timestamp>/newcase_N/batch_<method>/aggregate_full.json
```

详细结构见 [输出目录](../training/outputs.md)。
