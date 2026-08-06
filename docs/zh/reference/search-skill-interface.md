# Search Skill 接口

Search Skill 是一个目标条件化的闭环无人机能力。它每次选择一个短视界动作，只消费传感器观测产生的证据，并在满足成功条件或资源约束时终止。

## 输入

| 概念 | 仓库类型 | 必需内容 |
| --- | --- | --- |
| `EnvironmentModel` | `SearchGrid` | 可搜索单元、公开语义标签、几何边界和排除区域；不得包含目标真值。 |
| `BeliefState` | `BeliefMap` / `SearchState.belief` | 定义在全部可搜索单元上的归一化目标位置概率。 |
| `UAVState` | `Viewpoint` 与执行遥测 | map 坐标系位姿、已用时间、航程、能量和当前轨迹有效性。 |
| `SensorModel` | `BinarySensorModel` 与观测质量字段 | 条件检测率、误报率、预测可见性、传感器质量和负更新置信度。 |
| `TargetSpec` | `SearchTask.target` 与 `SearchSuccessCriteria` | 开放词表目标、属性、置信度阈值、独立确认数、持续时间和定位容差。 |
| `Constraints` | `SearchBudget`、排除区域和候选风险 | 时间、距离、能量、航点数、安全区域、规划速度和结束预留时间。 |

规划器分别维护目标概率、可见性概率和条件检测概率：

```text
P(found | viewpoint)
  = P(target in visible cells)
  * P(target visible | viewpoint)
  * P(sensor detects | target visible)
```

RGB-D 不完整、点云投影无效、视线遮挡或观测质量不足时，adapter 必须把
`negative_update_strength` 设为零，并填写
`negative_update_rejection_reason`。

## 输出

| 输出 | 契约 |
| --- | --- |
| 下一航点 | `SearchSession.next_viewpoint()` 返回 map 坐标系 `Viewpoint` |
| 短视界轨迹 | `SearchSession.remaining_plan()`；无碰插值和底层执行由控制器负责 |
| 更新后的 belief | 调用 `record_observation` 或 `record_transit_observation` 后的 `SearchState.belief` |
| 决策 | `continue`、`confirm`、`success` 或 `abort` |
| 解释 | `policy_decisions[-1]`，包含概率项、奖励贡献、候选池来源和最终分数 |

决策语义：

- `continue`：继续当前轨迹，迟滞判据未允许重规划。
- `confirm`：已有正检测，但尚未满足独立确认条件。
- `success`：返回 `SearchOutcomeStatus.FOUND`，且确认与定位条件全部满足。
- `abort`：预算耗尽、候选为空、控制器报告不可恢复的无效轨迹，或收到外部终止。

## 假设与职责

- 所有位置使用 map 坐标系和米制单位；坐标变换由传感器 adapter 负责。
- 避障和底层飞行稳定性由 UAV controller 保证，搜索策略不替代飞控。
- 可见性预测可以使用公开语义，但不能读取目标位姿或仅供 evaluator 使用的数据。
- 圆形 footprint 只能用于规划近似，不能单独构成 Bayesian 强负证据。
- 允许途中接受首次正证据；相邻视频帧不能算作两次独立确认。
- 正检测或轨迹无效必须重规划；负证据只有在满足最小间隔且 belief、KL 或期望奖励显著变化时才重规划。
- 即使未找到目标，终止结果也必须包含最终 belief 和资源消耗指标。
