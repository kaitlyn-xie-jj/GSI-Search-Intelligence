# Search Skill 验收报告（2026-08-05）

## 最终结论

**NOT READY**

改进策略 D 在统一有限预算评测中的成功率高于覆盖基线 A，但没有达到验收门槛：D 的成功航程更长、单位航程成功数更低，最差场景成功率仍为零。已有 PX4/Gazebo 四位置实验也只有 1/4 成功。当前实现适合继续受控仿真和策略迭代，不适合室外部署。

## 1. 能力描述

Search Skill 在公开环境语义、目标位置 belief、UAV 状态、传感器模型和资源约束下闭环选择航点。系统支持途中正负观测、置信门控 Bayesian 更新、确认航点、时间预算终止和可解释策略 trace。搜索核心不读取目标真值；目标位姿只在 evaluator 中用于评分。

## 2. 输入/输出

完整契约见 [Search Skill 接口](../reference/search-skill-interface.md)。输入包括 `EnvironmentModel`、`BeliefState`、`UAVState`、`SensorModel`、`TargetSpec` 和 `Constraints`。输出包括下一航点、短视界轨迹、更新 belief、`continue/confirm/success/abort` 决策及奖励分解。

## 3. 算法

观测模型分离三个概率项：

```text
P(found | viewpoint)
  = P(target in visible cells)
  * P(target visible | viewpoint)
  * P(sensor detects | target visible)
```

无有效点云投影、RGB-D 不完整、视线遮挡或质量不足时，负更新强度为零，并记录拒绝原因。改进策略的 horizon-2 候选池按 30% exploitation、40% exploration/frontier、30% semantic representatives 取样，并采用空间最远点选择。重规划由正检测、belief 总变差、KL divergence、轨迹失效或期望奖励变化触发，并受最小时间间隔约束。

奖励为：

```text
detection contribution
+ information gain contribution
+ exploration contribution
- flight cost
- revisit penalty
- risk penalty
```

所有权重均可配置，每次选点 trace 包含各贡献和最终分数。

## 4. 统一 benchmark

结果文件：`results/search_skill_acceptance/evaluation_report.json`。本次报告绑定 baseline ID `search-skill-baseline-2026-08-05`；冻结前的报告保留为 `evaluation_report_pre_baseline_isolation.json`。

- 方法：A 覆盖割草机、B 随机、C 原始 active search、D 改进 active search。
- 条件：4 环境 × 3 priors × 2 传感器质量 = 24 个条件。
- 重复：每个条件 20 seeds。
- 总量：1,920 episodes，每个方法 480 episodes。
- 共享条件：120 m × 100 m 地图、20 m 网格、10 m/s 控制模型、55 s/10 航点预算和同一二元传感器模型。

| 方法 | 成功率（95% CI） | 成功检测航程 | 成功检测时间 | success/km | 平均 replans | Brier（越低越好） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A Coverage | 25.6% [21.7, 29.5] | 143.9 m | 22.3 s | 1.506 | 8.46 | 0.02089 |
| B Random | 8.8% [6.2, 11.3] | 354.4 m | 41.2 s | 0.179 | 7.14 | 0.02251 |
| C Original active | 30.4% [26.3, 34.5] | 209.9 m | 26.1 s | 0.814 | 7.17 | 0.01996 |
| D Improved active | 33.3% [29.1, 37.6] | 181.1 m | 22.6 s | 0.963 | 6.17 | 0.01966 |

离线 `replans` 是每次新策略决策的计数代理；ROS 在线 trace 另外记录迟滞真正触发的 replan reason 和间隔。

## 5. 与覆盖基线比较

D 相对 A 的成功率提高 7.7 个百分点，但置信区间明显重叠。D 的成功检测航程是 A 的 1.259 倍，单位航程成功数仅为 A 的 0.639 倍。D 因此既没有证明“稳定地优于覆盖”，也没有以显著更低成本达到相近成功率，不满足用户给定的 READY 条件。

D 相对 C 的成功率提高 2.9 个百分点，成功检测航程下降 13.7%，全体 episode 总航程下降 7.4%，平均 replan 代理下降 14.0%，Brier score 下降约 1.5%。C 使用冻结的 `OriginalActiveSearchPolicy` 且关闭 confidence gating，D 才启用门控和其他改进，因此这些差异可以归因于完整改进栈，但现有 A/B/C/D 设计不能继续拆分每个模块的独立贡献。

## 6. 鲁棒性

D 的分组成功率：

| 条件 | 平均成功率 |
| --- | ---: |
| Open area | 77.5% |
| Street edge | 35.8% |
| Woodland | 12.5% |
| Building passage | 7.5% |
| Correct prior | 43.8% |
| Wrong prior | 20.0% |
| Uniform prior | 36.3% |
| Normal sensor | 44.6% |
| Reduced quality | 22.1% |

四个 D 条件为 0%：street-edge/wrong/reduced、woodland/wrong/reduced、building-passage/wrong/reduced 和 building-passage/uniform/reduced。所有方法的 worst-case scenario success 均为 0%。

## 7. 失败案例

D 的 480 episodes 中，160 成功；主要失败分类为 occluded 154、sensor miss 112、target region unsearched 32、budget exhausted before target region 22。多样候选池显著减少了“预算耗尽前未到目标区域”，但在低可见性环境中仍把大量预算花在无法形成两次确认的视点上。

已有 Yungu PX4/Gazebo 四位置结果已用修复后的 trial 汇总重新生成：road connector south 成功，定位误差 1.28 m；street edge、woodland edge 和 building passage 未找到。总体 1/4，Wilson 95% CI 为 4.6%–69.9%。这组实验早于本报告的最终观测门控、候选多样性和迟滞实现，因此只能证明执行链路可运行，不能验证最终策略提升。

## 8. 剩余限制

- 最差遮挡场景仍为 0%，不具备部署所需的下界保证。
- 可见性概率目前主要来自公开语义先验和点云质量门控，尚未完成真实室外标定。
- 离线 evaluator 使用简化运动和遮挡模型，未模拟风、GNSS 漂移、动态障碍和飞控失效。
- 在线四场样本量过小，且尚未用最终代码重新执行 A/B/C/D 配对实验。
- 两次独立确认在低可见性区域代价高，需要主动确认轨迹或多传感器融合。
- READY 前必须完成最终代码的 PX4/Gazebo 配对复验，并在 woodland/building passage 的 reduced-quality 条件中消除 0% 成功场景。

## 部署问题

**Is this Search Skill ready for deployment? NOT READY.**

证据是：统一 benchmark 未达到成功/成本验收标准，最差条件成功率为零，真实在线四位置结果仅 25%。
