# Hybrid Search Supervisor Pilot（2026-08-06）

## 1. 实验问题

本实验验证：在不修改冻结的 A/B/C/D 方法前提下，新增规则式监督器 E，按完整 viewpoint action 在四种策略间切换，是否能相对 D 同时改善成功率、资源效率和最差条件表现。

这是 5 paired seeds 的方向性 pilot，不是论文统计结果，也不更新 `search-skill-baseline-2026-08-05` 或既有 acceptance report。

## 2. 方法 E

- 默认与目标确认：D `AdaptiveBeliefLookaheadPolicy`。
- 全局覆盖不足：A `CoveragePolicy` fallback。
- belief 与高质量覆盖停滞：B `RandomPolicy` 单航段 escape。
- 连续低质量或被拒绝的负观测：C `OriginalActiveSearchPolicy` visibility fallback。
- 每种模式最少保持 2 个完整 viewpoint actions。
- E 继续使用 confidence-gated belief update；不混合四种策略不可比较的 utility score。
- trace 记录 mode、previous mode、switch reason、mode age 和触发信号。

## 3. 实验设置

- 24 条件：4 environment × 3 prior × 2 sensor quality。
- 每条件 5 paired seeds，seed block 为 0–4。
- D/E 各 120 episodes，共 240 episodes。
- 地图、目标、传感器、控制模型、时间预算和 viewpoint budget 完全相同。
- 结果：`results/search_skill_hybrid_pilot/experiment_report_5seeds.json`。

## 4. 结果

| 指标 | D Improved Active | E Hybrid Supervisor | E / D 或差值 |
|---|---:|---:|---:|
| Success rate | 33.3% | 30.0% | -3.3 pp |
| 成功 episode 发现距离 | 179.59 m | 174.03 m | 0.969× |
| 成功 episode 发现时间 | 22.41 s | 21.71 s | 0.969× |
| 平均总航程 | 343.51 m | 331.21 m | 0.964× |
| Success / km | 0.970 | 0.906 | 0.933× |
| Replan proxy | 6.175 | 7.108 | 1.151× |
| Belief Brier | 0.01990 | 0.02069 | 1.040× |
| Worst-case success | 0% | 0% | 无改善 |

Paired outcome：E 独有成功 1 次，D 独有成功 5 次，其余 114 对结果相同。

环境平均成功率变化：

- Open area：73.3% → 73.3%。
- Street edge：43.3% → 36.7%。
- Woodland：13.3% → 6.7%。
- Building passage：3.3% → 3.3%。

E 的 973 个 actions 中，D default 占 40.8%，A coverage fallback 占 26.2%，C visibility fallback 占 31.0%，B random escape 占 2.0%。排除初始模式赋值后，平均每个 episode 切换 2.64 次。

## 5. 失败分析

E 将平均航程降低 3.6%，且成功 episode 的发现距离和时间都降低约 3.1%，但这些节省以更低成功率为代价。`budget_exhausted_before_target_region` 从 D 的 5 次增加到 E 的 33 次，说明 coverage/visibility fallback 占用了有限预算，却没有稳定地把 UAV 带到目标区域。

当前 supervisor 通过 0/5 promotion gates：成功率、success/km、replan proxy、Brier 和 worst-case success 均未达到非劣或改善要求。

## 6. 判定

**DO NOT PROMOTE**。

本轮证明了四策略可以在工程上被可解释地融合并统一评测，但当前规则没有带来更好的 Search Skill。不得用 seed 0–4 继续反复调阈值后再报告为独立验证结果。

下一轮应保留本报告，使用新的 seed block 做归因实验：分别移除 coverage fallback、random escape、visibility fallback 和 mode hysteresis。只有某个组件在新 seeds 上同时改善 success、success/km 或 worst-case success，才进入 20-seed 验证。
