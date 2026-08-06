# Search Skill Baseline v1

本 baseline 自 2026-08-05 起冻结。机器可读定义见 `config/search_skill_baseline_v1.json`。

## 固定方法

| ID | 方法 | 冻结定义 |
| --- | --- | --- |
| A | Coverage | `CoveragePolicy`，完整 lawn-mower coverage route |
| B | Random | `RandomPolicy`，相同候选集合上的 seeded random ordering |
| C | Original active | `OriginalActiveSearchPolicy`，保留原始 active utility 和无置信门控的强负 belief update |
| D | Improved active | `AdaptiveBeliefLookaheadPolicy`，启用置信门控、diversity candidate pool、semantic/frontier、预算过滤与奖励分解 |

A、B、C 不再接受调参或算法修改。后续改动只能形成新的 D variant 或显式 ablation。若共享模拟器、地图或传感器模型发生变化，必须生成新的 baseline version，不能覆盖 v1 结果。

## 固定评测条件

- 地图：120 m × 100 m，20 m grid。
- 控制模型：10 m/s，航点观测时间 1 s。
- 预算：55 s，最多 10 个航点。
- 条件矩阵：4 environments × 3 priors × 2 sensor conditions。
- 重复：每个条件 20 seeds。
- 四种方法接收相同 raw visibility 和 detector event；C 使用 legacy observation interpretation，D 使用 confidence-gated interpretation。

## 保留结果

- 冻结前统一评测：`results/search_skill_acceptance/evaluation_report_pre_baseline_isolation.json`。
- 当前严格隔离评测：`results/search_skill_acceptance/evaluation_report.json`。
- Yungu 四位置实验：`results/yungu2030_robustness/20260805_optimized_policy_failed_four/batch_summary.json`。

源分支为 `Yungu-map-demo`，基础 Git revision 为 `692c322dd57f21ce73654a4fcb8502cfa574e5e5`。冻结实现位于该 revision 之上的当前研究工作树；正式投稿前应将 baseline 单独提交并打 tag。
