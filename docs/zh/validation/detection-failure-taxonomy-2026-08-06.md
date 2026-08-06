# Detection Failure Taxonomy（2026-08-06）

## 1. 范围

本分析冻结 `D_high_res_camera` 的 detector、相机 profile、搜索策略、预算和 held-out seed block，只重跑 480 episodes 以恢复逐 episode sensor trace。没有修改模型。

输出：`results/search_skill_success_first/failure_analysis.json`。

## 2. 串联概率

采用可审计的嵌套事件：

```text
P(success)
= P(searched)
× P(visible | searched)
× P(detected | visible)
× P(confirmed | detected)
```

| Stage | Count | Conditional probability |
|---|---:|---:|
| Total | 480 | - |
| Searched | 478 | 99.6% |
| Visible | 344 | 72.0% of searched |
| Detected | 285 | 82.8% of visible |
| Confirmed | 227 | 79.6% of detected |

乘积为 `47.29%`，与 held-out success rate 完全一致。

## 3. 失败分类

253 次失败被互斥拆分为：

| Category | Count | Failure share | Definition |
|---|---:|---:|---|
| not searched | 2 | 0.8% | 没有 observation footprint 覆盖目标 cell |
| searched but occluded | 134 | 53.0% | 覆盖目标 cell，但没有一次有效可见 |
| visible but missed | 59 | 23.3% | 至少一次可见，但没有真实目标检测 |
| false negative confirmation | 58 | 22.9% | 有真实检测，但没有满足两次独立确认 |
| bad localization | 0 | 0% | 检测存在，但定位不满足阈值 |

58 个 confirmation failure 全部只有一次真实检测。59 个 visible miss 中，41 个只有一次有效可见机会，15 个有两次，3 个有三次以上。

`bad localization = 0` 不能解释为真实定位已经解决：当前 synthetic profile 没有 localization noise，任务也没有启用严格 localization threshold，因此该分类在本轮基本不可触发。

## 4. 环境分解

| Environment | Success | Occluded | Visible miss | Confirm failure | Not searched |
|---|---:|---:|---:|---:|---:|
| Open area | 84.2% | 0 | 12 | 7 | 0 |
| Street edge | 64.2% | 16 | 17 | 9 | 1 |
| Woodland | 29.2% | 48 | 15 | 22 | 0 |
| Building passage | 11.7% | 70 | 15 | 20 | 1 |

Reduced-quality sensor 条件下 success 为 35.4%，包含 56 次 visible miss；normal 条件 success 为 59.2%，只有 3 次 visible miss。因此 detector quality 的主要空间集中在 degraded sensor，而遮挡在两类 sensor condition 中都很高。

## 5. 从 47.3% 到 70%

达到 70% 需要 `336/480` 次成功，即新增 109 次成功，挽回全部失败的 43.1%。

- 修复全部 not searched：47.7%，几乎没有空间。
- 修复全部 confirmation failure：59.4%，不足 70%。
- 修复全部 visible miss：59.6%，不足 70%。
- 同时修复全部 visible miss 和 confirmation failure：71.7%，理论上刚刚足够，但需要挽回其中 93.2%，不现实。
- 修复全部 occluded failure：75.2%；若只依靠遮挡恢复，需要挽回 134 次中的 109 次，即 81.3%。

所以 47.3% → 70% 不能依赖单一 detector 调参。最大空间是遮挡恢复，其次是增加有效可见机会并完成第二次确认。合理路径应组合：主动改变视角/高度减少 occlusion，多帧或再次接近减少 visible miss，并为已有一次正检测预留独立确认预算。
