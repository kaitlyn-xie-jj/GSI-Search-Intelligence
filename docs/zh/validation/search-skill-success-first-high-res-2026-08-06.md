# Search Skill Success-First + High-Resolution Camera（2026-08-06）

## 1. 目标与隔离

本轮使用词典序目标：先最大化 success rate，再比较 worst-case success、success/km 和 distance per success。冻结的 A/B/C/D 与既有 acceptance report 均未修改。

- 开发集：base seed 5，5 repetitions/condition。
- Held-out：base seed 10，20 repetitions/condition。
- 24 条件：4 environments × 3 priors × 2 sensor conditions。
- 调参：6 variants × 120 episodes = 720 episodes。
- Held-out：3 methods × 480 episodes = 1,440 episodes。

## 2. 高分辨率相机假设

离线 benchmark 将相机升级建模为一个 composite sensor profile：

- 假定分辨率：3840 × 2160。
- Detection probability：0.96。
- False-positive probability：0.005。
- Effective recognition radius：30 m（当前 profile 为 25 m）。
- Detection confidence：0.98。

这是合成传感器假设，不是像素级或实机验证。当前 energy proxy 不包含高分辨率相机、GPU/CPU 推理和散热功耗，因此不能据此宣称真实电量下降。

## 3. 调参结果

| Variant | Dev success | Worst case | Success/km | Distance/success |
|---|---:|---:|---:|---:|
| highres_d_frozen_weights | 51.7% | 0% | 1.827 | 547.3 m |
| detection_priority | 50.0% | 0% | 1.739 | 575.1 m |
| balanced_success | 50.0% | 0% | 1.737 | 575.6 m |
| late_recovery_3 | 50.0% | 0% | 1.733 | 577.2 m |
| late_recovery_2 | 50.0% | 0% | 1.730 | 578.0 m |
| late_recovery_2_high_coverage | 50.0% | 0% | 1.723 | 580.5 m |

词典序选择结果为 `highres_d_frozen_weights`：保持冻结 D 权重并关闭 recovery。也就是说，本轮没有发现优于 D 的策略参数；所有 E2 recovery 变体均被开发集拒绝。

## 4. Held-Out 结果

| 指标 | D current camera | D high-res | 变化 |
|---|---:|---:|---:|
| Success rate | 30.0% | 47.3% | +17.3 pp |
| Success 95% CI | [25.9%, 34.1%] | [42.8%, 51.8%] | - |
| Worst-case condition | 0% | 5% | +5 pp |
| Mean total distance | 359.06 m | 292.51 m | -18.5% |
| Detection distance（success） | 193.60 m | 172.76 m | -10.8% |
| Detection time（success） | 24.12 s | 21.68 s | -10.1% |
| Success/km | 0.836 | 1.617 | 1.94× |
| Distance/success | 1196.85 m | 618.52 m | -48.3% |
| Energy proxy/success | 72.08 | 38.22 | -47.0% |
| Brier | 0.02065 | 0.01411 | -31.7% |
| Replan proxy | 6.344 | 5.898 | -7.0% |

Paired outcome：high-res 独有成功 97 次，current camera 独有成功 14 次，369 对一致。

环境平均 success：

- Open area：64.2% → 84.2%。
- Street edge：38.3% → 64.2%。
- Woodland：12.5% → 29.2%。
- Building passage：5.0% → 11.7%。

## 5. 判定

- **采用 high-resolution camera profile 作为下一轮实机候选配置。**
- **DO NOT PROMOTE E2 policy**：held-out 中 E2 与 D-high-res 完全一致，因为开发集选中的就是无 recovery、冻结权重的 D。
- **Search Skill 仍为 NOT READY**：整体 success 仅 47.3%，building passage 仅 11.7%，最差单条件仅 5%。

下一步必须把该 composite profile 拆成检测概率、识别距离和误报率三项硬件消融，并在真实相机数据上测量 detection/false-positive curve、运动模糊、遮挡和额外功耗。通过真实传感器标定后，才能进入电量、航时和路径距离优化。
