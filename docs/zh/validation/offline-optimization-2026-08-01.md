# ActiveSearch 离线优化记录（2026-08-01）

## 目标

本轮只校准 `ActiveSearchPolicy` 的四个显式 utility 权重：检测概率、
信息增益、未观测概率质量和移动代价。传感器的检测概率和误报概率保持为
校准输入，不作为 policy 自由参数。本轮不使用 PPO 或神经网络训练。

## 数据隔离

默认场景共 24 个，每个布局包含近/远目标，以及 correct、diffuse、uniform、
misleading 四种 prior：

| Split | 未见布局 | 场景数 | 每场景重复数 | Episodes/候选 |
| --- | --- | ---: | ---: | ---: |
| Train | `compact_rectangle` | 8 | 5 | 40 |
| Validation | `large_rectangle` | 8 | 5 | 40 |
| Test | `l_shape` | 8 | 5 | 40 |

权重在 simplex 上归一化。距离代价由地图对角线归一化，避免固定 100 m 尺度
把地图大小编码进权重。64 个候选全部在 train 上运行，train 前 12 个（包含
预注册默认权重）进入 validation。test 只在 validation 完成选择之后运行。

选择 guardrail：候选相对默认权重的配对 validation objective 差异必须满足：

1. 95% 置信区间下界严格大于 0；
2. 平均实际改进不少于 `0.01`。

## 最终保留结果

结果目录：

```text
results/search_offline_optimization/20260801-search-v5/
```

没有候选同时满足两项 guardrail，因此保留预注册默认配置。归一化后的等价权重：

```text
detection:       0.4255319149
information_gain: 0.4255319149
novelty:         0.1063829787
travel:          0.0425531915
```

它与原始 `(1.0, 1.0, 0.25, 0.1)` 只相差统一比例，不改变候选排序。

| Split | Success | False positive | Mean SPL | Objective |
| --- | ---: | ---: | ---: | ---: |
| Train | 0.975 | 0.025 | 0.473 | 1.079 |
| Validation | 0.900 | 0.100 | 0.518 | 0.979 |
| Test | 0.950 | 0.050 | 0.476 | 1.037 |

## 开发诊断结果

`20260801-pilot-v1` 未强制默认权重进入 validation，选出了验证退化配置；
`pilot-v2` 加入默认 no-regression finalist 后保留默认；`search-v1` 使用均值选出
检测项单独占权重的配置，但 test success 从 0.95 降为 0.925；`search-v2` 加入
配对置信区间后仍接受仅 `0.00178` 的微小验证收益，而 test success 降为 0.90。
这些目录保留为方法修正证据，不作为最终 policy。

## 解释边界

本轮结论不是“默认权重已经最优”，而是：在当前搜索空间、样本量和单一验证
布局下，没有足够证据安全替换默认权重。由于开发过程中已经查看过 `l_shape`
结果，它从现在起视为已消耗测试集。下一轮修改选择方法后，必须增加从未查看过
的新地图族作为最终 held-out test，不能继续用 `l_shape` 声称无偏泛化。

优化 artifact 已包含可直接映射到 ROS 2 节点的 `ros_parameters`。Gazebo/PX4
验证必须冻结选出的权重，然后在配对 seed、target slot 和传感器条件下比较默认
与候选 policy。逐 episode 标量数据保存在
`offline_optimization_selected_episodes.csv`，可独立重算分组统计和置信区间。
