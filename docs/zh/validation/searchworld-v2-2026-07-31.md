# SearchWorld V2 验证记录（2026-07-31）

## 验证目标

SearchWorld V2 用于验证同一个 task-conditioned active search policy 能否在
不同室外物理布局中运行，而不修改机器人控制、ROS topic、检测器或 policy
权重。V2 首批固定三类地图：校园、工业物流区和郊区街区。

本记录区分两类数据：

- 开发诊断数据：保留失败 trial，用于定位仿真和算法契约问题；
- 最终配置基线：三张地图使用完全相同的最终机体、prior 投影和执行配置。

一次三地图 smoke batch 不能证明统计泛化。它只证明当前端到端接口在三个
不同物理场景上可运行，并给后续重复实验提供基线。

## 最终配置

| 项目 | 配置 |
| --- | --- |
| Gazebo | Harmonic 8.14.0 |
| ROS 2 | Humble |
| PX4 airframe | `4011_gz_x500_gsi_rgbd_nadir` |
| 相机 | 单个 640 x 480、10 Hz、正下视 RGB-D |
| 地图范围 | 100 m x 80 m |
| 搜索高度 | 15 m |
| 搜索网格 | 10 m |
| prior 投影 | `label_mass` |
| 仿真模式 | `HEADLESS=1` |
| 解锁前 setpoint | 100 个（5 s） |
| 单 trial 上限 | 300 s |

`label_mass` 先把每个语义类别权重除以该类别覆盖的 cell 数，再投影到
cell belief。它避免大面积语义区仅因网格更多而获得额外总先验质量。旧地图
默认继续使用 `cell_affinity`，因此该变更向后兼容。

## 场景复杂度

| 地图 | Seed | 建筑 | 树 | 停放车辆 | 集装箱 | 障碍墙 | 灯杆 | 可搜索 cell |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| campus | 101 | 4 | 20 | 12 | 3 | 3 | 8 | 72 |
| industrial | 202 | 5 | 6 | 10 | 14 | 4 | 10 | 66 |
| suburban | 303 | 6 | 18 | 10 | 2 | 3 | 8 | 70 |

三张地图包含不同道路拓扑、语义区、建筑遮挡、植被和重复物体。公共
`semantic_map.json` 不包含目标实体或真实位置；目标只存在于 evaluator-only
的 `ground_truth.json`。

## 最终三地图结果

最终干净 batch：

```text
results/gazebo_search_world_v2/20260731-172140/
```

| 地图 | 结果 | Viewpoints | 时间 (s) | 距离 (m) | 覆盖率 | 定位误差 (m) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| campus | found | 6 | 131.500 | 126.202 | 0.292 | 0.184 |
| industrial | found | 12 | 288.228 | 304.944 | 0.561 | 0.186 |
| suburban | found | 2 | 55.400 | 33.185 | 0.114 | 0.054 |

汇总：

```text
trial_count: 3
success_count: 3
success_rate: 1.0
mean_elapsed_time_s_success: 158.376
mean_distance_m_success: 154.777
mean_localization_error_m_success: 0.141
```

industrial 明显需要更多 viewpoint 和飞行距离，说明三张图已经产生不同难度，
而不是只有视觉外观变化。

每个 trial 保存配置快照、manifest、ground truth、完整 viewpoint trace、ROS
runtime log、Docker/Gazebo 诊断尾部和 per-trial summary。批次根目录保存
`trials.csv` 与 `batch_summary.json`。结果目录被 Git 忽略，但保存在本机。

## 开发中发现并修复的问题

开发诊断 batch：

```text
results/gazebo_search_world_v2/20260731-151646/
```

### 1. 高实体负载下过早解锁

初始 suburban 有 118 个 Gazebo model。两次 trial 都出现 armed/OFFBOARD 但
电机交替饱和、无法起飞。将无研究价值的重复实体降到 98 个，并把解锁前
setpoint 从 40 增到 100 后，staged takeoff 稳定完成。

### 2. GUI 扭曲实时因子

GUI trial 出现约 200 s 的 viewpoint 停顿。batch runner 增加 VisionFlow
`HEADLESS` 透传，默认只运行 `gz sim -r -s` server；可视单图启动仍默认 GUI。

### 3. 语义区域面积偏差

原 `cell_affinity` 将同一语义权重赋给每个 cell，大停车区因 cell 更多获得
过量总 belief。`label_mass` 修正后，类别权重不再随区域面积线性放大。

### 4. Policy 与真实相机视场不一致

候选生成器明确采用 nadir 圆形地面 footprint，但 V1.1 相机是 45 度斜视。
斜视 trial 在目标正上方仍无法检测。V2 新增独立 airframe 4011 和正下视
RGB-D；修正后的 suburban 在 2 个 viewpoint 内完成两次确认。V1.1 模型保持
不变。

## 自动化验证

在带 VisionFlow ROS overlay 的容器中：

```text
gsi_search_bridge tests: 33/33 passed
search_intelligence tests: 91/91 passed
total: 124/124 passed
```

另外完成：三场景确定性生成、SDF XML 解析、semantic prior 归一化、公共地图
不泄露 ground truth、Bash syntax 和 `git diff --check`。

## 尚未完成

- 最终三地图结果每图只有一个 seed/一次 trial，不能给出置信区间；
- 当前 prior 仍是符合 LLM 输出契约的确定性 fixture，不是在线 LLM；
- 当前黄色检测器是仿真接口 baseline，不是 open-world perception；
- 需要增加 held-out seeds、target slots、天气/光照和传感器噪声；
- 需要对 Coverage、GreedyPrior、ActiveSearch 在相同 Gazebo trial matrix 上比较；
- 25 分钟 headless 资源 gate 仍需单独执行，不应与搜索成功率混为一个指标。
