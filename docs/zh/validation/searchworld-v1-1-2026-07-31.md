# SearchWorld V1.1 验证记录（2026-07-31）

## 结论

SearchWorld V1.1 的场景、标准 X500 RGB-D 机体、ROS 2 搜索桥、官方 PX4
控制器选择、真实 `SearchObservation` 和 25 分钟自动门槛已经完成。

飞行与搜索链路已通过：无人机保持 `armed + OFFBOARD`，完成任务并持续停留在
SearchWorld 边界内。IMU 重复时间戳问题也已修复。当前未通过项是 Gazebo GUI
进程的持续内存增长，因此 V1.1 整体资源门槛仍是 `FAIL`，不能标记为完全稳定。

## 实现范围

- `SYS_AUTOSTART=4010` 使用 PX4 官方 `mc_rate_control`、
  `mc_att_control` 和 `mc_pos_control`。
- 其他 VisionFlow airframe 继续使用 Pregme 控制器。
- `GZBridge::imuCallback` 丢弃同一 lockstep tick 内的重复 IMU 回调。
- 稳定性 CSV 记录 MAVROS XYZ、armed、mode、边界状态和真实 observation 数量。
- 25 分钟门槛要求至少 2 条 observation，且所有已采样位姿位于
  `[-40,40] x [-30,30]`。

## 飞行门槛

控制器切换后的短测完成 5 个 viewpoint，没有复现沿 Y 轴持续漂移。随后两次
长测分别完成 10 条和 6 条真实 observation，并成功发现目标。

最终长测共有 26 次 MAVROS 位姿探测，全部位于地图边界内。最终位姿为：

```text
x = -35.0047 m
y = -24.9862 m
z =  12.0079 m
armed = true
mode = OFFBOARD
```

## 25 分钟结果

### 运行 20260731-134245

官方控制器解决了飞行漂移，资源指标也在阈值内，但 PX4 记录了 34 条 IMU
timestamp critical error，因此自动判定为 `FAIL`。

```text
duration                 1513.452 s
samples                  151
maximum memory           3400.704 MiB
memory growth            1518.592 MiB
memory slope             66.570 MiB/min
IMU timestamp errors     34
out-of-bounds probes     0
observations             10
```

日志显示每一组错误前都有 `NodeShared::Publish() Error: Interrupted system call`。
VisionFlow 的 Gazebo bridge 当时会把同一 `hrt_absolute_time()` 再次发布给
`vehicle_imu`。V1.1 安装器现在对该重复回调进行过滤。

### 运行 20260731-141910

IMU guard 生效，所有 critical error 为 0，飞行与传感器持续正常。但是 Gazebo
GUI 进程持续增长，超过 growth 和 slope 门槛，因此整体仍为 `FAIL`。

```text
duration                 1511.494 s
samples                  151
maximum memory           6766.592 MiB
memory growth            4485.120 MiB   (limit 2560 MiB)
memory slope             193.830 MiB/min (limit 100 MiB/min)
IMU timestamp errors     0
attitude/PX4/OOM errors  0
out-of-bounds probes     0 / 26
observations             6
```

进程采样确认主要增长来自 `gz sim -g`，而不是 PX4 或搜索 ROS 节点。

## 数据位置

本地原始数据保存在：

```text
results/gazebo_stability/search_world_v1_1/20260731-134245/
results/gazebo_stability/search_world_v1_1/20260731-141910/
```

每个完整目录包含：

```text
summary.json
resource_timeseries.csv
runtime.log
docker.log
search_trace.jsonl
```

## 下一步

将验证拆成两个 profile：可观看的 GUI 飞行/搜索门槛，以及无 GUI 的 25 分钟
headless 资源门槛。先证明 server、PX4、ROS 和 RGB-D 的资源稳定性，再单独定位
Gazebo Harmonic GUI/D3D12 渲染进程的内存增长。
