# SearchWorld V1 Gazebo/PX4 实时传感器验证

验证日期：2026-07-30

## 验收范围

本次验证检查以下闭环是否真实运行：

1. `ActiveSearchPolicy` 生成下一个 viewpoint；
2. ROS2 bridge 将 viewpoint 发送给 PX4；
3. PX4 进入 OFFBOARD、arming、起飞并到达 viewpoint；
4. Gazebo 的 RGB、深度、点云与 PX4 状态组成 `SearchObservation`；
5. observation 更新 Bayesian belief，并触发下一次 policy decision。

## Preflight 问题与处理

旧实例连续运行约 7 小时后，Gazebo server 的常驻内存增长到约 22 GB，
容器总内存约 24 GB，并发生 OOM。PX4 日志先后报告：

```text
Preflight Fail: no heading reference
Preflight Fail: Attitude failure (pitch)
vehicle_imu timestamp error
Arming denied: Resolve system health failures first
```

旧实例的机体已经倾倒，继续重试 arming 不能恢复传感器和姿态状态。保存日志后，
使用全新的 SearchWorld/PX4 容器重新启动。新实例内存约 1.4 GB，EKF 报告
attitude、local position 和 global position 均有效。PX4 随后成功完成：

```text
mode=OFFBOARD
armed=True
Staged takeoff complete; releasing horizontal search goals
```

因此本次修复没有关闭 PX4 preflight 检查，也没有使用 force-arm。根因属于长时间
运行后的 Gazebo 资源异常和失效姿态状态，处理方式是保存证据并重建干净仿真实例。

## 第一条真实 SearchObservation

第一条 observation 来自 `robotics_sensor_frame`，不是离线 benchmark 合成数据：

| 字段 | 值 |
| --- | ---: |
| step | 1 |
| commanded viewpoint | `(-15, -15, 12)` m |
| measured viewpoint | `(-15.1181, -14.8300, 12.0443)` m |
| position error | `0.2116 m` |
| observation quality | `0.8186` |
| visible cells | `6` |
| projected ground points | `869` |
| maximum sensor skew | `0.0800 s` |
| detections | `0` |

该 observation 同时确认以下实时输入可用：

- MAVROS odometry；
- MAVROS IMU；
- Gazebo RGB image 和 camera info；
- Gazebo depth image；
- Gazebo point cloud；
- 实时 detection topic；
- policy decision 和更新后的完整 belief map。

冻结实验前共记录 12 个 command 和 11 个 observation。实验在取得验收证据后主动
停止 ROS2 search launch；因此该记录没有伪造 `SearchOutcome`。

Gazebo/PX4 空闲约 51 分钟后，容器内存再次增长到 13.5 GB。停止仿真时 PX4 又报告
重复 IMU timestamp 和 pitch failure，说明当前 q940/多相机仿真存在独立的长时资源
与稳定性问题。原始证据保存后已安全停止容器，避免再次触发系统 OOM。该问题不影响
此前在健康时间窗内完成的 OFFBOARD、arming、飞行和 11 条 sensor-frame observation，
但后续长时间 benchmark 必须先解决 Gazebo server 的内存增长。

## 原始证据

原始文件保存在本地忽略目录：

```text
results/gazebo_sensor_validation/live_run_2026-07-30/
```

SHA-256：

```text
BCD66FE2B49ED66B488E15EE09216DC57CDFA14B9C95EA297D1263671FBD2AD0  search_world_v1_trace.jsonl
554BD393F714C8961802D3B910EDEEAA57699510D2C8285B93A6DCA6A9D9FEAF  search_launch.log
88C145FF462CAC418EE716EEFB61B8841CA790E7C4B2B0CB731651AC396C3B59  px4_container.log
```

`results/` 由 `.gitignore` 排除，仓库只提交本报告和可复现实验代码，不提交体积较大
且依赖具体运行机器的原始日志。

## 复现入口

在 `VisionFlow-22.04` WSL 终端中运行，并保持该终端打开：

```bash
bash /mnt/c/Users/96981/Documents/Codex/2026-07-27/files-mentioned-by-the-user-search/work/GSI-feature/ros2_ws/start_search_world_v1_sitl.sh
```

PX4 显示 ready 后，将 GSI 工程复制或挂载到容器中的 `/tmp/GSI`，构建 bridge，
再运行：

```bash
docker exec visionflow-px4-sitl bash -lc \
  'source /opt/ros/humble/setup.bash && cd /tmp/GSI/ros2_ws && colcon build --symlink-install'

docker exec visionflow-px4-sitl bash \
  /tmp/GSI/ros2_ws/run_search_world_v1_search.sh
```
