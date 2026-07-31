# GSI SearchWorld V1.1

SearchWorld V1.1 keeps the deterministic V1 environment and replaces the
q940/manipulator/four-camera vehicle with a standard PX4 X500 carrying one
640 x 480 RGB-D sensor at 10 Hz. The public ROS contract remains unchanged:

```text
/oakd1/rgb/image
/oakd1/rgb/camera_info
/oakd1/depth/image
/oakd1/depth/camera_info
/oakd1/depth/points
```

The generated Gazebo bridge maps those names to `/gsi/rgbd/*`. This isolates
airframe and sensor resource use without changing the search policy, detector,
belief update, or `SearchObservation` adapter.

The installer also adds an idempotent runtime selection in PX4's
`rc.mc_apps`: only `SYS_AUTOSTART=4010` uses `mc_rate_control`,
`mc_att_control`, and `mc_pos_control`. Existing VisionFlow airframes continue
to use `pregme_att_control` and `pregme_pos_control`. It also installs a
duplicate-IMU guard in the VisionFlow Gazebo bridge so an interrupted transport
publish cannot forward the same lockstep timestamp twice.

## Install and run

From the mounted GSI checkout in WSL:

```bash
cd /mnt/c/Users/96981/Documents/Codex/2026-07-27/files-mentioned-by-the-user-search/work/GSI-feature
bash ros2_ws/install_search_world_v1_1.sh /home/windylab/workspace/VisionFlow-PX4

cd /home/windylab/workspace/VisionFlow-PX4
bash docker/run_gz_sitl.sh --profile "GSI SearchWorld V1.1"
```

In a second WSL terminal, start the GSI bridge and search:

```bash
cd /tmp/GSI/ros2_ws
bash run_search_world_v1_1_search.sh
```

The launcher mirrors all ROS/MAVROS output to
`/tmp/GSI/results/gazebo_sensor_validation/gsi_search_world_v1_1_runtime.log`;
the stability gate includes this log in its final error classification.

## 25-minute stability gate

With the simulator and, optionally, the GSI bridge running:

```bash
cd /tmp/GSI/ros2_ws
bash run_search_world_v1_1_stability.sh
```

The default duration is 1,500 seconds with one sample every 10 seconds. MAVROS
connection, armed/OFFBOARD state, SearchWorld XY bounds, and at least two real
`SearchObservation` events are part of the same automatic gate. Each
run writes `resource_timeseries.csv` and `summary.json` below
`results/gazebo_stability/search_world_v1_1/<timestamp>/`. The command exits
zero only when the run passes.

Default failure thresholds are explicit CLI options:

```text
maximum container memory       8192 MiB
maximum total memory growth    2560 MiB
maximum fitted memory slope     100 MiB/min
critical PX4/Gazebo errors        0
```

Memory growth and slope are fitted after a 120-second startup warmup so model
loading and shader allocation are not misclassified as a long-term leak.

The collector also requires the container, Gazebo, PX4, and all four Gazebo
RGB-D topic streams to remain available. Activity/rate probes run once per
minute. Use `--help` to adjust thresholds for a documented experiment; do not
change thresholds after observing a result.

The current GUI-enabled 25-minute result passes flight, sensor, and PX4/IMU
checks but fails the memory growth and slope limits in `gz sim -g`. See the
[2026-07-31 validation record](../../../docs/zh/validation/searchworld-v1-1-2026-07-31.md).
