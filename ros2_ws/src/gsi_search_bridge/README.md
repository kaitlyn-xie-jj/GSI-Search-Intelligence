# GSI Search Bridge for ROS 2 Humble and Gazebo Harmonic

This package keeps ROS and Gazebo dependencies outside the platform-neutral
`modules/search_intelligence` package. It implements the first live loop:

```text
ActiveSearchPolicy -> PoseStamped goal -> UAV controller -> Gazebo sensors
-> SearchSensorFrame -> SearchObservation -> Bayesian update -> next goal
```

## Version note

ROS 2 Humble normally ships with Gazebo Fortress, while Gazebo Harmonic normally
pairs with newer ROS distributions. For Humble + Harmonic, install or build a
`ros_gz` release that targets Harmonic and confirm that `ros_gz_bridge` can load
the message pairs in `config/gz_bridge.yaml`. Do not install the default Fortress
bridge alongside a Harmonic bridge in the same environment.

## Supported interfaces

Two launch paths are retained:

- `gazebo_search.launch.py` is the original generic Gazebo baseline. It uses
  `/uav/cmd_vel` and `config/gz_bridge.yaml`.
- `visionflow_search.launch.py` is the PX4 integration for VisionFlow `Entity 1`.
  It uses MAVROS Offboard and the real OakD topics from
  `laboratory_landingbox` / `q940_ti_gripper4_0`.

The VisionFlow online data path is:

```text
/mavros/local_position/odom  -> actual robot pose (ENU)
/mavros/imu/data             -> motion quality
/oakd1/rgb/*                 -> RGB + intrinsics
/oakd1/depth/*               -> depth + point cloud
/gsi/detections              -> replaceable detector output
/gsi/uav/goal_pose           -> GSI viewpoint goal (ENU)
/mavros/setpoint_position/local -> PX4 Offboard setpoint
```

MAVROS performs PX4 NED to ROS ENU conversion. GSI never sends NED values. The
Gazebo topic `/model/q940_ti_gripper4_0/odometry` is deliberately absent from the
online launch and may only be recorded by an evaluator.

Target detectors publish `vision_msgs/msg/Detection3DArray` with map-frame poses.
The search policy only consumes this standard interface and does not depend on a
specific perception model.

## Build

From an Ubuntu ROS 2 Humble shell:

```bash
cd <GSI>/ros2_ws
source /opt/ros/humble/setup.bash
source /workspace/VisionFlow-PX4/thirdparty/install/setup.bash
colcon build --symlink-install --packages-select gsi_search_bridge
source install/setup.bash
```

The GSI repository root must be on `PYTHONPATH`, or the core Python package must be
installed into the ROS environment.

## Generic Gazebo launch

Start the matching generic Harmonic world and UAV model first, then run:

```bash
ros2 launch gsi_search_bridge gazebo_search.launch.py
```

Verify the interface before enabling search:

```bash
ros2 topic hz /uav/odom
ros2 topic hz /uav/camera/image_raw
ros2 topic hz /uav/depth/image_raw
ros2 topic hz /uav/lidar/points
ros2 topic echo /uav/detections --once
```

## VisionFlow PX4 launch

Start VisionFlow without rebuilding the image:

```bash
cd ~/workspace/VisionFlow-PX4
bash docker/run_gz_sitl.sh --profile "Entity 1"
```

In the same ROS/Docker network, build this package with the GSI root on
`PYTHONPATH`, then launch:

```bash
cd <GSI>/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select gsi_search_bridge
source install/setup.bash
export PYTHONPATH=<GSI>:$PYTHONPATH
ros2 launch gsi_search_bridge visionflow_search.launch.py
```

The repository also provides the equivalent build-and-launch helper:

```bash
bash <GSI>/ros2_ws/run_visionflow_search.sh
```

The launch starts four adapters around the platform-neutral policy:

1. MAVROS connects to PX4 on `udp://:14540@localhost:14557`.
2. `ros_gz_bridge` exposes OakD1 RGB/depth/point-cloud data and `/clock`.
3. `mavros_offboard_controller` pre-streams setpoints, enters OFFBOARD, arms the
   simulated vehicle, climbs vertically to search altitude, then releases XY
   goals.
4. `color_target_detector` provides a dependency-light yellow-target simulation
   baseline. It publishes the same interface expected from a future
   YOLO-World/Grounding-DINO detector.

For a MAVROS instance that is already running:

```bash
ros2 launch gsi_search_bridge visionflow_search.launch.py start_mavros:=false
```

Verify live input and control state:

```bash
ros2 topic hz /mavros/local_position/odom
ros2 topic echo /mavros/state --once
ros2 topic hz /oakd1/rgb/image
ros2 topic hz /oakd1/depth/image
ros2 topic hz /mavros/setpoint_position/local
```

`config/visionflow_search_params.yaml` intentionally uses a small 12 m by 12 m
lab search area at 3 m altitude. `require_detections` and `require_point_cloud`
are enabled, so an observation is only recorded when perception, depth, and pose
are fresh.

To run the positive-detection fixture, create the static yellow van after Gazebo
is ready and before launching GSI:

```bash
bash <GSI>/ros2_ws/spawn_visionflow_target.sh
bash <GSI>/ros2_ws/run_visionflow_search.sh
```

For negative-observation experiments without the baseline detector:

```bash
ros2 launch gsi_search_bridge visionflow_search.launch.py \
  start_baseline_detector:=false
```

The search outcome is published as JSON on `/gsi/search/outcome`.

## Parameterized SearchWorld V1

SearchWorld V1 generates a controlled 80 m by 60 m outdoor search benchmark
from one JSON parameter file. Gazebo receives physical SDF geometry, GSI receives
only public semantic regions and a task-conditioned prior, and the target pose is
kept in an evaluator-only artifact. See
`ros2_ws/simulation/search_world_v1/README.md` for the data contract.

Generate and install it into the mounted VisionFlow checkout:

```bash
bash /tmp/GSI/ros2_ws/install_search_world_v1.sh /workspace/VisionFlow-PX4
```

Start Gazebo/PX4 from WSL with the installed profile:

```bash
cd ~/workspace/VisionFlow-PX4
bash docker/run_gz_sitl.sh --profile "GSI SearchWorld V1"
```

Then start the SearchWorld-specific sensor bridge and policy configuration:

```bash
bash /tmp/GSI/ros2_ws/run_search_world_v1_search.sh
```

Leaving `semantic_map_path` and `search_prior_path` empty retains the uniform M7
baseline. SearchWorld sets both paths, annotates restricted/building cells before
candidate generation, and records matched/unmatched prior labels in policy state.

## Current boundary

M7-B projects the OakD point cloud through its SDF extrinsic and the live MAVROS
pose, then maps near-ground points into `SearchGrid` cells. Sensor skew uses ROS
receipt time because Gazebo camera headers and MAVROS headers are in different
clock domains; original header times remain in observation metadata. The included
color detector is only an interface-test baseline, not the proposed open-world
perception method. This launch is not an obstacle-avoidance system; viewpoints
must remain above known lab obstacles and inside the configured area.
