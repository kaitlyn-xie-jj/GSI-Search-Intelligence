# Yungu2030 RGB-D/PX4 Validation

## Coverage + YOLO branch

The `feature/yungu-coverage-yolo` configuration selects the deterministic
`CoveragePolicy` lawn-mower route at 10 m. Scan passes are spaced 8 m apart
and sampled every 8 m. The route is reversed when its far endpoint is closer
to the initial UAV pose. Building footprints remain hard obstacles: unsafe
coverage endpoints are removed and connecting flight legs use the existing
building route planner.

After the primary route, searchable cells whose best observation quality is
below 0.5 become deferred gaps. Recovery viewpoints approach each gap from its
center and eight offset directions; unsafe points are filtered and their flight
legs still route around buildings. Cells blocked by semantic-map restrictions
are excluded from the completion denominator.

Vehicle detection is provided by `yolo_target_detector`. It runs an
Ultralytics COCO model over `car`, `bus`, and `truck`, then reuses the RGB-D
centroid localization contract to publish `vision_msgs/Detection3DArray` on
`/gsi/detections`. Prepare the optional vision environment and weight with:

```bash
bash ros2_ws/prepare_yolo_detector.sh
```

The installer pins `setuptools>=77.0.3,<80`: ROS 2 Humble's `colcon-core 0.21`
requires a version below 80, while the current Torch runtime requires at least
77.0.3. It finishes with `pip check` so dependency conflicts fail before a
simulation run.

It also pins `numpy<2` and `opencv-python<4.12`, because the ROS 2 Humble
`cv_bridge` binary uses the NumPy 1.x ABI. Newer NumPy/OpenCV combinations fail
while importing `cv_bridge` with `_ARRAY_API` or `multiarray` errors.

Copy the resulting `artifacts/models/yolo11n.pt` into
`/tmp/GSI/models/yolo11n.pt` in the runtime container. Ultralytics is
AGPL-3.0; deployment and redistribution must be reviewed accordingly.

The included target is now a textured OpenRobotics SUV (CC BY 3.0), while its
historical Gazebo entity name remains `yellow_search_van` for compatibility.
Target color is not part of the detector contract.

The current PX4 model still has one fixed nadir RGB-D camera. Coverage and
YOLO are runnable with that camera. A true 45-degree search view followed by
a 90-degree confirmation view requires a controllable gimbal or a second
camera and is deliberately not claimed by this version.

The default search rectangle is the central open courtyard enclosed by campus
buildings: `x=[140, 210]`, `y=[70, 100]`. Navigation may use the full map
bounds so the UAV can enter from its spawn at `(42, 90)`. The default SUV pose
is the collision-free courtyard point `(175, 85, 0.4)`.
Re-run the geometry, route, ideal-visibility, and recovery audit without
starting Gazebo:

```bash
PYTHONPATH=. python3 run/validate_yungu_coverage_offline.py
```

The central route contains 42 primary viewpoints and at most 28
ideal-visibility recovery viewpoints. All 28 cells become covered with no
restricted or deferred cell. The route is about 405 m before optional recovery
and 644 m with every recovery viewpoint included.

For a watched 600-second run with Gazebo and the live UAV RGB window:

```bash
bash ros2_ws/start_yungu_central_visual_demo.sh
```

The first visual run installs the pinned YOLO stack and commits it into the
local `visionflow-px4:humble-gz` image. Later runs validate that cached stack
and skip pip entirely. Set `GSI_CACHE_YOLO_IMAGE=0` only when an ephemeral
container is explicitly desired.

This scenario joins the local Yungu CAD world to the existing PX4 X500 search
stack. It is a simulator validation using rendered Gazebo RGB-D data, not a
claim about physical camera hardware.

The public input is the Yungu semantic map. Yellow-van poses are passed only at
run time, spawned in Gazebo, and saved in the evaluator artifact directory. The
search policy does not read that file.

## Sensor contract

The dedicated X500 model publishes the normal ROS contract at 10 Hz:

```text
/oakd1/rgb/image
/oakd1/rgb/camera_info
/oakd1/depth/image
/oakd1/depth/camera_info
/oakd1/depth/points
```

Yungu has collision boxes up to 55.1 m tall. This branch deliberately uses
a 10 m flight altitude so the SUV remains large enough in the nadir
image. The search node therefore treats the semantic-map building rectangles
as hard navigation obstacles: it rejects viewpoints within a 3 m inflated
building boundary and routes intersecting flight segments around safe corners.
Intermediate navigation waypoints do not create observations or belief
updates. The RGB-D clipping range remains 90 m.

The expected search footprint is derived from the 60 degree horizontal FOV,
160x120 image geometry, flight altitude, and a 0.95 safety scale. At 10 m the
horizontal width is approximately 11.5 m, and the configured 8 m pass spacing
keeps about 30 percent cross-track overlap. Actual negative evidence still
requires valid projected point-cloud support.

The offboard adapter accepts the Gazebo spawn pose only after five plausible
stationary odometry samples. Losing OFFBOARD after arming, exceeding 5 m/s, or
leaving the map bounds latches a safety stop, disables automatic mode/arming
recovery, and aborts the search with an explicit outcome.

A low count of points projected onto the `z=0` ground plane is recorded as
`insufficient_ground_projection`. It is inconclusive sensor support, not proof
that the target is occluded, and it does not trigger an occlusion-inspection
viewpoint.

## One complete trial

Run this from WSL after Docker Desktop is running:

```bash
cd /mnt/c/Users/96981/Documents/Codex/2026-07-27/files-mentioned-by-the-user-search/work/GSI-feature
bash ros2_ws/run_yungu2030_sensor_trial.sh
```

The default evaluator-only SUV is at `(220, 94, 0.4, 0)`, on the stable
flight corridor previously observed between `(180, 84)` and `(260, 104)`. The UAV
starts at `(42, 90, 0.25)`. Override a
position only through the run environment, for example:

```bash
GSI_TARGET_X_M=72 GSI_TARGET_Y_M=90 GSI_TARGET_Z_M=0.4 \
  bash ros2_ws/run_yungu2030_sensor_trial.sh
```

Do not use the sunken plaza for the first trials: its physical ground is near
`z=-8.7 m`, so it needs a per-region target elevation contract.

Each successful run saves `capture/rgb.mp4`, or `capture/rgb.rgb24` when the
runtime lacks `ffmpeg`, plus one byte-exact depth image and point-cloud message
under `capture/sensor_snapshot/`. Its rosbag records the light-weight clock,
UAV state, detector, command and outcome trace. This avoids an accelerated
Gazebo run silently generating multi-gigabyte point-cloud bags. The capture
manifest gives the raw stream dimensions and rate; topic-rate probes, search
JSONL trace, runtime/simulator logs, the public scenario snapshot and the
private target-pose manifest are retained under
`results/yungu2030_sensor_validation/`.
