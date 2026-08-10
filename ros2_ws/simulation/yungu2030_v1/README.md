# Yungu2030 RGB-D/PX4 Validation

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

Yungu has collision boxes up to 55.1 m tall. This validation deliberately uses
a 30 m flight altitude so the yellow van remains large enough in the nadir
image. The search node therefore treats the semantic-map building rectangles
as hard navigation obstacles: it rejects viewpoints within a 3 m inflated
building boundary and routes intersecting flight segments around safe corners.
Intermediate navigation waypoints do not create observations or belief
updates. The RGB-D clipping range remains 90 m.

The expected search footprint is derived from the 60 degree horizontal FOV,
160x120 image geometry, flight altitude, and a 0.95 safety scale. At 30 m this
is approximately `+/-16.45 m` by `+/-12.34 m`; the legacy 30 m radius is not
used. Actual negative evidence still requires valid projected point-cloud
support.

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

The default evaluator-only yellow van is at `(220, 94, 0.4, 0)`, on the stable
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
