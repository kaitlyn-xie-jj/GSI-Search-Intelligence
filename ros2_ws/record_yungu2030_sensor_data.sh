#!/usr/bin/env bash
# ROS Humble's setup scripts read optional unset variables, so nounset cannot
# remain enabled while sourcing the runtime environment.
set -eo pipefail

ROS2_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${1:-/tmp/GSI/results/yungu2030_sensor_validation/capture_$(date -u +%Y%m%dT%H%M%SZ)}"
DURATION_S="${GSI_CAPTURE_DURATION_S:-180}"
WIDTH="${GSI_RGB_WIDTH:-160}"
HEIGHT="${GSI_RGB_HEIGHT:-120}"
FPS="${GSI_RGB_FPS:-10}"
ALLOW_RAW_RGB_FALLBACK="${GSI_ALLOW_RAW_RGB_FALLBACK:-0}"

if ! [[ "${DURATION_S}" =~ ^[1-9][0-9]*$ ]]; then
    echo "GSI_CAPTURE_DURATION_S must be a positive whole number." >&2
    exit 2
fi
source /opt/ros/humble/setup.bash
VISIONFLOW_OVERLAY="/workspace/VisionFlow-PX4/thirdparty/install/setup.bash"
if [[ -f "${VISIONFLOW_OVERLAY}" ]]; then
    source "${VISIONFLOW_OVERLAY}"
fi
mkdir -p "${OUTPUT_DIR}"
VIDEO_ARTIFACT="rgb.mp4"
VIDEO_ENCODING="h264_mp4"
if ! command -v ffmpeg >/dev/null 2>&1; then
    if [[ "${ALLOW_RAW_RGB_FALLBACK}" != "1" ]]; then
        echo "ffmpeg is required to write RGB MP4; set GSI_ALLOW_RAW_RGB_FALLBACK=1 to retain RGB24 frames." >&2
        exit 1
    fi
    VIDEO_ARTIFACT="rgb.rgb24"
    VIDEO_ENCODING="raw_rgb24"
fi

printf '{\n  "scenario_id": "yungu2030_rgbd_px4",\n  "started_at_utc": "%s",\n  "rgb": {"topic": "/oakd1/rgb/image", "width": %s, "height": %s, "fps": %s, "artifact": "%s", "encoding": "%s"},\n  "depth_topic": "/oakd1/depth/image",\n  "point_cloud_topic": "/oakd1/depth/points",\n  "odometry_topic": "/mavros/local_position/odom",\n  "detections_topic": "/gsi/detections"\n}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${WIDTH}" "${HEIGHT}" "${FPS}" "${VIDEO_ARTIFACT}" "${VIDEO_ENCODING}" \
    > "${OUTPUT_DIR}/capture_manifest.json"

if [[ "${VIDEO_ENCODING}" == "h264_mp4" ]]; then
    setsid bash -c \
        "python3 '${ROS2_WS}/tools/record_rgb_topic.py' --width '${WIDTH}' --height '${HEIGHT}' --fps '${FPS}' | ffmpeg -y -f rawvideo -pixel_format rgb24 -video_size '${WIDTH}x${HEIGHT}' -framerate '${FPS}' -i - -an -c:v libx264 -preset fast -crf 18 '${OUTPUT_DIR}/${VIDEO_ARTIFACT}'" \
        > "${OUTPUT_DIR}/video_recorder.log" 2>&1 &
else
    setsid python3 "${ROS2_WS}/tools/record_rgb_topic.py" \
        --width "${WIDTH}" --height "${HEIGHT}" --fps "${FPS}" \
        > "${OUTPUT_DIR}/${VIDEO_ARTIFACT}" 2> "${OUTPUT_DIR}/video_recorder.log" &
fi
VIDEO_PID=$!

python3 "${ROS2_WS}/tools/capture_sensor_snapshot.py" \
    --output "${OUTPUT_DIR}/sensor_snapshot" \
    > "${OUTPUT_DIR}/sensor_snapshot.log" 2>&1 &
SNAPSHOT_PID=$!

ros2 bag record --storage sqlite3 -o "${OUTPUT_DIR}/rosbag2" \
    /clock \
    /mavros/local_position/odom \
    /mavros/state \
    /gsi/detections \
    /gsi/uav/goal_pose \
    /gsi/search/outcome \
    > "${OUTPUT_DIR}/rosbag_recorder.log" 2>&1 &
BAG_PID=$!

cleanup() {
    kill -- "-${VIDEO_PID}" >/dev/null 2>&1 || true
    kill "${BAG_PID}" >/dev/null 2>&1 || true
    wait "${SNAPSHOT_PID}" >/dev/null 2>&1 || true
    wait "${VIDEO_PID}" >/dev/null 2>&1 || true
    wait "${BAG_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

{
    ros2 topic info /oakd1/rgb/image || true
    ros2 topic info /oakd1/depth/image || true
    ros2 topic info /oakd1/depth/points || true
    ros2 topic info /mavros/local_position/odom || true
} > "${OUTPUT_DIR}/topic_info.txt" 2>&1
timeout 20 ros2 topic hz /oakd1/rgb/image > "${OUTPUT_DIR}/rgb_rate.txt" 2>&1 || true
timeout 20 ros2 topic hz /oakd1/depth/image > "${OUTPUT_DIR}/depth_rate.txt" 2>&1 || true

sleep "${DURATION_S}"
