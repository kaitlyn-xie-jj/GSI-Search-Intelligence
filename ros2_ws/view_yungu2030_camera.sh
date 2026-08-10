#!/usr/bin/env bash
set -eo pipefail

ROS2_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_NAME="${1:-${GSI_SITL_CONTAINER:-visionflow-px4-sitl}}"
WIDTH="${GSI_RGB_WIDTH:-160}"
HEIGHT="${GSI_RGB_HEIGHT:-120}"
FPS="${GSI_RGB_FPS:-10}"

docker exec "${CONTAINER_NAME}" bash -lc \
    "source /opt/ros/humble/setup.bash; source /workspace/VisionFlow-PX4/thirdparty/install/setup.bash; python3 /tmp/GSI/ros2_ws/tools/record_rgb_topic.py --width '${WIDTH}' --height '${HEIGHT}' --fps '${FPS}'" \
    | python3 "${ROS2_WS}/tools/show_rgb24_stream.py" \
        --width "${WIDTH}" --height "${HEIGHT}" \
        --title "Yungu UAV RGB camera (Esc/Q to close)"
