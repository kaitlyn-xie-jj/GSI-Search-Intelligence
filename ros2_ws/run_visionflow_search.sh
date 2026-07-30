#!/usr/bin/env bash
set -eo pipefail

ROS2_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GSI_ROOT="$(cd "${ROS2_WS}/.." && pwd)"

source /opt/ros/humble/setup.bash
VISIONFLOW_OVERLAY="/workspace/VisionFlow-PX4/thirdparty/install/setup.bash"
if [[ -f "${VISIONFLOW_OVERLAY}" ]]; then
    source "${VISIONFLOW_OVERLAY}"
fi
cd "${ROS2_WS}"

colcon build --symlink-install --packages-select gsi_search_bridge
source install/setup.bash
export PYTHONPATH="${GSI_ROOT}:${PYTHONPATH:-}"

exec ros2 launch gsi_search_bridge visionflow_search.launch.py "$@"
