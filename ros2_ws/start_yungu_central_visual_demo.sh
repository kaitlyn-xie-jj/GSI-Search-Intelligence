#!/usr/bin/env bash
set -euo pipefail

ROS2_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export HEADLESS=0
export GSI_SHOW_CAMERA=0
export GSI_OPERATOR_UI=1
export GSI_YUNGU_GUI_CONFIG="${ROS2_WS}/simulation/yungu2030_v1/gui_drone_follow.config"
export GSI_SEARCH_TIMEOUT_S=600
export GSI_CAPTURE_DURATION_S=600
export GSI_TARGET_X_M=200
export GSI_TARGET_Y_M=120
export GSI_TARGET_Z_M=0.28

echo "Starting Yungu central coverage demo"
echo "  onboard camera: /gsi/rgbd/image (Gazebo Image Display)"
echo "  planning log: live in this terminal and saved under results/"
echo "  search: 600 s, 1.5 m/s, area x=[90,140] y=[57,72]"
echo "  target: north flat road at (115,65); operator confirmation is enabled"
echo "  confirmation: separate terminal + 90-degree nadir camera window"

exec bash "${ROS2_WS}/run_yungu2030_sensor_trial.sh"
