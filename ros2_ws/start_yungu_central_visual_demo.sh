#!/usr/bin/env bash
set -euo pipefail

ROS2_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export HEADLESS=0
export GSI_SHOW_CAMERA=1
export GSI_SEARCH_TIMEOUT_S=600
export GSI_CAPTURE_DURATION_S=600
export GSI_TARGET_X_M=175
export GSI_TARGET_Y_M=85
export GSI_TARGET_Z_M=0.4

exec bash "${ROS2_WS}/run_yungu2030_sensor_trial.sh"
