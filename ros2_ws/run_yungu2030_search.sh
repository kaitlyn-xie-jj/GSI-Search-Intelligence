#!/usr/bin/env bash
set -eo pipefail

ROS2_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENARIO_DIR="${ROS2_WS}/simulation/yungu2030_v1"
RUNTIME_LOG="${GSI_RUNTIME_LOG:-/tmp/GSI/results/yungu2030_sensor_validation/runtime.log}"
SEARCH_TIME_BUDGET_S="${GSI_SEARCH_TIME_BUDGET_S:-12000}"
START_YOLO_DETECTOR="${GSI_START_YOLO_DETECTOR:-true}"
START_COLOR_DETECTOR="${GSI_START_COLOR_DETECTOR:-false}"
mkdir -p "$(dirname "${RUNTIME_LOG}")"

bash "${ROS2_WS}/run_visionflow_search.sh" \
    search_config:="${SCENARIO_DIR}/yungu_search_params.yaml" \
    sensor_bridge_config:="${SCENARIO_DIR}/gz_bridge.yaml" \
    search_time_budget_s:="${SEARCH_TIME_BUDGET_S}" \
    start_yolo_detector:="${START_YOLO_DETECTOR}" \
    start_baseline_detector:="${START_COLOR_DETECTOR}" \
    "$@" 2>&1 | tee "${RUNTIME_LOG}"
