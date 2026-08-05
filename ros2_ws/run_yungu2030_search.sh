#!/usr/bin/env bash
set -eo pipefail

ROS2_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENARIO_DIR="${ROS2_WS}/simulation/yungu2030_v1"
RUNTIME_LOG="${GSI_RUNTIME_LOG:-/tmp/GSI/results/yungu2030_sensor_validation/runtime.log}"
mkdir -p "$(dirname "${RUNTIME_LOG}")"

bash "${ROS2_WS}/run_visionflow_search.sh" \
    search_config:="${SCENARIO_DIR}/yungu_search_params.yaml" \
    sensor_bridge_config:="${SCENARIO_DIR}/gz_bridge.yaml" \
    "$@" 2>&1 | tee "${RUNTIME_LOG}"
