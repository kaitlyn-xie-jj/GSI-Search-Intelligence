#!/usr/bin/env bash
set -eo pipefail

ROS2_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENERATED="${ROS2_WS}/simulation/search_world_v1_1/generated"
RUNTIME_LOG="${GSI_RUNTIME_LOG:-/tmp/GSI/results/gazebo_sensor_validation/gsi_search_world_v1_1_runtime.log}"

bash "${ROS2_WS}/generate_search_world_v1_1.sh"
mkdir -p "$(dirname "${RUNTIME_LOG}")"

bash "${ROS2_WS}/run_visionflow_search.sh" \
    search_config:="${GENERATED}/search_params.yaml" \
    sensor_bridge_config:="${GENERATED}/gz_bridge.yaml" \
    "$@" 2>&1 | tee "${RUNTIME_LOG}"
