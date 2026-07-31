#!/usr/bin/env bash
set -eo pipefail

ROS2_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENARIO="${1:-campus}"
shift || true
GENERATED="${ROS2_WS}/simulation/search_world_v2/${SCENARIO}/generated"

if [[ ! -f "${GENERATED}/search_params.yaml" ]]; then
    bash "${ROS2_WS}/generate_search_world_v2.sh" "${SCENARIO}"
fi
RUNTIME_LOG="${GSI_RUNTIME_LOG:-/tmp/GSI/results/gazebo_sensor_validation/gsi_search_world_v2_${SCENARIO}_runtime.log}"
mkdir -p "$(dirname "${RUNTIME_LOG}")"

bash "${ROS2_WS}/run_visionflow_search.sh" \
    search_config:="${GENERATED}/search_params.yaml" \
    sensor_bridge_config:="${GENERATED}/gz_bridge.yaml" \
    "$@" 2>&1 | tee "${RUNTIME_LOG}"
