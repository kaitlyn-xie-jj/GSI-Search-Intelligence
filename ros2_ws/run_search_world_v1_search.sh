#!/usr/bin/env bash
set -eo pipefail

ROS2_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENERATED="${ROS2_WS}/simulation/search_world_v1/generated"

bash "${ROS2_WS}/generate_search_world_v1.sh"

exec bash "${ROS2_WS}/run_visionflow_search.sh" \
    search_config:="${GENERATED}/search_params.yaml" \
    sensor_bridge_config:="${GENERATED}/gz_bridge.yaml" \
    "$@"
