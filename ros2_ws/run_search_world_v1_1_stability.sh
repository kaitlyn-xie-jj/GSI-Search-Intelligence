#!/usr/bin/env bash
set -eo pipefail

ROS2_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GSI_ROOT="$(cd "${ROS2_WS}/.." && pwd)"
OUTPUT_ROOT="${GSI_STABILITY_OUTPUT:-${GSI_ROOT}/results/gazebo_stability/search_world_v1_1}"

export PYTHONPATH="${ROS2_WS}/src/gsi_search_bridge:${PYTHONPATH:-}"
exec python3 -m gsi_search_bridge.stability \
    --output-root "${OUTPUT_ROOT}" \
    --require-mavros \
    --require-flight \
    "$@"
