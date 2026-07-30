#!/usr/bin/env bash
set -eo pipefail

ROS2_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-${ROS2_WS}/simulation/search_world_v1/search_world_v1.json}"
OUTPUT="${2:-${ROS2_WS}/simulation/search_world_v1/generated}"

export PYTHONPATH="${ROS2_WS}/src/gsi_search_bridge:${PYTHONPATH:-}"
python3 -m gsi_search_bridge.search_world_generator \
    --config "${CONFIG}" \
    --output "${OUTPUT}"
