#!/usr/bin/env bash
set -eo pipefail

ROS2_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENARIO="${1:-all}"
SCENARIOS=(campus industrial suburban)

if [[ "${SCENARIO}" != "all" ]]; then
    SCENARIOS=("${SCENARIO}")
fi

export PYTHONPATH="${ROS2_WS}/src/gsi_search_bridge:${PYTHONPATH:-}"
for name in "${SCENARIOS[@]}"; do
    config="${ROS2_WS}/simulation/search_world_v2/${name}/scenario.json"
    output="${ROS2_WS}/simulation/search_world_v2/${name}/generated"
    if [[ ! -f "${config}" ]]; then
        echo "Unknown SearchWorld V2 scenario: ${name}" >&2
        exit 2
    fi
    echo "[generate] ${name}"
    python3 -m gsi_search_bridge.search_world_generator \
        --config "${config}" \
        --output "${output}"
done
