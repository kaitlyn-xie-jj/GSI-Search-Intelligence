#!/usr/bin/env bash
set -eo pipefail

MODEL_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/simulation/yellow_search_van.sdf"
WORLD_NAME="${1:-laboratory_landingbox}"
X_M="${2:-3.5}"
Y_M="${3:-0.0}"
Z_M="${4:-0.4}"
YAW_RAD="${5:-0.0}"
ENTITY_NAME="${6:-yellow_search_van}"

if [[ ! -f "${MODEL_FILE}" ]]; then
    echo "Target model not found: ${MODEL_FILE}" >&2
    exit 1
fi

if ! [[ "${WORLD_NAME}" =~ ^[A-Za-z0-9_]+$ ]] || \
   ! [[ "${ENTITY_NAME}" =~ ^[A-Za-z0-9_]+$ ]]; then
    echo "World and entity names may contain only letters, numbers, and underscores." >&2
    exit 2
fi
for value in "${X_M}" "${Y_M}" "${Z_M}" "${YAW_RAD}"; do
    if ! [[ "${value}" =~ ^-?[0-9]+([.][0-9]+)?$ ]]; then
        echo "Target pose values must be finite decimal numbers." >&2
        exit 2
    fi
done

HALF_YAW="$(awk -v yaw="${YAW_RAD}" 'BEGIN { printf "%.12f", yaw / 2.0 }')"
QZ="$(awk -v half_yaw="${HALF_YAW}" 'BEGIN { printf "%.12f", sin(half_yaw) }')"
QW="$(awk -v half_yaw="${HALF_YAW}" 'BEGIN { printf "%.12f", cos(half_yaw) }')"

gz service \
    -s "/world/${WORLD_NAME}/create" \
    --reqtype gz.msgs.EntityFactory \
    --reptype gz.msgs.Boolean \
    --timeout 5000 \
    --req "sdf_filename: '${MODEL_FILE}', name: '${ENTITY_NAME}', allow_renaming: false, pose: { position: { x: ${X_M}, y: ${Y_M}, z: ${Z_M} }, orientation: { z: ${QZ}, w: ${QW} } }"
