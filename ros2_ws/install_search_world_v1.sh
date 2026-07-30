#!/usr/bin/env bash
set -eo pipefail

ROS2_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENERATED="${ROS2_WS}/simulation/search_world_v1/generated"
VISIONFLOW_ROOT="${1:-/workspace/VisionFlow-PX4}"
WORLD_FILE="${GENERATED}/gsi_search_world_v1.sdf"
PROFILE_SNIPPET="${GENERATED}/visionflow_profile.conf"
PROFILE_CONFIG="${VISIONFLOW_ROOT}/docker/gz_sitl_profiles.conf"
WORLD_DIRECTORY="${VISIONFLOW_ROOT}/Tools/simulation/gz/worlds"

bash "${ROS2_WS}/generate_search_world_v1.sh"

if [[ ! -d "${WORLD_DIRECTORY}" || ! -f "${PROFILE_CONFIG}" ]]; then
    echo "VisionFlow-PX4 tree not found at: ${VISIONFLOW_ROOT}" >&2
    exit 1
fi

install -m 0644 "${WORLD_FILE}" "${WORLD_DIRECTORY}/gsi_search_world_v1.sdf"
# Gazebo make targets are generated from the world glob at CMake configure time.
touch "${VISIONFLOW_ROOT}/src/modules/simulation/gz_bridge/CMakeLists.txt"

if [[ ! -f "${PROFILE_CONFIG}.before-gsi-searchworld-v1" ]]; then
    cp -a "${PROFILE_CONFIG}" "${PROFILE_CONFIG}.before-gsi-searchworld-v1"
fi

PROFILE_TEMP="$(mktemp)"
awk '
    /^# BEGIN GSI SearchWorld V1 profile \(managed\)\.$/ { managed=1; next }
    /^# END GSI SearchWorld V1 profile \(managed\)\.$/ { managed=0; next }
    /^# Installed by GSI SearchWorld V1\.$/ { exit }
    !managed { print }
' "${PROFILE_CONFIG}" > "${PROFILE_TEMP}"
printf '\n' >> "${PROFILE_TEMP}"
sed -n '/# BEGIN GSI SearchWorld V1 profile (managed)\./,$p' "${PROFILE_SNIPPET}" \
    >> "${PROFILE_TEMP}"
install -m 0644 "${PROFILE_TEMP}" "${PROFILE_CONFIG}"
rm -f "${PROFILE_TEMP}"

echo "Installed world: ${WORLD_DIRECTORY}/gsi_search_world_v1.sdf"
echo "Installed profile: GSI SearchWorld V1"
echo "PX4 CMake regeneration requested for the new world target"
echo "Start from the WSL VisionFlow checkout with:"
echo "  bash docker/run_gz_sitl.sh --profile \"GSI SearchWorld V1\""
