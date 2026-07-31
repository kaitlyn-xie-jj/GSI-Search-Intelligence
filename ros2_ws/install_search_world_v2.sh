#!/usr/bin/env bash
set -eo pipefail

ROS2_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VISIONFLOW_ROOT="${1:-/home/windylab/workspace/VisionFlow-PX4}"
PROFILE_CONFIG="${VISIONFLOW_ROOT}/docker/gz_sitl_profiles.conf"
WORLD_DIRECTORY="${VISIONFLOW_ROOT}/Tools/simulation/gz/worlds"
GZ_CMAKE="${VISIONFLOW_ROOT}/src/modules/simulation/gz_bridge/CMakeLists.txt"
MODEL_DIRECTORY="${VISIONFLOW_ROOT}/Tools/simulation/gz/models/x500_gsi_rgbd_nadir"
AIRFRAME_DIRECTORY="${VISIONFLOW_ROOT}/ROMFS/px4fmu_common/init.d-posix/airframes"
AIRFRAME_CMAKE="${AIRFRAME_DIRECTORY}/CMakeLists.txt"
RUN_GZ_SITL="${VISIONFLOW_ROOT}/docker/run_gz_sitl.sh"
HEADLESS_PATCH="${ROS2_WS}/simulation/search_world_v2/visionflow_headless.patch"

# Reuse the validated X500 airframe, official controller selection, and
# duplicate-IMU guard maintained by V1.1.
bash "${ROS2_WS}/install_search_world_v1_1.sh" "${VISIONFLOW_ROOT}"
bash "${ROS2_WS}/generate_search_world_v2.sh" all

if ! grep -q -- '-e HEADLESS="${HEADLESS:-0}"' "${RUN_GZ_SITL}"; then
    patch --batch --forward -d "${VISIONFLOW_ROOT}" -p1 < "${HEADLESS_PATCH}"
fi
if ! grep -q -- '-e HEADLESS="${HEADLESS:-0}"' "${RUN_GZ_SITL}"; then
    echo "Could not install the VisionFlow HEADLESS passthrough." >&2
    exit 1
fi

for scenario in campus industrial suburban; do
    generated="${ROS2_WS}/simulation/search_world_v2/${scenario}/generated"
    world="gsi_search_world_v2_${scenario}"
    install -m 0644 "${generated}/${world}.sdf" "${WORLD_DIRECTORY}/${world}.sdf"
done

install -d "${MODEL_DIRECTORY}"
install -m 0644 "${ROS2_WS}/simulation/search_world_v2/models/x500_gsi_rgbd_nadir/model.config" \
    "${MODEL_DIRECTORY}/model.config"
install -m 0644 "${ROS2_WS}/simulation/search_world_v2/models/x500_gsi_rgbd_nadir/model.sdf" \
    "${MODEL_DIRECTORY}/model.sdf"
install -m 0755 "${ROS2_WS}/simulation/search_world_v2/airframes/4011_gz_x500_gsi_rgbd_nadir" \
    "${AIRFRAME_DIRECTORY}/4011_gz_x500_gsi_rgbd_nadir"
if ! grep -q '4011_gz_x500_gsi_rgbd_nadir' "${AIRFRAME_CMAKE}"; then
    airframe_temp="$(mktemp)"
    awk '
        /^\)$/ && !inserted { print "\t4011_gz_x500_gsi_rgbd_nadir"; inserted=1 }
        { print }
    ' "${AIRFRAME_CMAKE}" > "${airframe_temp}"
    install -m 0644 "${airframe_temp}" "${AIRFRAME_CMAKE}"
    rm -f "${airframe_temp}"
fi

profile_temp="$(mktemp)"
awk '
    /^# BEGIN GSI SearchWorld V2 .* profile \(managed\)\.$/ { managed=1; next }
    /^# END GSI SearchWorld V2 .* profile \(managed\)\.$/ { managed=0; next }
    !managed { print }
' "${PROFILE_CONFIG}" > "${profile_temp}"
for scenario in campus industrial suburban; do
    cat "${ROS2_WS}/simulation/search_world_v2/${scenario}/generated/visionflow_profile.conf" \
        >> "${profile_temp}"
done
install -m 0644 "${profile_temp}" "${PROFILE_CONFIG}"
rm -f "${profile_temp}"

# Gazebo world targets are enumerated during CMake configure.
touch "${GZ_CMAKE}"
touch "${AIRFRAME_CMAKE}"
echo "Installed SearchWorld V2 scenarios and HEADLESS passthrough."
echo "List profiles with: bash docker/run_gz_sitl.sh --list"
