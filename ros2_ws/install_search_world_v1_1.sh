#!/usr/bin/env bash
set -eo pipefail

ROS2_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENARIO_ROOT="${ROS2_WS}/simulation/search_world_v1_1"
GENERATED="${SCENARIO_ROOT}/generated"
VISIONFLOW_ROOT="${1:-/home/windylab/workspace/VisionFlow-PX4}"
PROFILE_CONFIG="${VISIONFLOW_ROOT}/docker/gz_sitl_profiles.conf"
WORLD_DIRECTORY="${VISIONFLOW_ROOT}/Tools/simulation/gz/worlds"
MODEL_DIRECTORY="${VISIONFLOW_ROOT}/Tools/simulation/gz/models/x500_gsi_rgbd"
AIRFRAME_DIRECTORY="${VISIONFLOW_ROOT}/ROMFS/px4fmu_common/init.d-posix/airframes"
AIRFRAME_CMAKE="${AIRFRAME_DIRECTORY}/CMakeLists.txt"
MC_APPS="${VISIONFLOW_ROOT}/ROMFS/px4fmu_common/init.d/rc.mc_apps"
GZ_BRIDGE_CPP="${VISIONFLOW_ROOT}/src/modules/simulation/gz_bridge/GZBridge.cpp"
GZ_BRIDGE_HPP="${VISIONFLOW_ROOT}/src/modules/simulation/gz_bridge/GZBridge.hpp"
GZ_BRIDGE_PATCH="${SCENARIO_ROOT}/px4/gz_bridge_duplicate_imu.patch"

bash "${ROS2_WS}/generate_search_world_v1_1.sh"

for required in \
    "${PROFILE_CONFIG}" \
    "${AIRFRAME_CMAKE}" \
    "${WORLD_DIRECTORY}" \
    "${MC_APPS}" \
    "${GZ_BRIDGE_CPP}" \
    "${GZ_BRIDGE_HPP}" \
    "${GZ_BRIDGE_PATCH}"; do
    if [[ ! -e "${required}" ]]; then
        echo "VisionFlow-PX4 tree is incomplete at: ${VISIONFLOW_ROOT}" >&2
        exit 1
    fi
done

if ! grep -q '_last_imu_timestamp' "${GZ_BRIDGE_CPP}" ||
   ! grep -q '_last_imu_timestamp' "${GZ_BRIDGE_HPP}"; then
    patch --batch --forward -d "${VISIONFLOW_ROOT}" -p1 < "${GZ_BRIDGE_PATCH}"
fi
if ! grep -q 'timestamp <= _last_imu_timestamp' "${GZ_BRIDGE_CPP}" ||
   ! grep -q 'hrt_abstime _last_imu_timestamp{};' "${GZ_BRIDGE_HPP}"; then
    echo "Could not install the Gazebo duplicate-IMU guard." >&2
    exit 1
fi

# VisionFlow normally replaces the stock PX4 multicopter controllers with its
# Pregme controllers. The standard X500 used by SearchWorld V1.1 needs the
# stock controller chain, while existing VisionFlow airframes must remain
# unchanged. Keep this runtime selection scoped to the GSI airframe ID.
MC_APPS_TEMP="$(mktemp)"
awk '
    function print_attitude_block() {
        print "# BEGIN GSI SearchWorld V1.1 attitude controllers (managed)"
        print "if param compare -s SYS_AUTOSTART 4010 || param compare -s SYS_AUTOSTART 4011"
        print "then"
        print "\tmc_rate_control start"
        print "\tmc_att_control start"
        print "else"
        print "\tpregme_att_control start"
        print "fi"
        print "# END GSI SearchWorld V1.1 attitude controllers (managed)"
    }
    function print_position_block() {
        print "# BEGIN GSI SearchWorld V1.1 position controller (managed)"
        print "if param compare -s SYS_AUTOSTART 4010 || param compare -s SYS_AUTOSTART 4011"
        print "then"
        print "\tmc_pos_control start"
        print "else"
        print "\tpregme_pos_control start"
        print "fi"
        print "# END GSI SearchWorld V1.1 position controller (managed)"
    }
    /^# BEGIN GSI SearchWorld V1\.1 attitude controllers \(managed\)$/ {
        print_attitude_block(); managed="attitude"; next
    }
    /^# BEGIN GSI SearchWorld V1\.1 position controller \(managed\)$/ {
        print_position_block(); managed="position"; next
    }
    managed == "attitude" && /^# END GSI SearchWorld V1\.1 attitude controllers \(managed\)$/ {
        managed=""; next
    }
    managed == "position" && /^# END GSI SearchWorld V1\.1 position controller \(managed\)$/ {
        managed=""; next
    }
    managed != "" { next }
    /^# Start Official Multicopter Rate Controller\.$/ {
        print_attitude_block(); legacy="attitude"; next
    }
    legacy == "attitude" && /^pregme_att_control start$/ {
        legacy=""; next
    }
    /^# Start Official Position Controller\.$/ {
        print_position_block(); legacy="position"; next
    }
    legacy == "position" && /^pregme_pos_control start$/ {
        legacy=""; next
    }
    legacy != "" { next }
    { print }
' "${MC_APPS}" > "${MC_APPS_TEMP}"

if ! grep -q '^# BEGIN GSI SearchWorld V1\.1 attitude controllers (managed)$' "${MC_APPS_TEMP}" ||
   ! grep -q '^# BEGIN GSI SearchWorld V1\.1 position controller (managed)$' "${MC_APPS_TEMP}"; then
    echo "Could not install the SearchWorld V1.1 PX4 controller selection." >&2
    rm -f "${MC_APPS_TEMP}"
    exit 1
fi
install -m 0755 "${MC_APPS_TEMP}" "${MC_APPS}"
rm -f "${MC_APPS_TEMP}"

install -m 0644 "${GENERATED}/gsi_search_world_v1_1.sdf" \
    "${WORLD_DIRECTORY}/gsi_search_world_v1_1.sdf"
install -d "${MODEL_DIRECTORY}"
install -m 0644 "${SCENARIO_ROOT}/models/x500_gsi_rgbd/model.config" \
    "${MODEL_DIRECTORY}/model.config"
install -m 0644 "${SCENARIO_ROOT}/models/x500_gsi_rgbd/model.sdf" \
    "${MODEL_DIRECTORY}/model.sdf"
install -m 0755 "${SCENARIO_ROOT}/airframes/4010_gz_x500_gsi_rgbd" \
    "${AIRFRAME_DIRECTORY}/4010_gz_x500_gsi_rgbd"

if ! grep -q '4010_gz_x500_gsi_rgbd' "${AIRFRAME_CMAKE}"; then
    CMAKE_TEMP="$(mktemp)"
    awk '
        /^\)$/ && !inserted { print "\t4010_gz_x500_gsi_rgbd"; inserted=1 }
        { print }
    ' "${AIRFRAME_CMAKE}" > "${CMAKE_TEMP}"
    install -m 0644 "${CMAKE_TEMP}" "${AIRFRAME_CMAKE}"
    rm -f "${CMAKE_TEMP}"
fi

PROFILE_TEMP="$(mktemp)"
awk '
    /^# BEGIN GSI SearchWorld V1\.1 profile \(managed\)\.$/ { managed=1; next }
    /^# END GSI SearchWorld V1\.1 profile \(managed\)\.$/ { managed=0; next }
    !managed { print }
' "${PROFILE_CONFIG}" > "${PROFILE_TEMP}"
printf '\n' >> "${PROFILE_TEMP}"
cat "${GENERATED}/visionflow_profile.conf" >> "${PROFILE_TEMP}"
install -m 0644 "${PROFILE_TEMP}" "${PROFILE_CONFIG}"
rm -f "${PROFILE_TEMP}"

# Gazebo worlds and PX4 make targets are discovered at CMake configure time.
touch "${VISIONFLOW_ROOT}/src/modules/simulation/gz_bridge/CMakeLists.txt"
touch "${AIRFRAME_CMAKE}"

echo "Installed SearchWorld V1.1 world, x500_gsi_rgbd model, PX4 airframe, controller selection, and IMU guard."
echo "Start with:"
echo "  cd ${VISIONFLOW_ROOT}"
echo "  bash docker/run_gz_sitl.sh --profile \"GSI SearchWorld V1.1\""
