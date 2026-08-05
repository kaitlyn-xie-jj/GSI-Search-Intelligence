#!/usr/bin/env bash
set -euo pipefail

ROS2_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GSI_ROOT="$(cd "${ROS2_WS}/.." && pwd)"
VISIONFLOW_ROOT="${1:-/home/windylab/workspace/VisionFlow-PX4}"
CONTAINER_VISIONFLOW_ROOT="${GSI_CONTAINER_VISIONFLOW_ROOT:-/workspace/VisionFlow-PX4}"
SCENARIO_DIR="${ROS2_WS}/simulation/yungu2030_v1"
ASSET_DIR="${GSI_ROOT}/data/yungu2030_v1"
WORLD_NAME="yungu2030_local_origin"
PROFILE_CONFIG="${VISIONFLOW_ROOT}/docker/gz_sitl_profiles.conf"
WORLD_DIRECTORY="${VISIONFLOW_ROOT}/Tools/simulation/gz/worlds"
MODEL_DIRECTORY="${VISIONFLOW_ROOT}/Tools/simulation/gz/models/x500_gsi_rgbd_nadir_longrange"
AIRFRAME_DIRECTORY="${VISIONFLOW_ROOT}/ROMFS/px4fmu_common/init.d-posix/airframes"
AIRFRAME_CMAKE="${AIRFRAME_DIRECTORY}/CMakeLists.txt"
MC_APPS="${VISIONFLOW_ROOT}/ROMFS/px4fmu_common/init.d/rc.mc_apps"
GZ_CMAKE="${VISIONFLOW_ROOT}/src/modules/simulation/gz_bridge/CMakeLists.txt"

for path in \
    "${ASSET_DIR}/yungu_local_origin.sdf" \
    "${ASSET_DIR}/semantic_map.json" \
    "${ASSET_DIR}/meshes/yungu_visual_local_origin.glb" \
    "${SCENARIO_DIR}/models/x500_gsi_rgbd_nadir_longrange/model.sdf" \
    "${SCENARIO_DIR}/airframes/4012_gz_x500_gsi_rgbd_nadir_longrange" \
    "${PROFILE_CONFIG}" \
    "${WORLD_DIRECTORY}" \
    "${MC_APPS}"; do
    if [[ ! -e "${path}" ]]; then
        echo "Required file or directory is missing: ${path}" >&2
        exit 1
    fi
done

# VisionFlow defaults to Pregme controllers. Keep the validated stock PX4
# controller chain for all GSI X500 airframes, including Yungu's 4012.
mc_apps_temp="$(mktemp)"
awk '
    function print_attitude_block() {
        print "# BEGIN GSI SearchWorld V1.1 attitude controllers (managed)"
        print "if param compare -s SYS_AUTOSTART 4010 || param compare -s SYS_AUTOSTART 4011 || param compare -s SYS_AUTOSTART 4012"
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
        print "if param compare -s SYS_AUTOSTART 4010 || param compare -s SYS_AUTOSTART 4011 || param compare -s SYS_AUTOSTART 4012"
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
' "${MC_APPS}" > "${mc_apps_temp}"

if ! grep -q 'SYS_AUTOSTART 4012' "${mc_apps_temp}" ||
   ! grep -q '^[[:space:]]*mc_pos_control start$' "${mc_apps_temp}"; then
    echo "Could not install the Yungu2030 PX4 controller selection." >&2
    rm -f "${mc_apps_temp}"
    exit 1
fi
install -m 0755 "${mc_apps_temp}" "${MC_APPS}"
rm -f "${mc_apps_temp}"

python3 - "${ASSET_DIR}/semantic_map.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    semantic_map = json.load(stream)
if semantic_map.get("world_name") != "yungu2030_local_origin":
    raise SystemExit("semantic map world_name does not match Yungu SDF world")
PY

# The original SDF deliberately uses a relative meshes/ URI. Keep the same
# directory layout beside the installed world file so Gazebo resolves its GLB.
install -m 0644 "${ASSET_DIR}/yungu_local_origin.sdf" "${WORLD_DIRECTORY}/${WORLD_NAME}.sdf"
install -d "${WORLD_DIRECTORY}/meshes"
install -m 0644 "${ASSET_DIR}/meshes/yungu_visual_local_origin.glb" \
    "${WORLD_DIRECTORY}/meshes/yungu_visual_local_origin.glb"
# PX4 launches Gazebo inside Docker. Preserve the source asset's relative
# layout on disk but use the container's mounted path in the installed copy.
# The editable asset package remains unchanged.
sed -i \
    "s#meshes/yungu_visual_local_origin.glb#file://${CONTAINER_VISIONFLOW_ROOT}/Tools/simulation/gz/worlds/meshes/yungu_visual_local_origin.glb#" \
    "${WORLD_DIRECTORY}/${WORLD_NAME}.sdf"
install -m 0644 "${ASSET_DIR}/semantic_map.json" \
    "${WORLD_DIRECTORY}/yungu2030_semantic_map.json"

install -d "${MODEL_DIRECTORY}"
install -m 0644 "${SCENARIO_DIR}/models/x500_gsi_rgbd_nadir_longrange/model.config" \
    "${MODEL_DIRECTORY}/model.config"
install -m 0644 "${SCENARIO_DIR}/models/x500_gsi_rgbd_nadir_longrange/model.sdf" \
    "${MODEL_DIRECTORY}/model.sdf"
install -m 0755 "${SCENARIO_DIR}/airframes/4012_gz_x500_gsi_rgbd_nadir_longrange" \
    "${AIRFRAME_DIRECTORY}/4012_gz_x500_gsi_rgbd_nadir_longrange"

if ! grep -q '4012_gz_x500_gsi_rgbd_nadir_longrange' "${AIRFRAME_CMAKE}"; then
    airframe_temp="$(mktemp)"
    awk '
        /^\)$/ && !inserted { print "\t4012_gz_x500_gsi_rgbd_nadir_longrange"; inserted=1 }
        { print }
    ' "${AIRFRAME_CMAKE}" > "${airframe_temp}"
    install -m 0644 "${airframe_temp}" "${AIRFRAME_CMAKE}"
    rm -f "${airframe_temp}"
fi

profile_temp="$(mktemp)"
awk '
    /^# BEGIN GSI Yungu2030 RGB-D PX4 profile \(managed\)\.$/ { managed=1; next }
    /^# END GSI Yungu2030 RGB-D PX4 profile \(managed\)\.$/ { managed=0; next }
    !managed { print }
' "${PROFILE_CONFIG}" > "${profile_temp}"
cat >> "${profile_temp}" <<'EOF'

# BEGIN GSI Yungu2030 RGB-D PX4 profile (managed).
add_sitl_profile \
    --id "GSI Yungu2030 RGB-D PX4" \
    --name "Yungu2030 CAD semantic-map search with long-range nadir RGB-D" \
    --world "yungu2030_local_origin" \
    --target "gz_x500_gsi_rgbd_nadir_longrange_yungu2030_local_origin" \
    --pose "42,90,0.25,0,0,0"
# END GSI Yungu2030 RGB-D PX4 profile (managed).
EOF
install -m 0644 "${profile_temp}" "${PROFILE_CONFIG}"
rm -f "${profile_temp}"

# PX4 discovers Gazebo worlds at CMake configure time.
touch "${GZ_CMAKE}" "${AIRFRAME_CMAKE}"
echo "Installed Yungu2030 world, semantics, long-range RGB-D model, and PX4 profile."
