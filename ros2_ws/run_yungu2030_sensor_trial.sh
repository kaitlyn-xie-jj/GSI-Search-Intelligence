#!/usr/bin/env bash
set -euo pipefail

ROS2_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GSI_ROOT="$(cd "${ROS2_WS}/.." && pwd)"
if [[ -z "${VISIONFLOW_ROOT:-}" ]]; then
    for candidate in \
        "${GSI_ROOT}/../../projects/VisionFlow-PX4" \
        "${GSI_ROOT}/../VisionFlow-PX4" \
        /home/windylab/workspace/VisionFlow-PX4; do
        if [[ -f "${candidate}/docker/gz_sitl_profiles.conf" ]]; then
            VISIONFLOW_ROOT="$(cd "${candidate}" && pwd)"
            break
        fi
    done
fi
if [[ -z "${VISIONFLOW_ROOT:-}" ]]; then
    echo "VisionFlow-PX4 was not found. Set VISIONFLOW_ROOT to its checkout path." >&2
    exit 1
fi
CONTAINER_NAME="${GSI_SITL_CONTAINER:-visionflow-px4-sitl}"
OUTPUT_ROOT="${GSI_YUNGU_RESULTS_ROOT:-${GSI_ROOT}/results/yungu2030_sensor_validation/$(date -u +%Y%m%dT%H%M%SZ)}"
TARGET_X_M="${GSI_TARGET_X_M:-175}"
TARGET_Y_M="${GSI_TARGET_Y_M:-85}"
TARGET_Z_M="${GSI_TARGET_Z_M:-0.4}"
TARGET_YAW_RAD="${GSI_TARGET_YAW_RAD:-0}"
CAPTURE_DURATION_S="${GSI_CAPTURE_DURATION_S:-600}"
STARTUP_TIMEOUT_S="${GSI_STARTUP_TIMEOUT_S:-300}"
SEARCH_TIMEOUT_S="${GSI_SEARCH_TIMEOUT_S:-600}"
SEARCH_TIME_BUDGET_S="${GSI_SEARCH_TIME_BUDGET_S:-$(python3 -c 'import sys; print(max(1.0, float(sys.argv[1]) - 5.0))' "${SEARCH_TIMEOUT_S}")}"
GUI_CONFIG="${GSI_YUNGU_GUI_CONFIG:-}"
PRE_SEARCH_DELAY_S="${GSI_PRE_SEARCH_DELAY_S:-0}"
YOLO_MODEL_PATH="${GSI_YOLO_MODEL_PATH:-${GSI_ROOT}/artifacts/models/yolo11n.pt}"
INSTALL_YOLO_DEPS="${GSI_INSTALL_YOLO_DEPS:-1}"
SHOW_CAMERA="${GSI_SHOW_CAMERA:-0}"

if [[ ! -s "${YOLO_MODEL_PATH}" ]]; then
    echo "YOLO model is missing: ${YOLO_MODEL_PATH}" >&2
    echo "Prepare it with: bash ros2_ws/prepare_yolo_detector.sh" >&2
    exit 1
fi

if [[ -d "${OUTPUT_ROOT}" ]] && find "${OUTPUT_ROOT}" -mindepth 1 -print -quit | grep -q .; then
    OUTPUT_ROOT="${OUTPUT_ROOT}-$(date -u +%Y%m%dT%H%M%SZ)"
    echo "Requested result directory already contains artifacts; using ${OUTPUT_ROOT}"
fi
mkdir -p "${OUTPUT_ROOT}"

cleanup() {
    if [[ -n "${CAMERA_VIEW_PID:-}" ]]; then
        kill -- "-${CAMERA_VIEW_PID}" >/dev/null 2>&1 || true
        wait "${CAMERA_VIEW_PID}" >/dev/null 2>&1 || true
    fi
    if [[ -n "${CAPTURE_PID:-}" ]]; then
        kill "${CAPTURE_PID}" >/dev/null 2>&1 || true
        wait "${CAPTURE_PID}" >/dev/null 2>&1 || true
    fi
    if [[ -n "${SIM_PID:-}" ]]; then
        bash "${ROS2_WS}/start_yungu2030_sitl.sh" stop >/dev/null 2>&1 || true
        wait "${SIM_PID}" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT INT TERM

bash "${ROS2_WS}/install_yungu2030_sitl.sh" "${VISIONFLOW_ROOT}"
cp "${ROS2_WS}/simulation/yungu2030_v1/public_scenario.json" "${OUTPUT_ROOT}/public_scenario.json"
printf '{\n  "entity_name": "yellow_search_van",\n  "pose_enu_m_rad": {"x": %s, "y": %s, "z": %s, "yaw": %s}\n}\n' \
    "${TARGET_X_M}" "${TARGET_Y_M}" "${TARGET_Z_M}" "${TARGET_YAW_RAD}" \
    > "${OUTPUT_ROOT}/evaluator_target_pose.json"

(
    SIM_HEADLESS="${HEADLESS:-1}"
    if [[ "${HEADLESS:-1}" == "0" && -n "${GUI_CONFIG}" ]]; then
        SIM_HEADLESS=1
    fi
    VISIONFLOW_ROOT="${VISIONFLOW_ROOT}" HEADLESS="${SIM_HEADLESS}" \
        bash "${ROS2_WS}/start_yungu2030_sitl.sh" foreground
) > "${OUTPUT_ROOT}/simulator.log" 2>&1 &
SIM_PID=$!

deadline=$((SECONDS + STARTUP_TIMEOUT_S))
while (( SECONDS < deadline )); do
    if docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}" 2>/dev/null | grep -qx true && \
       docker exec "${CONTAINER_NAME}" bash -lc \
           'pgrep -x px4 >/dev/null && gz topic -l | grep -qx /gsi/rgbd/image'; then
        break
    fi
    sleep 5
done
if (( SECONDS >= deadline )); then
    echo "Timed out waiting for PX4 and the RGB-D Gazebo topic." >&2
    exit 1
fi

# PX4 starts the Gazebo server itself. For watched validation runs, attach a
# GUI client to that server after it is ready; the container restart cleans up
# the client together with the rest of the trial.
if [[ "${HEADLESS:-1}" == "0" ]]; then
    if [[ -n "${GUI_CONFIG}" ]]; then
        if [[ ! -f "${GUI_CONFIG}" ]]; then
            echo "Gazebo GUI config not found: ${GUI_CONFIG}" >&2
            exit 1
        fi
        docker cp "${GUI_CONFIG}" "${CONTAINER_NAME}:/tmp/gsi_yungu_gui.config" >/dev/null
        docker cp "${ROS2_WS}/set_yungu_gui_follow.sh" \
            "${CONTAINER_NAME}:/tmp/set_yungu_gui_follow.sh" >/dev/null
        docker exec -d "${CONTAINER_NAME}" gz sim -g \
            --gui-config /tmp/gsi_yungu_gui.config
        docker exec "${CONTAINER_NAME}" bash /tmp/set_yungu_gui_follow.sh
    elif ! docker exec "${CONTAINER_NAME}" pgrep -f 'gz sim -g' >/dev/null 2>&1; then
        docker exec -d "${CONTAINER_NAME}" gz sim -g
    fi
fi

docker exec "${CONTAINER_NAME}" bash -lc 'rm -rf /tmp/GSI && mkdir -p /tmp/GSI/results/yungu2030_sensor_validation'
docker cp "${GSI_ROOT}/modules" "${CONTAINER_NAME}:/tmp/GSI/modules"
docker cp "${ROS2_WS}" "${CONTAINER_NAME}:/tmp/GSI/ros2_ws"
docker exec "${CONTAINER_NAME}" mkdir -p /tmp/GSI/models
docker cp "${YOLO_MODEL_PATH}" "${CONTAINER_NAME}:/tmp/GSI/models/yolo11n.pt"
if [[ "${INSTALL_YOLO_DEPS}" == "1" ]]; then
    # Always reconcile the complete pinned stack. Checking only that
    # Ultralytics imports can miss incompatible transitive build tools.
    docker exec "${CONTAINER_NAME}" python3 -m pip install \
        -r /tmp/GSI/ros2_ws/src/gsi_search_bridge/requirements-vision.txt
else
    if ! docker exec "${CONTAINER_NAME}" python3 -c \
        'import cv2, numpy, packaging, setuptools, ultralytics' >/dev/null 2>&1; then
        echo "Ultralytics is not installed in ${CONTAINER_NAME}." >&2
        echo "Set GSI_INSTALL_YOLO_DEPS=1 or bake requirements-vision.txt into the image." >&2
        exit 1
    fi
fi
docker exec "${CONTAINER_NAME}" bash -lc \
    "cd /tmp/GSI/ros2_ws && bash spawn_visionflow_target.sh yungu2030_local_origin '${TARGET_X_M}' '${TARGET_Y_M}' '${TARGET_Z_M}' '${TARGET_YAW_RAD}' yellow_search_van"

docker exec "${CONTAINER_NAME}" bash -lc \
    "cd /tmp/GSI/ros2_ws && GSI_CAPTURE_DURATION_S='${CAPTURE_DURATION_S}' GSI_ALLOW_RAW_RGB_FALLBACK=1 bash record_yungu2030_sensor_data.sh /tmp/GSI/results/yungu2030_sensor_validation/capture" \
    > "${OUTPUT_ROOT}/capture_console.log" 2>&1 &
CAPTURE_PID=$!

if (( PRE_SEARCH_DELAY_S > 0 )); then
    echo "Waiting ${PRE_SEARCH_DELAY_S}s before starting search for GUI recording setup."
    sleep "${PRE_SEARCH_DELAY_S}"
fi

set +e
setsid timeout --foreground "${SEARCH_TIMEOUT_S}" \
    docker exec "${CONTAINER_NAME}" bash -lc \
        "cd /tmp/GSI/ros2_ws && GSI_SEARCH_TIME_BUDGET_S='${SEARCH_TIME_BUDGET_S}' GSI_RUNTIME_LOG=/tmp/GSI/results/yungu2030_sensor_validation/runtime.log bash run_yungu2030_search.sh" \
    > >(tee "${OUTPUT_ROOT}/search_console.log") 2>&1 &
SEARCH_PID=$!
if [[ "${SHOW_CAMERA}" == "1" ]]; then
    setsid bash "${ROS2_WS}/view_yungu2030_camera.sh" "${CONTAINER_NAME}" &
    CAMERA_VIEW_PID=$!
fi
SEARCH_STATUS=""
while kill -0 "${SEARCH_PID}" >/dev/null 2>&1; do
    if docker exec "${CONTAINER_NAME}" bash -lc \
        "grep -q '\"event\": \"outcome\"' /tmp/GSI/results/yungu2030_sensor_validation/search_trace.jsonl 2>/dev/null"; then
        SEARCH_STATUS=0
        sleep 1
        kill -- "-${SEARCH_PID}" >/dev/null 2>&1 || true
        wait "${SEARCH_PID}" >/dev/null 2>&1 || true
        break
    fi
    sleep 2
done
if [[ -z "${SEARCH_STATUS}" ]]; then
    wait "${SEARCH_PID}"
    SEARCH_STATUS=$?
fi
set -e
printf '%s\n' "${SEARCH_STATUS}" > "${OUTPUT_ROOT}/search_exit_status.txt"

# Stop the recorder before collecting artifacts. This lets ffmpeg write the
# MP4 trailer and rosbag close its metadata before docker cp observes them.
if [[ -n "${CAPTURE_PID:-}" ]]; then
    kill "${CAPTURE_PID}" >/dev/null 2>&1 || true
    wait "${CAPTURE_PID}" >/dev/null 2>&1 || true
    CAPTURE_PID=""
fi
docker cp "${CONTAINER_NAME}:/tmp/GSI/results/yungu2030_sensor_validation/." "${OUTPUT_ROOT}/" >/dev/null 2>&1 || true
# Preserve PX4's native flight log before container cleanup. MAVROS may not
# decode development-version PX4 event IDs, while the ULog retains the exact
# estimator, attitude, actuator, and failure-detector state.
PX4_ULOG_PATH="$(docker exec "${CONTAINER_NAME}" bash -lc \
    "find /workspace/VisionFlow-PX4/build/px4_sitl_default/rootfs/log -type f -name '*.ulg' -printf '%T@ %p\\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-" \
    2>/dev/null || true)"
if [[ -n "${PX4_ULOG_PATH}" ]]; then
    mkdir -p "${OUTPUT_ROOT}/px4"
    docker cp "${CONTAINER_NAME}:${PX4_ULOG_PATH}" "${OUTPUT_ROOT}/px4/flight.ulg" \
        >/dev/null 2>&1 || true
fi
docker logs --tail 200 "${CONTAINER_NAME}" > "${OUTPUT_ROOT}/docker.log" 2>&1 || true

if [[ ! -s "${OUTPUT_ROOT}/capture/rgb.mp4" && ! -s "${OUTPUT_ROOT}/capture/rgb.rgb24" ]]; then
    echo "No RGB video artifact was produced; inspect ${OUTPUT_ROOT}/capture/video_recorder.log" >&2
    exit 1
fi
if [[ ! -s "${OUTPUT_ROOT}/search_trace.jsonl" ]]; then
    echo "Search trace was not produced; inspect ${OUTPUT_ROOT}/runtime.log" >&2
    exit 1
fi
echo "Yungu2030 sensor trial artifacts: ${OUTPUT_ROOT}"
