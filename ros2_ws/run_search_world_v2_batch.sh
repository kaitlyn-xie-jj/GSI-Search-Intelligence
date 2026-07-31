#!/usr/bin/env bash
set -euo pipefail

ROS2_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GSI_ROOT="$(cd "${ROS2_WS}/.." && pwd)"
VISIONFLOW_ROOT="${VISIONFLOW_ROOT:-/home/windylab/workspace/VisionFlow-PX4}"
CONTAINER="${GSI_SITL_CONTAINER:-visionflow-px4-sitl}"
REPETITIONS="${GSI_REPETITIONS:-3}"
REPETITION_START="${GSI_REPETITION_START:-1}"
TRIAL_TIMEOUT_S="${GSI_TRIAL_TIMEOUT_S:-600}"
STARTUP_TIMEOUT_S="${GSI_STARTUP_TIMEOUT_S:-240}"
HEADLESS="${GSI_HEADLESS:-1}"
BATCH_ID="$(date +%Y%m%d-%H%M%S)"
OUTPUT_ROOT="${GSI_V2_BATCH_DIR:-${GSI_V2_OUTPUT_ROOT:-${GSI_ROOT}/results/gazebo_search_world_v2}/${BATCH_ID}}"
read -r -a SCENARIOS <<< "${GSI_SCENARIOS:-campus industrial suburban}"

mkdir -p "${OUTPUT_ROOT}"
export PYTHONPATH="${ROS2_WS}/src/gsi_search_bridge:${PYTHONPATH:-}"
bash "${ROS2_WS}/install_search_world_v2.sh" "${VISIONFLOW_ROOT}"

sim_pid=""
search_pid=""
cleanup_trial() {
    if [[ -n "${search_pid}" ]]; then
        kill "${search_pid}" >/dev/null 2>&1 || true
        for _ in 1 2 3 4 5; do
            kill -0 "${search_pid}" >/dev/null 2>&1 || break
            sleep 1
        done
        kill -KILL "${search_pid}" >/dev/null 2>&1 || true
        wait "${search_pid}" >/dev/null 2>&1 || true
        search_pid=""
    fi
    docker exec "${CONTAINER}" bash -lc \
        "pkill -TERM -f 'ros2 launch gsi_search_bridge' || true" >/dev/null 2>&1 || true
    # Removing the container first unblocks the attached VisionFlow runner.
    docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
    if [[ -n "${sim_pid}" ]]; then
        kill -TERM "${sim_pid}" >/dev/null 2>&1 || true
        for _ in 1 2 3 4 5; do
            kill -0 "${sim_pid}" >/dev/null 2>&1 || break
            sleep 1
        done
        kill -KILL "${sim_pid}" >/dev/null 2>&1 || true
        wait "${sim_pid}" >/dev/null 2>&1 || true
        sim_pid=""
    fi
}
trap cleanup_trial EXIT INT TERM

for scenario in "${SCENARIOS[@]}"; do
    repetition_end=$((REPETITION_START + REPETITIONS - 1))
    for repetition in $(seq "${REPETITION_START}" "${repetition_end}"); do
        trial_id="${scenario}-r$(printf '%02d' "${repetition}")"
        trial_dir="${OUTPUT_ROOT}/${trial_id}"
        generated="${ROS2_WS}/simulation/search_world_v2/${scenario}/generated"
        trace_container="/tmp/GSI/results/gazebo_sensor_validation/gsi_search_world_v2_${scenario}_trace.jsonl"
        runtime_container="/tmp/GSI/results/gazebo_sensor_validation/gsi_search_world_v2_${scenario}_runtime.log"
        mkdir -p "${trial_dir}"
        cp "${ROS2_WS}/simulation/search_world_v2/${scenario}/scenario.json" "${trial_dir}/scenario.json"
        cp "${generated}/scenario_manifest.json" "${trial_dir}/scenario_manifest.json"
        cp "${generated}/ground_truth.json" "${trial_dir}/ground_truth.json"

        echo "[trial ${trial_id}] starting simulator"
        (
            VISIONFLOW_ROOT="${VISIONFLOW_ROOT}" HEADLESS="${HEADLESS}" \
                bash "${ROS2_WS}/start_search_world_v2_sitl.sh" "${scenario}" foreground \
                2>&1 | tail -c 20971520 > "${trial_dir}/simulator.log"
        ) &
        sim_pid="$!"

        ready=false
        deadline=$((SECONDS + STARTUP_TIMEOUT_S))
        while (( SECONDS < deadline )); do
            if docker inspect -f '{{.State.Running}}' "${CONTAINER}" 2>/dev/null | grep -qx true && \
               docker exec "${CONTAINER}" bash -lc \
                   "pgrep -x px4 >/dev/null && gz topic -l | grep -qx /gsi/rgbd/image"; then
                ready=true
                break
            fi
            sleep 5
        done
        if [[ "${ready}" != true ]]; then
            echo "[trial ${trial_id}] simulator startup timeout" >&2
            cleanup_trial
            continue
        fi

        echo "[trial ${trial_id}] synchronizing GSI runtime"
        docker exec "${CONTAINER}" bash -lc 'rm -rf /tmp/GSI && mkdir -p /tmp/GSI'
        docker cp "${GSI_ROOT}/modules" "${CONTAINER}:/tmp/GSI/modules" >/dev/null
        docker cp "${ROS2_WS}" "${CONTAINER}:/tmp/GSI/ros2_ws" >/dev/null
        docker exec "${CONTAINER}" bash -lc \
            "mkdir -p /tmp/GSI/results/gazebo_sensor_validation; rm -f '${trace_container}' '${runtime_container}'"

        echo "[trial ${trial_id}] running closed-loop search"
        docker exec "${CONTAINER}" bash -lc \
            "cd /tmp/GSI/ros2_ws && GSI_RUNTIME_LOG='${runtime_container}' bash run_search_world_v2_search.sh '${scenario}'" \
            > "${trial_dir}/search-console.log" 2>&1 &
        search_pid="$!"

        outcome=false
        deadline=$((SECONDS + TRIAL_TIMEOUT_S))
        while (( SECONDS < deadline )); do
            if docker exec "${CONTAINER}" bash -lc \
                "test -f '${trace_container}' && grep -q '\"event\": \"outcome\"' '${trace_container}'"; then
                outcome=true
                break
            fi
            if ! docker inspect -f '{{.State.Running}}' "${CONTAINER}" 2>/dev/null | grep -qx true; then
                break
            fi
            sleep 5
        done
        echo "[trial ${trial_id}] outcome=${outcome}; collecting artifacts"
        docker cp "${CONTAINER}:${trace_container}" "${trial_dir}/search_trace.jsonl" >/dev/null 2>&1 || true
        docker cp "${CONTAINER}:${runtime_container}" "${trial_dir}/runtime.log" >/dev/null 2>&1 || true
        docker logs --tail 100 "${CONTAINER}" 2>&1 \
            | tail -c 20971520 > "${trial_dir}/docker.log" || true
        if [[ ! -f "${trial_dir}/search_trace.jsonl" ]]; then
            touch "${trial_dir}/search_trace.jsonl"
        fi

        python3 -m gsi_search_bridge.v2_results \
            --scenario "${scenario}" \
            --trace "${trial_dir}/search_trace.jsonl" \
            --ground-truth "${trial_dir}/ground_truth.json" \
            --output "${trial_dir}/summary.json"
        cleanup_trial
    done
done

python3 -m gsi_search_bridge.v2_results --aggregate-root "${OUTPUT_ROOT}"
echo "SearchWorld V2 batch complete: ${OUTPUT_ROOT}"
