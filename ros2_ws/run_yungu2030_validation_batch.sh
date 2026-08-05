#!/usr/bin/env bash
set -uo pipefail

ROS2_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GSI_ROOT="$(cd "${ROS2_WS}/.." && pwd)"
MODE="${1:-all}"
MANIFEST_SOURCE="${GSI_YUNGU_VALIDATION_MANIFEST:-${ROS2_WS}/simulation/yungu2030_v1/validation_manifest.json}"
SUMMARY_TOOL="${ROS2_WS}/tools/summarize_yungu2030_validation.py"
TRIAL_RUNNER="${GSI_YUNGU_TRIAL_RUNNER:-${ROS2_WS}/run_yungu2030_sensor_trial.sh}"
TARGET_FILTER="${GSI_VALIDATION_TARGET_IDS:-}"

case "${MODE}" in
    preflight|stability|positions|all) ;;
    *)
        echo "Usage: $0 {preflight|stability|positions|all} [batch-root]" >&2
        exit 2
        ;;
esac

DEFAULT_BATCH_ROOT="${GSI_ROOT}/results/yungu2030_validation_batches/$(date -u +%Y%m%dT%H%M%SZ)_${MODE}"
BATCH_ROOT="${2:-${GSI_VALIDATION_BATCH_ROOT:-${DEFAULT_BATCH_ROOT}}}"
EPISODE_ROOT="${BATCH_ROOT}/episodes"
FROZEN_MANIFEST="${BATCH_ROOT}/validation_manifest.json"
mkdir -p "${EPISODE_ROOT}"

python3 "${SUMMARY_TOOL}" validate-manifest --manifest "${MANIFEST_SOURCE}" >/dev/null || exit 2
SOURCE_SHA="$(sha256sum "${MANIFEST_SOURCE}" | awk '{print $1}')"
if [[ -f "${FROZEN_MANIFEST}" ]]; then
    FROZEN_SHA="$(sha256sum "${FROZEN_MANIFEST}" | awk '{print $1}')"
    if [[ "${SOURCE_SHA}" != "${FROZEN_SHA}" ]]; then
        echo "Refusing to resume with a different validation manifest." >&2
        exit 2
    fi
else
    cp "${MANIFEST_SOURCE}" "${FROZEN_MANIFEST}"
fi
printf '%s  %s\n' "${SOURCE_SHA}" "validation_manifest.json" > "${BATCH_ROOT}/validation_manifest.sha256"

EPISODE_ARGUMENTS=(episodes --manifest "${FROZEN_MANIFEST}" --mode "${MODE}")
if [[ -n "${TARGET_FILTER}" ]]; then
    IFS=',' read -ra TARGET_IDS <<< "${TARGET_FILTER}"
    for target_id in "${TARGET_IDS[@]}"; do
        EPISODE_ARGUMENTS+=(--target-id "${target_id}")
    done
fi
mapfile -t EPISODES < <(python3 "${SUMMARY_TOOL}" "${EPISODE_ARGUMENTS[@]}")
EXPECTED_COUNT="${#EPISODES[@]}"
if (( EXPECTED_COUNT == 0 )); then
    echo "No validation episodes matched the requested mode and target filter." >&2
    exit 2
fi
python3 - "${BATCH_ROOT}/batch_metadata.json" "${MODE}" "${EXPECTED_COUNT}" "${SOURCE_SHA}" "${TARGET_FILTER}" <<'PY'
import datetime as dt
import json
import os
from pathlib import Path
import sys

path = Path(sys.argv[1])
mode = sys.argv[2]
expected = int(sys.argv[3])
sha256 = sys.argv[4]
target_filter = [value for value in sys.argv[5].split(",") if value]
metadata = {}
if path.is_file():
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("batch_mode") != mode:
        raise SystemExit("batch root already belongs to a different mode")
    if metadata.get("target_filter", []) != target_filter:
        raise SystemExit("batch root already belongs to a different target filter")
metadata.update({
    "schema_version": "gsi-yungu2030-validation-batch-metadata-v1",
    "batch_mode": mode,
    "expected_episode_count": expected,
    "manifest_sha256": sha256,
    "target_filter": target_filter,
    "started_utc": metadata.get("started_utc") or dt.datetime.now(dt.timezone.utc).isoformat(),
})
temporary = path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY

touch "${BATCH_ROOT}/batch.log"
# Keep long validation batches independent from the launching terminal. A
# closed stdout pipe must not send SIGPIPE into a later trial.
exec >> "${BATCH_ROOT}/batch.log" 2>&1
echo "Yungu2030 validation batch: mode=${MODE} root=${BATCH_ROOT} episodes=${EXPECTED_COUNT}"

for row in "${EPISODES[@]}"; do
    IFS=$'\t' read -r EPISODE_ID COHORT TARGET_ID SEMANTIC_REGION REPETITION TARGET_X TARGET_Y TARGET_Z TARGET_YAW <<< "${row}"
    TRIAL_DIR="${EPISODE_ROOT}/${EPISODE_ID}"
    if [[ -s "${TRIAL_DIR}/trial_summary.json" ]]; then
        echo "SKIP ${EPISODE_ID}: trial_summary.json already exists"
        continue
    fi
    mkdir -p "${TRIAL_DIR}"
    STARTED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    python3 "${SUMMARY_TOOL}" metadata \
        --trial-dir "${TRIAL_DIR}" \
        --manifest "${FROZEN_MANIFEST}" \
        --episode-id "${EPISODE_ID}" \
        --batch-mode "${MODE}" \
        --cohort "${COHORT}" \
        --target-id "${TARGET_ID}" \
        --semantic-region "${SEMANTIC_REGION}" \
        --repetition "${REPETITION}" \
        --x "${TARGET_X}" --y "${TARGET_Y}" --z "${TARGET_Z}" --yaw "${TARGET_YAW}" \
        --started-utc "${STARTED_UTC}" \
        --runner-status 1

    echo "START ${EPISODE_ID} target=(${TARGET_X},${TARGET_Y},${TARGET_Z})"
    GSI_YUNGU_RESULTS_ROOT="${TRIAL_DIR}" \
    GSI_TARGET_X_M="${TARGET_X}" \
    GSI_TARGET_Y_M="${TARGET_Y}" \
    GSI_TARGET_Z_M="${TARGET_Z}" \
    GSI_TARGET_YAW_RAD="${TARGET_YAW}" \
    GSI_SEARCH_TIMEOUT_S="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["configuration"]["search_timeout_s"])' "${FROZEN_MANIFEST}")" \
    GSI_CAPTURE_DURATION_S="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["configuration"]["capture_duration_s"])' "${FROZEN_MANIFEST}")" \
        bash "${TRIAL_RUNNER}"
    RUNNER_STATUS=$?
    ENDED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    python3 "${SUMMARY_TOOL}" metadata \
        --trial-dir "${TRIAL_DIR}" \
        --manifest "${FROZEN_MANIFEST}" \
        --episode-id "${EPISODE_ID}" \
        --batch-mode "${MODE}" \
        --cohort "${COHORT}" \
        --target-id "${TARGET_ID}" \
        --semantic-region "${SEMANTIC_REGION}" \
        --repetition "${REPETITION}" \
        --x "${TARGET_X}" --y "${TARGET_Y}" --z "${TARGET_Z}" --yaw "${TARGET_YAW}" \
        --started-utc "${STARTED_UTC}" \
        --ended-utc "${ENDED_UTC}" \
        --runner-status "${RUNNER_STATUS}"

    if python3 "${SUMMARY_TOOL}" trial --trial-dir "${TRIAL_DIR}" >/dev/null; then
        echo "END ${EPISODE_ID} runner_status=${RUNNER_STATUS}"
    else
        echo "SUMMARY_ERROR ${EPISODE_ID} runner_status=${RUNNER_STATUS}" >&2
    fi
    python3 "${SUMMARY_TOOL}" batch --batch-root "${BATCH_ROOT}" >/dev/null || \
        echo "BATCH_SUMMARY_ERROR after ${EPISODE_ID}" >&2
done

python3 "${SUMMARY_TOOL}" batch --batch-root "${BATCH_ROOT}" >/dev/null || exit 2
python3 - "${BATCH_ROOT}/batch_metadata.json" <<'PY'
import datetime as dt
import json
import os
from pathlib import Path
import sys

path = Path(sys.argv[1])
metadata = json.loads(path.read_text(encoding="utf-8"))
metadata["ended_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
temporary = path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY
echo "Yungu2030 validation batch complete: ${BATCH_ROOT}"
