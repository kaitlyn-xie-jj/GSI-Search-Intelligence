#!/usr/bin/env bash
set -euo pipefail

VISIONFLOW_ROOT="${VISIONFLOW_ROOT:-/home/windylab/workspace/VisionFlow-PX4}"
SITL_PROFILE="${GSI_SITL_PROFILE:-GSI SearchWorld V1}"
CONTAINER_NAME="visionflow-px4-sitl"

run_foreground() {
    cd "${VISIONFLOW_ROOT}"
    exec bash docker/run_gz_sitl.sh --profile "${SITL_PROFILE}"
}

status() {
    if docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}" 2>/dev/null \
        | grep -qx true; then
        docker inspect -f 'running container={{.Name}} started={{.State.StartedAt}}' \
            "${CONTAINER_NAME}"
    else
        echo "stopped container=${CONTAINER_NAME}"
        exit 1
    fi
}

stop() {
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    echo "Stopped ${CONTAINER_NAME}"
}

case "${1:-foreground}" in
    foreground) run_foreground ;;
    status) status ;;
    stop) stop ;;
    *)
        echo "Usage: $0 {foreground|status|stop}" >&2
        echo "Keep the foreground terminal open while PX4/Gazebo is running." >&2
        exit 2
        ;;
esac
