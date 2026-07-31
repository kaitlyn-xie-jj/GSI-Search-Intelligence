#!/usr/bin/env bash
set -euo pipefail

VISIONFLOW_ROOT="${VISIONFLOW_ROOT:-/home/windylab/workspace/VisionFlow-PX4}"
SCENARIO="${1:-campus}"
ACTION="${2:-foreground}"
CONTAINER_NAME="${GSI_SITL_CONTAINER:-visionflow-px4-sitl}"

case "${SCENARIO}" in
    campus) profile="GSI SearchWorld V2 Campus" ;;
    industrial) profile="GSI SearchWorld V2 Industrial" ;;
    suburban) profile="GSI SearchWorld V2 Suburban" ;;
    *) echo "Usage: $0 {campus|industrial|suburban} {foreground|status|stop}" >&2; exit 2 ;;
esac

case "${ACTION}" in
    foreground)
        cd "${VISIONFLOW_ROOT}"
        exec bash docker/run_gz_sitl.sh --profile "${profile}"
        ;;
    status)
        docker inspect -f 'running={{.State.Running}} started={{.State.StartedAt}} oom={{.State.OOMKilled}}' \
            "${CONTAINER_NAME}"
        ;;
    stop)
        docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
        echo "Stopped ${CONTAINER_NAME}"
        ;;
    *) echo "Usage: $0 {campus|industrial|suburban} {foreground|status|stop}" >&2; exit 2 ;;
esac
