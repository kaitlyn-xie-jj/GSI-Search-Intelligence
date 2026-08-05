#!/usr/bin/env bash
set -euo pipefail

VISIONFLOW_ROOT="${VISIONFLOW_ROOT:-/home/windylab/workspace/VisionFlow-PX4}"
ACTION="${1:-foreground}"
CONTAINER_NAME="${GSI_SITL_CONTAINER:-visionflow-px4-sitl}"

case "${ACTION}" in
    foreground)
        cd "${VISIONFLOW_ROOT}"
        # The VisionFlow compose service allocates a TTY, which makes the PX4
        # shell repaint an empty prompt continuously when this trial runs in
        # the background. Keep real simulator output while dropping only that
        # terminal repaint sequence so long trials do not create huge logs.
        set -o pipefail
        bash docker/run_gz_sitl.sh --profile "GSI Yungu2030 RGB-D PX4" 2>&1 \
            | sed -u $'s/pxh> \033\\[2K\r//g'
        ;;
    status)
        docker inspect -f 'running={{.State.Running}} started={{.State.StartedAt}} oom={{.State.OOMKilled}}' \
            "${CONTAINER_NAME}"
        ;;
    stop)
        docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
        echo "Stopped ${CONTAINER_NAME}"
        ;;
    *)
        echo "Usage: $0 {foreground|status|stop}" >&2
        exit 2
        ;;
esac
