#!/usr/bin/env bash
set -euo pipefail

TARGET_NAME="${1:-x500_gsi_rgbd_nadir_longrange_0}"
FOLLOW_X="${GSI_GUI_FOLLOW_X:--12}"
FOLLOW_Y="${GSI_GUI_FOLLOW_Y:--12}"
FOLLOW_Z="${GSI_GUI_FOLLOW_Z:-8}"

deadline=$((SECONDS + 60))
while (( SECONDS < deadline )); do
    if gz service -l 2>/dev/null | grep -qx /gui/follow; then
        break
    fi
    sleep 1
done
if (( SECONDS >= deadline )); then
    echo "Timed out waiting for the Gazebo camera tracking plugin." >&2
    exit 1
fi

TRACK_MESSAGE="track_mode: FOLLOW_LOOK_AT,
follow_target: {name: '${TARGET_NAME}'},
follow_offset: {x: ${FOLLOW_X}, y: ${FOLLOW_Y}, z: ${FOLLOW_Z}},
follow_pgain: 1.0,
track_pgain: 1.0"

gz topic -t /gui/track \
    -m gz.msgs.CameraTrack \
    -p "${TRACK_MESSAGE}"
