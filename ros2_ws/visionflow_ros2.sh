#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/humble/setup.bash
source /workspace/VisionFlow-PX4/thirdparty/install/setup.bash

GSI_SETUP="${GSI_SETUP:-/tmp/GSI/ros2_ws/install/setup.bash}"
if [[ -f "${GSI_SETUP}" ]]; then
    source "${GSI_SETUP}"
fi

exec ros2 "$@"
