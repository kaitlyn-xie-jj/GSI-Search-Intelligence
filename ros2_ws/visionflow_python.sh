#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/humble/setup.bash
source /workspace/VisionFlow-PX4/thirdparty/install/setup.bash
exec python3 "$@"
