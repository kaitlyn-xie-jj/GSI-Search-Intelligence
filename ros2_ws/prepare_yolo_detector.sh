#!/usr/bin/env bash
set -euo pipefail

ROS2_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${GSI_YOLO_MODEL_DIR:-${ROS2_WS}/../artifacts/models}"
MODEL_NAME="${GSI_YOLO_MODEL_NAME:-yolo11n.pt}"

python3 -m pip install -r \
    "${ROS2_WS}/src/gsi_search_bridge/requirements-vision.txt"
mkdir -p "${MODEL_DIR}"
python3 - "${MODEL_DIR}" "${MODEL_NAME}" <<'PY'
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

destination = Path(sys.argv[1]).resolve() / sys.argv[2]
model = YOLO(sys.argv[2])
source = Path(model.ckpt_path).resolve()
if source != destination:
    shutil.copy2(source, destination)
print(destination)
PY
