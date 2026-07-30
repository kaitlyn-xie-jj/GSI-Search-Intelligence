#!/usr/bin/env bash
set -eo pipefail

MODEL_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/simulation/yellow_search_van.sdf"

gz service \
    -s /world/laboratory_landingbox/create \
    --reqtype gz.msgs.EntityFactory \
    --reptype gz.msgs.Boolean \
    --timeout 5000 \
    --req "sdf_filename: '${MODEL_FILE}', name: 'yellow_search_van', allow_renaming: false"
