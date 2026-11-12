#!/usr/bin/env bash

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# CANN's set_env.sh is not compatible with nounset/errexit while sourcing.
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -euo pipefail

cmake -S "${PROJECT_DIR}" -B "${PROJECT_DIR}/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "${PROJECT_DIR}/build" -j4
