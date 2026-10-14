#!/usr/bin/env bash
set -euo pipefail
TT_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${TT_PROJECT_ROOT}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
exec "${TT_PROJECT_ROOT}/.venv/bin/python" -m orbvi_occ.run --config configs/lidar.json "$@"
