#!/usr/bin/env bash
set -euo pipefail
TT_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${TT_PROJECT_ROOT}"
exec "${TT_PROJECT_ROOT}/.venv/bin/python" scripts/live_viewer.py "$@"
