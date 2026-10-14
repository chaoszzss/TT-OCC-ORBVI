#!/usr/bin/env bash
set -euo pipefail
TT_PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TT_PROJECT"
exec "$TT_PROJECT/.venv/bin/python" -m orbvi_occ.camera_semantic_run "$@"
