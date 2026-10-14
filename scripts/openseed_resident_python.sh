#!/usr/bin/env bash
set -euo pipefail
TT_RESIDENCY_PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TT_RESIDENCY_PROJECT"
if [[ "${1-}" == "-m" && "${2-}" == "orbvi_occ.openseed_provider" ]]; then
    shift 2
    set -- -m orbvi_occ.openseed_residency "$@"
fi
exec "$TT_RESIDENCY_PROJECT/.openseed-venv/bin/python" "$@"
