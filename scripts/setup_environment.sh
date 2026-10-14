#!/usr/bin/env bash
set -euo pipefail

TT_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TT_RUNTIME_PYTHON="${TT_PROJECT_ROOT}/.venv/bin/python"
TT_CHECK_ONLY=0
TT_INSTALL_EXISTING=0
TT_BASE_PYTHON="${TT_BASE_PYTHON:-}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --check) TT_CHECK_ONLY=1; shift ;;
        --install) TT_INSTALL_EXISTING=1; shift ;;
        --python) TT_BASE_PYTHON="$2"; shift 2 ;;
        -h|--help)
            echo 'Usage: scripts/setup_environment.sh [--check] [--install] [--python /path/to/python3.12]'
            echo 'Existing valid .venv: verification only. Missing .venv: clean installation. --install explicitly updates existing project dependencies.'
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
cd "${TT_PROJECT_ROOT}"

if [[ ! -x "${TT_RUNTIME_PYTHON}" ]]; then
    if [[ "${TT_CHECK_ONLY}" == 1 ]]; then
        echo 'Missing project .venv. Run setup_environment.sh with a Python 3.12 executable.' >&2
        exit 1
    fi
    if [[ -e "${TT_PROJECT_ROOT}/.venv" ]]; then
        echo 'Existing .venv is incomplete; preserve or move it before creating a new environment.' >&2
        exit 1
    fi
    if [[ -z "${TT_BASE_PYTHON}" ]]; then
        TT_BASE_PYTHON="$(command -v python3.12 || true)"
    fi
    if [[ -z "${TT_BASE_PYTHON}" && -x /home/chaoszzss/APP/miniconda3/envs/x-anylabeling-cu12/bin/python ]]; then
        TT_BASE_PYTHON=/home/chaoszzss/APP/miniconda3/envs/x-anylabeling-cu12/bin/python
    fi
    if [[ -z "${TT_BASE_PYTHON}" ]]; then
        echo 'Python 3.12 is required. Supply --python /path/to/python3.12.' >&2
        exit 1
    fi
    "${TT_BASE_PYTHON}" -c 'import sys; assert sys.version_info[:2] == (3,12), "Python 3.12 required"'
    # No --system-site-packages: new installations use clean project dependencies.
    "${TT_BASE_PYTHON}" -m venv "${TT_PROJECT_ROOT}/.venv"
    TT_INSTALL_EXISTING=1
fi

"${TT_RUNTIME_PYTHON}" - <<'PY'
from pathlib import Path
import sys
assert sys.version_info[:2] == (3,12), 'Project runtime must use Python 3.12'
assert Path(sys.prefix).resolve() == Path.cwd() / '.venv', 'Refusing to install into a user/base environment'
PY

if [[ "${TT_INSTALL_EXISTING}" == 1 && "${TT_CHECK_ONLY}" == 0 ]]; then
    "${TT_RUNTIME_PYTHON}" -m pip install --require-virtualenv --disable-pip-version-check 'pip==25.0.1' 'setuptools==83.0.0'
    "${TT_RUNTIME_PYTHON}" -m pip install --require-virtualenv --disable-pip-version-check \
        --index-url https://download.pytorch.org/whl/cu126 \
        'torch==2.7.1' 'torchvision==0.22.1'
    "${TT_RUNTIME_PYTHON}" -m pip install --require-virtualenv --disable-pip-version-check \
        -r "${TT_PROJECT_ROOT}/requirements-runtime.txt"
    "${TT_RUNTIME_PYTHON}" -m pip install --require-virtualenv --disable-pip-version-check \
        --no-deps -e "${TT_PROJECT_ROOT}"
fi

"${TT_RUNTIME_PYTHON}" - <<'PY'
import sys
import numpy, scipy, cv2, PIL, psutil, matplotlib, numba, requests, yaml
import torch, torchvision, transformers, huggingface_hub, safetensors, tokenizers
assert torch.__version__.split('+')[0] == '2.7.1', torch.__version__
assert torchvision.__version__.split('+')[0] == '0.22.1', torchvision.__version__
assert torch.version.cuda == '12.6', torch.version.cuda
assert transformers.__version__ == '4.51.3', transformers.__version__
assert huggingface_hub.__version__ == '0.36.2', huggingface_hub.__version__
from orbvi_occ.data import load_frames
from orbvi_occ.calibration import deskew_scan
print('Runtime:', sys.executable)
print('Torch:', torch.__version__, 'Torchvision:', torchvision.__version__)
print('CUDA available:', torch.cuda.is_available(), '(CPU fallback remains supported)')
print('NumPy:', numpy.__version__, 'SciPy:', scipy.__version__, 'OpenCV:', cv2.__version__)
print('Runtime verification passed. Model download: scripts/setup_tt_backend.sh [--camera]')
PY
