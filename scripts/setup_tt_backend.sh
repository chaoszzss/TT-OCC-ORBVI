#!/usr/bin/env bash
set -euo pipefail

TT_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TT_PYTHON="${TT_PYTHON:-${TT_PROJECT_ROOT}/.venv/bin/python}"
cd "${TT_PROJECT_ROOT}"
if [[ ! -x "${TT_PYTHON}" ]]; then
    echo "Create the isolated project .venv and install Torch using the README environment instructions first." >&2
    exit 1
fi
"${TT_PYTHON}" - <<'PY'
import sys
from pathlib import Path
root = Path.cwd().resolve()
if Path(sys.prefix).resolve() != root / '.venv':
    raise RuntimeError('Install dependencies only in the isolated TT-Occ/.venv')
import torch
print('Torch:', torch.__version__, 'CUDA available:', torch.cuda.is_available())
PY
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_DISABLE_XET=1
export HF_HOME="${TT_PROJECT_ROOT}/models/hf-cache"
"${TT_PYTHON}" -m pip install --disable-pip-version-check \
    'transformers==4.51.3' 'huggingface-hub==0.36.2' 'safetensors==0.8.0'
"${TT_PYTHON}" -m orbvi_occ.semantics --download

if [[ "${1:-}" == "--camera" ]]; then
    "${TT_PYTHON}" -m pip install --disable-pip-version-check --no-deps \
        'uniception==0.1.7' 'hydra-core==1.3.7' 'omegaconf==2.4.0' \
        'antlr4-python3-runtime==4.9.3' 'einops==0.8.2' 'timm==1.0.30' \
        'jaxtyping==0.3.11' 'natsort' 'pillow-heif' 'plyfile' 'trimesh==5.1.1' 'python-box==7.4.1'
    "${TT_PYTHON}" -m orbvi_occ.camera_depth --download
fi
echo 'Frozen pretrained backend setup complete.'
