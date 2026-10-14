#!/usr/bin/env python3
"""Check or download pinned camera weights without changing the environment.

Run with the project's existing .venv. --check is local and never loads a model
onto a GPU. --download reuses complete valid weights and pinned sources.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import sys

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
B5_REPO='nvidia/segformer-b5-finetuned-ade-640-640'
B5_REVISION='739f5d4692954e4a185eac280dec1ba5a7d52f1d'
B5_DIRECTORY=PROJECT/'models/segformer-b5-ade20k'
EXPECTED_B5={
    'config.json':'083fbb6bc7ce4a2fc54bdb08686943cc5a72d14b32e81187b6ae12a58092bc13',
    'preprocessor_config.json':'2f6a855d3288caa2a34980aafe7353394fdd5582a5049cb4c9875581854597a8',
    'pytorch_model.bin':'a3dbf2d7a9912203078f304d88df8cfee9fd752ca2d2413f82036bdb02c5e7df',
    'model.safetensors':'4ff9bef4fc23da75ec1a6199f327a80963a1faa9a7ebc45ea73fc6c53c87cda3',
}
EXPECTED_MAP='fa06c0fdccefc5048e072c85935d5789b1e36b307f3859033c17f9dcb9fd5201'


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(8*1024**2),b''):
            digest.update(chunk)
    return digest.hexdigest()


def assert_digest(path,expected):
    if not Path(path).is_file():
        raise RuntimeError(f'Missing required model artifact: {path}')
    actual=sha256(path)
    if actual!=expected:
        raise RuntimeError(f'Pinned model SHA-256 mismatch: {path.name} expected {expected}, got {actual}')
    return {'bytes':Path(path).stat().st_size,'sha256':actual}


def canonical_safetensors(source,destination):
    """Canonicalize JSON header only; tensor payload is copied byte for byte.

    The Rust safetensors writer may randomize metadata key ordering. Sorting
    header JSON makes a converted file reproducible at the binary SHA level.
    Tensor-relative data offsets remain unchanged.
    """
    with Path(source).open('rb') as src,Path(destination).open('wb') as dst:
        length=struct.unpack('<Q',src.read(8))[0]
        header=json.loads(src.read(length))
        content=json.dumps(header,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
        content+=b' '*((-len(content))%8)
        dst.write(struct.pack('<Q',len(content)));dst.write(content)
        shutil.copyfileobj(src,dst,8*1024**2)


def download_b5():
    from huggingface_hub import snapshot_download
    B5_DIRECTORY.mkdir(parents=True,exist_ok=True)
    if any(not (B5_DIRECTORY/name).is_file() for name in EXPECTED_B5 if name!='model.safetensors'):
        snapshot_download(B5_REPO,revision=B5_REVISION,local_dir=B5_DIRECTORY,token=False,
            allow_patterns=['config.json','preprocessor_config.json','pytorch_model.bin','README.md'])
    for name,expected in EXPECTED_B5.items():
        if name!='model.safetensors':
            assert_digest(B5_DIRECTORY/name,expected)
    weights=B5_DIRECTORY/'model.safetensors'
    if not weights.exists() or sha256(weights)!=EXPECTED_B5['model.safetensors']:
        import torch
        from safetensors.torch import save_file
        state=torch.load(B5_DIRECTORY/'pytorch_model.bin',map_location='cpu',weights_only=True)
        if not isinstance(state,dict) or not all(torch.is_tensor(v) for v in state.values()):
            raise RuntimeError('Official semantic checkpoint must be a tensor state dictionary')
        temporary=weights.with_suffix('.safetensors.writer.tmp')
        canonical=weights.with_suffix('.safetensors.canonical.tmp')
        save_file({k:v.contiguous() for k,v in state.items()},str(temporary),metadata={
            'format':'pt','original_repository':B5_REPO,'original_revision':B5_REVISION,
            'conversion':'torch.load(weights_only=True), exact tensor copy'})
        canonical_safetensors(temporary,canonical)
        assert_digest(canonical,EXPECTED_B5['model.safetensors'])
        os.replace(canonical,weights);temporary.unlink()
    files={name:assert_digest(B5_DIRECTORY/name,expected) for name,expected in EXPECTED_B5.items()}
    readme=B5_DIRECTORY/'README.md'
    if readme.exists():
        files['README.md']={'bytes':readme.stat().st_size,'sha256':sha256(readme)}
    provenance={'repository':B5_REPO,'revision':B5_REVISION,'files':files,
        'conversion':'torch.load(weights_only=True), exact tensor copy, deterministic sorted JSON safetensors header; no training, no tensor changes'}
    (B5_DIRECTORY/'provenance.json').write_text(json.dumps(provenance,indent=2))


def check_b5():
    files={name:assert_digest(B5_DIRECTORY/name,expected) for name,expected in EXPECTED_B5.items()}
    manifest=json.loads((B5_DIRECTORY/'provenance.json').read_text())
    if manifest.get('repository')!=B5_REPO or manifest.get('revision')!=B5_REVISION:
        raise RuntimeError('B5 provenance does not identify the pinned official checkpoint')
    for name,item in files.items():
        if manifest.get('files',{}).get(name)!=item:
            raise RuntimeError(f'B5 provenance file entry mismatch: {name}')
    return {'repository':B5_REPO,'revision':B5_REVISION,'files':files}


def check_map_sources():
    from orbvi_occ.camera_depth import MAPANYTHING_SOURCE,DINOV2_SOURCE,MODEL_SOURCE_COMMIT,DINOV2_SOURCE_COMMIT
    out={}
    for name,path,commit in [('mapanything',MAPANYTHING_SOURCE,MODEL_SOURCE_COMMIT),('dinov2',DINOV2_SOURCE,DINOV2_SOURCE_COMMIT)]:
        provenance=path/'provenance.json'
        if not provenance.exists():
            raise RuntimeError(f'Missing pinned model architecture provenance: {path}')
        manifest=json.loads(provenance.read_text())
        if manifest.get('commit')!=commit:
            raise RuntimeError(f'Pinned architecture commit mismatch: {name}')
        for relative,item in manifest['files'].items():
            assert_digest(path/relative,item['sha256'])
        out[name]={'commit':commit,'source_files_checked':len(manifest['files'])}
    return out


def check_map():
    from orbvi_occ.camera_depth import DEFAULT_MODEL_DIR,MODEL_REPO,MODEL_REVISION
    manifest=json.loads((DEFAULT_MODEL_DIR/'provenance.json').read_text())
    if manifest.get('repository')!=MODEL_REPO or manifest.get('revision')!=MODEL_REVISION:
        raise RuntimeError('MapAnything checkpoint provenance differs from the pinned official revision')
    info=assert_digest(DEFAULT_MODEL_DIR/'model.safetensors',EXPECTED_MAP)
    if manifest['files']['model.safetensors']!=info:
        raise RuntimeError('MapAnything checkpoint provenance SHA mismatch')
    return {'repository':MODEL_REPO,'revision':MODEL_REVISION,'weights':info,'architectures':check_map_sources()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    choices=parser.add_mutually_exclusive_group(required=True)
    choices.add_argument('--check',action='store_true',help='Offline SHA/revision checks only; never use GPU')
    choices.add_argument('--download',action='store_true',help='Download missing pinned assets, safely convert B5 and verify all')
    args=parser.parse_args()
    os.environ.setdefault('HF_HUB_DISABLE_IMPLICIT_TOKEN','1')
    os.environ.setdefault('HF_HUB_DISABLE_XET','1')
    if args.download:
        from orbvi_occ.camera_depth import download_sources,download_model,DEFAULT_MODEL_DIR
        download_sources()
        # A complete valid 4.9 GB MapAnything file is never downloaded again.
        try:
            check_map()
        except (OSError,RuntimeError,KeyError,ValueError):
            download_model(DEFAULT_MODEL_DIR)
        download_b5()
    report={'complete':True,'mapanything':check_map(),'semantics_b5':check_b5(),
        'gpu_model_loaded':False,'environment_changed':False,'training_executed':False}
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
