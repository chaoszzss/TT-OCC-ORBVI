#!/usr/bin/env python3
"""Provision the optional official OpenSeeD worker without changing main runtime."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time

PROJECT = Path(__file__).resolve().parents[1]
SOURCES = [
    ('MaureenZOU/detectron2-xyz','42121d75e10d9f858f3a91b6a39f5722c02868f0','detectron2-openseed-source'),
    ('cocodataset/panopticapi','7bb4655548f98f3fedc07bf37e9040a992b054b0','panopticapi-source'),
    ('IDEA-Research/OpenSeeD','5dc694ed55e4b71ff57ee3d98ef6ac0dfcaad847','openseed-source'),
]
WEIGHT_SHA256 = '9b6a20c81cf8206845da161cfaf0490563ae79c35e7797058d68913a5f4a61a2'
TOKENIZER_REVISION = '3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'

# Resolve transitive dependencies normally, while retaining model ABI and old
# Transformers/tokenizer isolation. iopath/omegaconf satisfy Detectron2/Hydra.
INSTALL_PACKAGES = [
    "transformers==4.45.2", "tokenizers==0.20.3", "huggingface-hub==0.36.2",
    "timm==0.6.13", "fvcore==0.1.5.post20221221", "iopath==0.1.9",
    "yacs==0.1.8", "pycocotools==2.0.10", "omegaconf==2.3.0",
    "cloudpickle==3.1.2", "einops==0.8.2", "nltk==3.10.3", "diffdist==0.1",
    "kornia==0.8.3", "torchmetrics==1.9.0", "tensorboard==2.21.0",
    "hydra-core==1.3.7", "antlr4-python3-runtime==4.9.3",
    "ftfy==6.3.1", "ipdb==0.13.13", "regex==2026.9.29", "safetensors==0.8.0",
    "setuptools==83.0.0", "wheel==0.47.0",
]
CORE_VERSION_CONSTRAINTS = {
    "torch": "2.7.1+cu126", "torchvision": "0.22.1+cu126",
    "transformers": "4.45.2", "tokenizers": "0.20.3", "timm": "0.6.13",
}
LICENSES = {
    "openseed-source": ("LICENSE", "Apache-2.0"),
    "detectron2-openseed-source": ("LICENSE", "Apache-2.0"),
    "panopticapi-source": ("license.txt", "BSD-2-Clause"),
}


def _read_only_env():
    return dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")


def _python_info(python):
    code = "import json,sys,sysconfig;print(json.dumps({'prefix':sys.prefix,'base_prefix':sys.base_prefix,'version':list(sys.version_info[:2]),'sites':list(dict.fromkeys([sysconfig.get_path('purelib'),sysconfig.get_path('platlib')]))}))"
    output = subprocess.check_output([str(python), "-I", "-B", "-c", code], text=True, env=_read_only_env())
    return json.loads(output)


def configure_parent_runtime_pth(parent_python, child_python):
    """Add the parent package directory after child packages, never before them.

    venv --system-site-packages inherits BASE sites, not its calling venv's sites.
    The plain .pth file is installed only in the child and contains no executable
    code. In particular, child Transformers/tokenizers override main versions.
    """
    parent, child = _python_info(parent_python), _python_info(child_python)
    child_prefix = Path(child["prefix"]).resolve()
    if child["prefix"] == child["base_prefix"] or child_prefix == Path(parent["prefix"]).resolve():
        raise ValueError("Parent runtime bridge can be installed only in a distinct child virtual environment")
    if parent["version"] != child["version"]:
        raise ValueError("Parent and isolated child Python major/minor versions differ")
    child_site = Path(child["sites"][0]).resolve()
    if not child_site.is_relative_to(child_prefix):
        raise ValueError("Child site directory is outside its own virtual environment")
    parent_sites = [str(Path(value).resolve()) for value in parent["sites"] if Path(value).is_dir()]
    if not parent_sites or any("\n" in value or "\r" in value for value in parent_sites):
        raise ValueError("Invalid parent site-packages directories")
    child_site.mkdir(parents=True,exist_ok=True)
    path = child_site / "ttocc_parent_runtime.pth"
    value = "# TT-Occ: child packages take priority; reuse parent frozen Torch runtime.\n" + "\n".join(parent_sites) + "\n"
    path.write_text(value)
    return {"path": str(path), "parent_sites": parent_sites, "child_site": str(child_site)}


def _probe_cpu_environment(python, source):
    """Imports only: no network, model construction, checkpoint load or CUDA tensors."""
    code = r'''
import sys,json,importlib,importlib.metadata
sys.dont_write_bytecode=True
from PIL import Image
if not hasattr(Image,'LINEAR'):Image.LINEAR=Image.BILINEAR
source=sys.argv[1]
sys.path.insert(0,source)
report={'imports':{},'versions':{},'sys_path':sys.path,'no_neural_inference':True,'no_cuda_tensor_created':True}
for name,package in [('torch','torch'),('torchvision','torchvision'),('transformers','transformers'),('tokenizers','tokenizers'),('timm','timm'),('detectron2','detectron2'),('detectron2._C',None),('MultiScaleDeformableAttention','MultiScaleDeformableAttention'),('panopticapi',None),('openseed',None),('ipdb','ipdb')]:
 try:
  module=importlib.import_module(name)
  report['imports'][name]={'ok':True,'file':getattr(module,'__file__',None)}
  if package:report['versions'][package]=importlib.metadata.version(package)
 except Exception as exc:report['imports'][name]={'ok':False,'error':type(exc).__name__+': '+str(exc)}
for name,symbol in [('detectron2._C','get_compiler_version'),('MultiScaleDeformableAttention','ms_deform_attn_forward')]:
 module=sys.modules.get(name)
 report['imports'][name]['compiled_binary_symbol_available']=bool(module is not None and callable(getattr(module,symbol,None)))
torch_module=sys.modules.get('torch')
report['cuda_initialized']=bool(torch_module is not None and torch_module.cuda.is_initialized())
report['no_cuda_tensor_created']=not report['cuda_initialized']
print(json.dumps(report))
'''
    result = subprocess.run([str(python), "-I", "-B", "-c", code, str(source)],
                            text=True,capture_output=True,env=_read_only_env(),timeout=60)
    if result.returncode:
        return {"ok":False,"error":result.stderr[-5000:],"returncode":result.returncode}
    try:
        return json.loads(result.stdout)
    except ValueError:
        return {"ok":False,"error":"CPU import probe did not return valid JSON","stdout":result.stdout[-5000:]}


def check_installation(project=PROJECT):
    """Read-only audit; findings never trigger download, installation or repair."""
    project = Path(project)
    checks = []
    def record(name,passed,details=None):
        checks.append({"name":name,"passed":bool(passed),"details":details})
    for repository,commit,name in SOURCES:
        target = project / "models" / name
        provenance_path = target / "source-provenance.json"
        try:
            provenance = json.loads(provenance_path.read_text())
            record(name+"_pinned_source",provenance.get("repository")==repository and provenance.get("commit")==commit,provenance)
        except (OSError,ValueError) as exc:
            record(name+"_pinned_source",False,str(exc))
        filename,license_id = LICENSES[name]
        license_path = target/filename
        record(name+"_license",license_path.is_file() and license_path.stat().st_size>100,{"path":str(license_path),"license":license_id})
    weight = project/"models/openseed-swint/model_state_dict_swint_51.2ap.pt"
    record("official_checkpoint",weight.is_file() and weight.stat().st_size==465260032 and sha256(weight)==WEIGHT_SHA256,
           {"path":str(weight),"expected_sha256":WEIGHT_SHA256,"expected_bytes":465260032})
    tokenizer = project/"models/clip-tokenizer"
    try:
        provenance=json.loads((tokenizer/"provenance.json").read_text())
        record("pinned_tokenizer",provenance.get("repository")=="openai/clip-vit-base-patch32" and provenance.get("revision")==TOKENIZER_REVISION,provenance)
    except (OSError,ValueError) as exc:
        record("pinned_tokenizer",False,str(exc))
    names=("vocab.json","merges.txt","tokenizer_config.json","special_tokens_map.json","tokenizer.json","config.json")
    record("tokenizer_files",all((tokenizer/name).is_file() for name in names),list(names))
    cuda_source=project/"models/openseed-source/openseed/body/encoder/ops/src/cuda/ms_deform_attn_cuda.cu"
    original=cuda_source.with_suffix(".cu.original")
    try:
        expected=original.read_text().replace("AT_DISPATCH_FLOATING_TYPES(value.type(),","AT_DISPATCH_FLOATING_TYPES(value.scalar_type(),")
        record("cuda_dispatch_only_compatibility_patch",expected==cuda_source.read_text(),{"source":str(cuda_source),"original":str(original),"math_changed":False})
    except OSError as exc:
        record("cuda_dispatch_only_compatibility_patch",False,str(exc))
    python=project/".openseed-venv/bin/python"
    record("isolated_python",python.is_file(),str(python))
    cpu = _probe_cpu_environment(python,project/"models/openseed-source") if python.is_file() else {}
    record("cpu_probe_did_not_initialize_cuda",cpu.get("no_cuda_tensor_created") is True,{"cuda_initialized":cpu.get("cuda_initialized")})
    for name in ("torch","torchvision","transformers","tokenizers","timm","detectron2","detectron2._C","MultiScaleDeformableAttention","panopticapi","openseed","ipdb"):
        entry=cpu.get("imports",{}).get(name,{})
        record("cpu_import_"+name,entry.get("ok") is True,entry)
    for name in ("detectron2._C","MultiScaleDeformableAttention"):
        entry=cpu.get("imports",{}).get(name,{})
        record("compiled_symbol_"+name,entry.get("compiled_binary_symbol_available") is True,entry)
    for name,version in CORE_VERSION_CONSTRAINTS.items():
        record("version_"+name,cpu.get("versions",{}).get(name)==version,{"expected":version,"actual":cpu.get("versions",{}).get(name)})
    dependency_audit = {}
    if python.is_file():
        result=subprocess.run([str(python),"-I","-B","-m","pip","check"],text=True,capture_output=True,env=_read_only_env(),timeout=30)
        dependency_audit={"all_metadata_requirements_satisfied":result.returncode==0,"details":(result.stdout+result.stderr).strip(),
                          "scope":"Includes optional package metadata and inherited base packages. Never repaired by --check."}
    parent_site = None
    parent_python=project/".venv/bin/python"
    if python.is_file() and parent_python.is_file():
        parent_info, child_info = _python_info(parent_python), _python_info(python)
        path=Path(child_info["sites"][0])/"ttocc_parent_runtime.pth"
        parent_site={"bridge_exists":path.is_file(),"bridge_path":str(path),"parent_sites":parent_info["sites"],
                     "active_parent_sites":[value for value in parent_info["sites"] if value in cpu.get("sys_path",[])],
                     "note":"Existing environment can use base Torch without the bridge; fresh setup installs the bridge in child only."}
    return {"status":"ready_for_inference" if all(c["passed"] for c in checks) else "not_ready",
            "read_only":True,"neural_inference_performed":False,"cuda_tensor_created":False,
            "checks":checks,"cpu_import_probe":cpu,"dependency_audit":dependency_audit,"parent_runtime_bridge":parent_site,
            "note":"Readiness checks binaries/imports/assets; actual full-model strict loading is documented by the previous real inference run. Dependency metadata warnings remain distinct."}




def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--download-only',action='store_true')
    modes.add_argument('--check',action='store_true',help='Fully read-only asset, CPU import and compiled-extension audit; no downloads or NN')
    args=parser.parse_args()
    if args.check:
        report=check_installation()
        print(json.dumps(report,indent=2,ensure_ascii=False))
        raise SystemExit(0 if report["status"]=="ready_for_inference" else 1)
    import requests
    session=requests.Session()
    for repository,commit,name in SOURCES:
        target=PROJECT/'models'/name
        if target.exists():
            provenance=json.loads((target/'source-provenance.json').read_text())
            if provenance['commit']!=commit:
                raise ValueError(f'{name} belongs to a different source version')
            continue
        response=session.get(f'https://api.github.com/repos/{repository}/tarball/{commit}',timeout=180);response.raise_for_status()
        target.mkdir(parents=True)
        with tarfile.open(fileobj=io.BytesIO(response.content),mode='r:gz') as archive:
            for member in archive.getmembers():
                parts=member.name.split('/',1)
                if len(parts)<2 or not parts[1]:
                    continue
                member.name=parts[1];archive.extract(member,target,filter='data')
        (target/'source-provenance.json').write_text(json.dumps({'repository':repository,'commit':commit,'url':response.url},indent=2)+'\n')
    weights=PROJECT/'models/openseed-swint/model_state_dict_swint_51.2ap.pt'
    weights.parent.mkdir(exist_ok=True)
    if not weights.is_file():
        url='https://api.github.com/repos/IDEA-Research/OpenSeeD/releases/assets/105743964'
        response=session.get(url,params={'cache_bust':str(time.time_ns())},headers={'Accept':'application/octet-stream'},stream=True,timeout=180)
        response.raise_for_status()
        partial=weights.with_suffix('.pt.partial')
        with partial.open('wb') as output:
            for block in response.iter_content(8*1024*1024):output.write(block)
        if sha256(partial)!=WEIGHT_SHA256:
            raise ValueError('Downloaded checkpoint hash differs from independently verified official release')
        partial.replace(weights)
    if sha256(weights)!=WEIGHT_SHA256:
        raise ValueError('Existing OpenSeeD checkpoint hash is invalid')
    (weights.parent/'provenance.json').write_text(json.dumps({'repository':'IDEA-Research/OpenSeeD',
        'revision':'release-openseed-asset-105743964','release_asset_id':105743964,
        'release_asset':'https://github.com/IDEA-Research/OpenSeeD/releases/download/openseed/model_state_dict_swint_51.2ap.pt',
        'sha256':WEIGHT_SHA256,'bytes':weights.stat().st_size,'strict_checkpoint_keys':793,'trained_on_orbvi':False},indent=2)+'\n')
    tokenizer=PROJECT/'models/clip-tokenizer';tokenizer.mkdir(exist_ok=True)
    provenance_path=tokenizer/'provenance.json'
    revision=TOKENIZER_REVISION
    if provenance_path.is_file() and json.loads(provenance_path.read_text())['revision']!=revision:
        raise ValueError('Tokenizer cache belongs to a different pinned revision')
    for name in ('vocab.json','merges.txt','tokenizer_config.json','special_tokens_map.json','tokenizer.json','config.json'):
        if (tokenizer/name).is_file():continue
        response=session.get(f'https://huggingface.co/openai/clip-vit-base-patch32/resolve/{revision}/{name}',timeout=90);response.raise_for_status()
        (tokenizer/name).write_bytes(response.content)
    provenance_path.write_text(json.dumps({'repository':'openai/clip-vit-base-patch32','revision':revision,
        'scope':'tokenizer only; learned CLIP text weights are in official OpenSeeD checkpoint'},indent=2)+'\n')
    cuda_source=PROJECT/'models/openseed-source/openseed/body/encoder/ops/src/cuda/ms_deform_attn_cuda.cu'
    code=cuda_source.read_text()
    old='AT_DISPATCH_FLOATING_TYPES(value.type(),'
    if old in code:
        cuda_source.with_suffix('.cu.original').write_text(code)
        cuda_source.write_text(code.replace(old,'AT_DISPATCH_FLOATING_TYPES(value.scalar_type(),'))
    (PROJECT/'models/openseed-source/compatibility-patches.json').write_text(json.dumps({
        'patches':[{'file':str(cuda_source.relative_to(PROJECT/'models/openseed-source')),'before':old,
            'after':'AT_DISPATCH_FLOATING_TYPES(value.scalar_type(),','purpose':'Torch2 scalar dispatch API compatibility; mathematical operation unchanged'}],
        'upstream_directory_modified':False},indent=2)+'\n')
    if args.download_only:
        print('Official source copies, tokenizer and verified checkpoint are ready.');return
    python=PROJECT/'.openseed-venv/bin/python'
    if not python.is_file():
        subprocess.run([str(PROJECT/'.venv/bin/python'),'-m','venv','--system-site-packages',str(PROJECT/'.openseed-venv')],check=True)
    configure_parent_runtime_pth(PROJECT/'.venv/bin/python',python)
    constraints_path=PROJECT/'.openseed-venv/constraints.txt'
    constraints={**CORE_VERSION_CONSTRAINTS, "iopath":"0.1.9", "omegaconf":"2.3.0"}
    constraints_path.write_text("\n".join(f"{name}=={version}" for name,version in constraints.items())+"\n")
    parent_versions=json.loads(subprocess.check_output([str(PROJECT/'.venv/bin/python'),'-I','-B','-c',
        "import importlib.metadata,json;print(json.dumps({name:importlib.metadata.version(name) for name in ['torch','torchvision']}))"],text=True,env=_read_only_env()))
    for package in ('torch','torchvision'):
        if parent_versions.get(package)!=CORE_VERSION_CONSTRAINTS[package]:
            raise RuntimeError(f"Main runtime {package} must already be {CORE_VERSION_CONSTRAINTS[package]}; initialize the main project runtime first. Isolated setup will not replace it.")
    # Normal resolution includes IPython/antlr/kornia-rs/lightning-utilities and
    # Detectron2's declared tools, rather than leaving --no-deps gaps.
    subprocess.run([str(python),'-m','pip','install','--constraint',str(constraints_path),*INSTALL_PACKAGES],check=True)
    env=dict(os.environ,CUDA_HOME='/usr/local/cuda-12.8',TORCH_CUDA_ARCH_LIST='8.9',MAX_JOBS='2')
    for path in [PROJECT/'models/panopticapi-source',PROJECT/'models/detectron2-openseed-source',
                 PROJECT/'models/openseed-source/openseed/body/encoder/ops']:
        subprocess.run([str(python),'-m','pip','install','--constraint',str(constraints_path),'--no-build-isolation',str(path)],check=True,env=env)
    versions=subprocess.check_output([str(python),'-m','pip','list','--format=json'],text=True)
    (weights.parent/'environment-packages.json').write_text(versions)
    print('Isolated OpenSeeD environment ready; main .venv and upstream sources preserved.')


if __name__=='__main__':main()
