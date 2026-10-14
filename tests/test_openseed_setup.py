"""CPU-only provisioning checks: parent package reuse and read-only audits."""
import importlib.util
import json
from pathlib import Path
import subprocess
import shutil
import sys

PROJECT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('openseed_setup_audit',PROJECT/'scripts/setup_openseed.py')
setup=importlib.util.module_from_spec(spec);spec.loader.exec_module(setup)


def run_json(python,code):
    return json.loads(subprocess.check_output([str(python),'-I','-B','-c',code],text=True,env=setup._read_only_env()))


def test_distinct_empty_child_reuses_parent_packages_but_keeps_child_priority(tmp_path):
    parent=tmp_path/'parent';child=tmp_path/'child'
    try:
        subprocess.run([sys.executable,'-m','venv','--without-pip',str(parent)],check=True,env=setup._read_only_env())
        parent_python=parent/'bin/python'
        subprocess.run([str(parent_python),'-m','venv','--without-pip','--system-site-packages',str(child)],check=True,env=setup._read_only_env())
        child_python=child/'bin/python'
        parent_site=Path(setup._python_info(parent_python)['sites'][0])
        child_site=Path(setup._python_info(child_python)['sites'][0])
        parent_site.mkdir(parents=True,exist_ok=True);child_site.mkdir(parents=True,exist_ok=True)
        (parent_site/'ttocc_parent_only_probe.py').write_text('origin="parent package"\n')
        (parent_site/'ttocc_priority_probe.py').write_text('origin="parent version"\n')
        (child_site/'ttocc_priority_probe.py').write_text('origin="child version"\n')
        before=run_json(child_python,'import importlib.util,json;print(json.dumps({"parent_visible":importlib.util.find_spec("ttocc_parent_only_probe") is not None}))')
        assert not before['parent_visible'], '--system-site-packages must not be mistaken for parent venv inheritance'
        bridge=setup.configure_parent_runtime_pth(parent_python,child_python)
        after=run_json(child_python,'import json,sys,ttocc_parent_only_probe,ttocc_priority_probe;print(json.dumps({"parent":ttocc_parent_only_probe.origin,"priority":ttocc_priority_probe.origin,"path":sys.path}))')
        assert after['parent']=='parent package' and after['priority']=='child version'
        assert after['path'].index(str(child_site))<after['path'].index(str(parent_site))
        assert Path(bridge['path']).is_relative_to(child)
        assert not (parent_site/'ttocc_parent_runtime.pth').exists()
        assert not list(parent_site.glob('__pycache__/ttocc_*'))
    finally:
        shutil.rmtree(parent,ignore_errors=True)
        shutil.rmtree(child,ignore_errors=True)

def test_readonly_audit_of_missing_project_never_creates_or_repairs_files(tmp_path):
    nonexistent=tmp_path/'not-created'
    report=setup.check_installation(nonexistent)
    assert report['status']=='not_ready' and report['read_only']
    assert not report['neural_inference_performed'] and not report['cuda_tensor_created']
    assert not nonexistent.exists()
    assert not report['dependency_audit']


def test_child_bridge_refuses_to_write_into_parent_or_base_interpreter(tmp_path):
    import pytest
    with pytest.raises(ValueError,match='distinct child'):
        setup.configure_parent_runtime_pth(sys.executable,sys.executable)
