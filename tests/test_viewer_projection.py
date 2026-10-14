"""Projection properties of the actual browser math, without a GPU or model."""
import json
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
import pytest


def test_official_parallel_projection_preserves_image_scale_across_depth():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node unavailable; real browser acceptance also verifies projection")
    page = (Path(__file__).resolve().parents[1]/"scripts/live_viewer.html").read_text()
    source = re.search(r'<script id="viewer-math">(.*?)</script>',page,re.S).group(1)
    check = """
const math=module.exports;
const points=[[3,5,-10,1],[3,5,-40,1]];
const parallel=math.orthographic(-10,10,-10,10,.1,1000);
const perspective=math.perspective(Math.PI/4,1,.1,1000);
const project=m=>points.map(p=>{const v=math.transform(m,p);return v.map(x=>x/v[3]);});
process.stdout.write(JSON.stringify({parallel:project(parallel),perspective:project(perspective)}));
"""
    result = subprocess.run([node],input=source+check,text=True,capture_output=True,check=True)
    rendered = json.loads(result.stdout)
    np.testing.assert_allclose(np.array(rendered['parallel'])[:,:2],[[.3,.5],[.3,.5]],rtol=0,atol=1e-7)
    first,second = np.array(rendered['perspective'])[:,:2]
    np.testing.assert_allclose(first,second*4,rtol=0,atol=1e-7)
