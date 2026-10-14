"""The evidence wrapper must preserve official panoptic masks and segment IDs."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip('torch')
from orbvi_occ.openseed_provider import _panoptic_with_evidence

PROJECT = Path(__file__).resolve().parents[1]


def canonical_function():
    source = PROJECT/'models/openseed-source/openseed/architectures/openseed_model.py'
    if not source.is_file():
        pytest.skip('Optional official OpenSeeD source not downloaded')
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='OpenSeeD')
    function = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='panoptic_inference')
    module = ast.Module(body=[function],type_ignores=[])
    namespace = {'torch':torch,'F':torch.nn.functional}
    exec(compile(ast.fix_missing_locations(module),str(source),'exec'),namespace)
    return namespace['panoptic_inference']


def test_recording_query_evidence_preserves_canonical_mask_without_inventing_scores():
    official = canonical_function()
    model = SimpleNamespace(sem_seg_head=SimpleNamespace(num_classes=11),object_mask_threshold=.25,
        transform_eval=True,pano_temp=.06,overlap_threshold=.8,
        metadata=SimpleNamespace(thing_dataset_id_to_contiguous_id={4:4,5:5}))
    model._canonical_panoptic_inference=lambda cls,masks:official(model,cls,masks)
    logits=torch.full((3,11),-6.)
    logits[0,4]=6.;logits[1,1]=6.;logits[2,1]=6.
    masks=torch.full((3,4,6),-6.)
    masks[0,:2,:]=6.;masks[1,2:,:3]=6.;masks[2,2:,3:]=6.
    original,original_segments=official(model,logits,masks)
    augmented,segments=_panoptic_with_evidence(model,logits,masks)
    torch.testing.assert_close(augmented,original)
    assert segments==original_segments
    recorded=model._last_panoptic_evidence
    assert set(torch.unique(recorded['query_ids']).tolist())=={0,1,2}
    assert recorded['evidence'].shape==(11,4,6)
    classes=recorded['evidence'].argmax(0)
    assert (classes[:2]==4).all() and (classes[2:]==1).all()
    expected=torch.softmax(logits.sigmoid()/.06,dim=-1).max(-1).values[0]*torch.sigmoid(torch.tensor(6.))
    torch.testing.assert_close(recorded['confidence'],torch.full((4,6),float(expected)),rtol=1e-6,atol=1e-6)
    assert recorded['confidence'].max()<1, 'Observed model scores must not become hard one'
    torch.testing.assert_close(recorded['evidence'].max(0).values,recorded['confidence'])


def test_rejected_panoptic_pixels_keep_zero_evidence_and_no_query_id():
    official=canonical_function()
    model=SimpleNamespace(sem_seg_head=SimpleNamespace(num_classes=11),object_mask_threshold=.25,
        transform_eval=True,pano_temp=.06,overlap_threshold=.8,
        metadata=SimpleNamespace(thing_dataset_id_to_contiguous_id={4:4}))
    model._canonical_panoptic_inference=lambda cls,masks:official(model,cls,masks)
    logits=torch.full((1,11),-8.);logits[0,4]=6.
    masks=torch.full((1,2,2),-6.)
    pano,segments=_panoptic_with_evidence(model,logits,masks)
    assert not pano.any() and segments==[]
    evidence=model._last_panoptic_evidence
    assert not evidence['evidence'].any() and not evidence['confidence'].any()
    assert (evidence['query_ids']==-1).all()
