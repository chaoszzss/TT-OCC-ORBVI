import hashlib
import json

import numpy as np
import pytest

from orbvi_occ.semantics import DEFAULT_MODEL_DIR, MODEL_REVISION, SegformerSemantics


def test_real_pretrained_semantics_are_pinned_and_frozen():
    weights = DEFAULT_MODEL_DIR / "model.safetensors"
    if not weights.exists():
        pytest.skip("Run setup_tt_backend.sh to download the real public checkpoint")
    metadata = json.loads((DEFAULT_MODEL_DIR / "provenance.json").read_text())
    assert metadata["revision"] == MODEL_REVISION
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    assert digest == metadata["files"]["model.safetensors"]["sha256"]
    provider = SegformerSemantics(device="cpu", image_size=128)
    assert not provider.model.training
    assert all(not p.requires_grad for p in provider.model.parameters())
    assert len(provider.id2label) == 150
    assert provider.id2label[0] == "wall"
    assert provider.dynamic_class_ids
    assert provider.metadata()["trained_on_orbvi"] is False
    with pytest.raises(ValueError, match="RGB uint8"):
        provider.predict(np.zeros((32, 32), np.uint8))
