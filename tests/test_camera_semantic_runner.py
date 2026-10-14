from pathlib import Path
import json
import numpy as np
import pytest
from orbvi_occ.camera_semantic_run import atomic_npz, read_npz, read_committed_journal


def test_lossless_npz_preserves_semantic_lineage_and_strides(tmp_path):
    arrays = {'xyz':np.arange(60,dtype=np.float32).reshape(10,6)[:,::2],
              'stamp_ns':np.int64(1720198192900208445),
              'view_ids':np.asarray(['3_v0','2_v0','4_v0','5_v0']),
              'valid_mask':np.asarray([[True,False],[False,True]])}
    path = tmp_path/'source.npz'
    atomic_npz(path,**arrays)
    actual = read_npz(path)
    assert all(np.asarray(value).dtype == actual[key].dtype and np.array_equal(value,actual[key])
               for key,value in arrays.items())
    assert not path.with_suffix('.npz.tmp').exists()


def test_resume_preserves_and_removes_only_uncommitted_tail(tmp_path):
    path = tmp_path/'frames.jsonl'
    committed = b'{"index":0,"stamp_ns":1720198192900208445}\n'
    interrupted = b'{"index":1,"stamp_ns":17201981'
    path.write_bytes(committed+interrupted)
    assert read_committed_journal(path) == [json.loads(committed)]
    assert path.read_bytes() == committed
    assert next(tmp_path.glob('uncommitted_journal_tail_*.bin')).read_bytes() == interrupted
    assert read_committed_journal(path) == [json.loads(committed)]


def test_resume_rejects_corrupted_committed_record(tmp_path):
    path = tmp_path/'frames.jsonl'
    path.write_bytes(b'{"index":broken}\n')
    with pytest.raises(json.JSONDecodeError):
        read_committed_journal(path)
