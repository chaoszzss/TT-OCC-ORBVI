import numpy as np
from orbvi_occ.camera_run import QualityCameraMotion,json_finite

def test_nonfinite_diagnostics_are_json_null():
    import json
    value=json_finite({'condition':np.inf,'nested':[np.nan,np.int64(3)]})
    assert json.loads(json.dumps(value,allow_nan=False))=={'condition':None,'nested':[None,3]}

def test_empty_predicted_geometry_resets_history_instead_of_fake_pose():
    motion=QualityCameraMotion()
    pose,info=motion.update(np.empty((0,3)))
    assert not info['accepted'] and info['history_reset']
    assert info['segment_id']==1
    np.testing.assert_array_equal(pose,np.eye(4))
