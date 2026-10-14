import numpy as np
import pytest
from orbvi_occ.camera_fusion import CameraSemanticFusion,semantic_gaussian_voxelize,ray_inconsistent_history

BOUNDS=np.array([[-2,-2,-2],[2,2,2]],np.float32)


def evidence(labels,conf=1):
    out=np.zeros((len(labels),11),np.float32)
    for row,label in enumerate(labels):
        if label!=255:
            out[row,label]=conf
    return out


def test_unknown_geometry_does_not_outvote_known_semantics():
    points=np.repeat([[0,0,0]],101,axis=0)
    ev=evidence([4]+[255]*100)
    actual=semantic_gaussian_voxelize(points,ev,bounds=BOUNDS,sigma=.01,voxel_size=.3)
    assert actual['labels'].tolist()==[4]
    assert actual['geometry_support'][0]==101
    assert actual['support_count'][0]==1


def test_all_unknown_remains_occupied_without_known_semantics():
    actual=semantic_gaussian_voxelize(np.array([[0,0,0]]),evidence([255]),bounds=BOUNDS,sigma=.3)
    assert len(actual['indices'])>0
    assert np.all(actual['labels']==255)
    assert np.all(actual['known_support']==0)
    assert np.all(actual['confidence']==0)


def test_parked_car_contributes_past_history_and_current_evidence_has_priority():
    fusion=CameraSemanticFusion({'bounds':BOUNDS.tolist(),'gaussian_sigma':.01,
        'ray_consistency_enabled':False,'history_seconds':1,'input_voxel_size':0})
    pose=np.eye(4)
    fusion.update(np.repeat([[0,0,0]],30,axis=0),pose,1000000000,evidence([0]*30))
    car=fusion.update(np.array([[0,0,0]]),pose,1100000000,evidence([4]))
    assert car['fused_labels'].tolist()==[4]
    assert car['diagnostics']['current_semantic_priority_overrides']==1
    future=fusion.update(np.array([[1,0,0]]),pose,1200000000,evidence([1]))
    assert future['diagnostics']['past_points']>=31
    assert np.any(future['fused_evidence'][:,4]>0)


def test_entropy_reports_real_class_conflict_and_unknown_has_zero_evidence():
    actual=semantic_gaussian_voxelize(np.repeat([[0,0,0]],2,axis=0),evidence([0,4]),bounds=BOUNDS,sigma=.01)
    assert actual['confidence'][0]==.5
    assert actual['entropy'][0]>0
    assert actual['known_support'][0]==2


def test_near_old_surface_contradicted_by_current_ray_but_unseen_direction_retained():
    old=np.array([[2,0,0],[0,2,0],[8,0,0]])
    new=np.array([[5,0,0]])
    np.testing.assert_array_equal(ray_inconsistent_history(old,new),[True,False,False])


def test_camera_ray_origins_and_minimum_range_prevent_false_near_contradiction():
    old=np.array([[2,1,0],[.5,1,0]])
    new=np.array([[5,1,0],[5,1.01,0],[5,.99,0]])
    origins=np.tile([0,1,0],(3,1))
    actual=ray_inconsistent_history(old,new,ray_origins=origins,min_range=1,min_neighbors=3)
    np.testing.assert_array_equal(actual,[True,False])


def test_zero_semantic_confidence_cannot_invent_known_voxel():
    fusion=CameraSemanticFusion({'bounds':BOUNDS.tolist(),'history_seconds':0})
    out=fusion.update(np.array([[0,0,0]]),np.eye(4),1,evidence([4]),semantic_confidence=np.zeros(1))
    assert len(out['current_indices'])>0
    assert np.all(out['current_labels']==255)


def test_bounded_query_matches_original_unbounded_removal_in_covered_and_unseen_directions():
    from scipy.spatial import cKDTree
    rng=np.random.default_rng(12)
    directions=rng.normal(size=(2000,3));directions/=np.linalg.norm(directions,axis=1)[:,None]
    new=directions*rng.uniform(3,15,(2000,1))
    old=np.concatenate([new[:100]*.5,rng.normal(size=(100,3))*5,np.zeros((2,3))])
    angular_deg=5.;range_margin=.6;min_range=2.;neighbors=3
    nr=np.linalg.norm(new,axis=1);rr=np.linalg.norm(old,axis=1)
    chord,idx=cKDTree(new/nr[:,None]).query(old/np.maximum(rr[:,None],1e-6),k=neighbors)
    limit=2*np.sin(np.deg2rad(angular_deg)/2)
    reference=(rr>min_range)&np.all(chord<limit,axis=1)&(rr+range_margin<nr[idx].min(1))
    actual=ray_inconsistent_history(old,new,angular_tolerance_deg=angular_deg,
        range_margin=range_margin,min_range=min_range,min_neighbors=neighbors)
    np.testing.assert_array_equal(actual,reference)


def test_causality_and_segment_reset():
    fusion=CameraSemanticFusion({'bounds':BOUNDS.tolist()})
    fusion.update(np.array([[0,0,0]]),np.eye(4),100,evidence([4]))
    with pytest.raises(ValueError,match='increasing'):
        fusion.update(np.array([[0,0,0]]),np.eye(4),100,evidence([4]))
    fusion.reset()
    fresh=fusion.update(np.array([[0,0,0]]),np.eye(4),1,evidence([4]))
    assert fresh['diagnostics']['past_points']==0


@pytest.mark.parametrize('axis',[0,1,2])
def test_float32_last_physical_point_below_upper_stays_in_last_axis_cell(axis):
    bounds=np.array([[-12,-12,-2],[12,12,4]],np.float64)
    point=np.array([[0,0,0]],np.float32)
    point[0,axis]=np.nextafter(np.float32(bounds[1,axis]),np.float32(-np.inf))
    actual=semantic_gaussian_voxelize(point,evidence([4]),bounds=bounds,voxel_size=.3,sigma=.01)
    expected=np.floor((point.astype(np.float64)-bounds[0])/.3).astype(np.int32)
    np.testing.assert_array_equal(actual['indices'],expected)
    assert actual['indices'][0,axis]==np.array([79,79,19])[axis]
    assert actual['labels'].tolist()==[4]
    assert actual['geometry_support'].tolist()==[1]


@pytest.mark.parametrize('axis',[0,1,2])
def test_minimum_face_is_inclusive_and_exact_maximum_face_exclusive(axis):
    bounds=np.array([[-12,-12,-2],[12,12,4]],np.float64)
    low=np.array([[0,0,0]],np.float32);low[0,axis]=bounds[0,axis]
    high=low.copy();high[0,axis]=bounds[1,axis]
    below=low.copy();below[0,axis]=np.nextafter(np.float32(bounds[0,axis]),np.float32(-np.inf))
    inside=semantic_gaussian_voxelize(low,evidence([4]),bounds=bounds,voxel_size=.3,sigma=.01)
    assert len(inside['indices'])==1 and inside['indices'][0,axis]==0
    for point in (high,below):
        outside=semantic_gaussian_voxelize(point,evidence([4]),bounds=bounds,voxel_size=.3,sigma=.01)
        assert len(outside['indices'])==0


def test_last_yz_cells_do_not_alias_other_axes_when_flattened():
    bounds=np.array([[-12,-12,-2],[12,12,4]],np.float64)
    points=np.array([[0,np.nextafter(np.float32(12),np.float32(-np.inf)),0],
                     [0,0,np.nextafter(np.float32(4),np.float32(-np.inf))]],np.float32)
    actual=semantic_gaussian_voxelize(points,evidence([4,3]),bounds=bounds,voxel_size=.3,sigma=.01)
    expected=np.floor((points.astype(np.float64)-bounds[0])/.3).astype(np.int32)
    assert {tuple(x) for x in actual['indices']}=={tuple(x) for x in expected}
    for point,cls in zip(expected,[4,3]):
        row=np.flatnonzero(np.all(actual['indices']==point,axis=1))[0]
        assert actual['labels'][row]==cls
