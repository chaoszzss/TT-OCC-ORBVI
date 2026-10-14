import numpy as np
import pytest

from orbvi_occ.traversability import (BLOCKED, OBSERVED_CLEARANCE, PARTIAL_CLEARANCE,
                                     UNKNOWN, LocalTraversability)


def floor_points(slope=0.0):
    x, y = np.meshgrid(np.arange(-4.5, 4.6, 0.25), np.arange(-4.5, 4.6, 0.25), indexing="ij")
    return np.c_[x.ravel(), y.ravel(), (-1.0 + slope * x).ravel()]


def mapper(**kwargs):
    config = {"resolution": 0.25, "bounds": [-6, -6, -2, 6, 6, 2],
              "footprint_length": 0.4, "footprint_width": 0.4,
              "safety_margin": 0.1, "vehicle_height": 0.8,
              "ground_tolerance": 0.06, "min_ground_inliers": 20,
              "ray_max_points": 2500, "gaussian_sigma": 0.3}
    config.update(kwargs)
    return LocalTraversability(config)


def cell(m, xy):
    return tuple(np.floor((np.array(xy) - m.origin[:2]) / m.config.resolution).astype(int))


def test_unobserved_never_becomes_free():
    m = mapper()
    out = m.update(np.empty((0, 3)), np.empty((0, 3)))
    assert np.all(out.traversability == UNKNOWN)
    assert not out.observed_free.any()
    assert not out.summary["ground"]["valid"]


def test_nearest_return_stops_ray_before_farther_return():
    m = mapper()
    free = m.ray_free_mask(np.array([[1, 0, 0], [3, 0, 0]], dtype=float))
    near = m.voxel_indices(np.array([[0.4, 0, 0]]))[0]
    behind = m.voxel_indices(np.array([[2, 0, 0]]))[0]
    endpoint = m.voxel_indices(np.array([[1, 0, 0]]))[0]
    assert free[tuple(near)]
    assert not free[tuple(behind)]
    assert not free[tuple(endpoint)]


def test_slope_floor_wall_and_footprint_inflation():
    m = mapper()
    floor = floor_points(slope=0.06)
    y, z = np.meshgrid(np.arange(-2, 2.1, 0.25), np.arange(-0.6, 0.05, 0.25), indexing="ij")
    wall = np.c_[np.full(y.size, 2.0), y.ravel(), z.ravel()]
    points = np.r_[floor, wall]
    out = m.update(points, wall)
    assert out.summary["ground"]["valid"]
    assert abs(out.summary["ground"]["slope_degrees"] - np.rad2deg(np.arctan(0.06))) < 0.3
    assert out.traversability[cell(m, [2, 0])] == BLOCKED
    assert out.traversability[cell(m, [1.75, 0])] == BLOCKED
    assert out.traversability[cell(m, [5.5, 5.5])] == UNKNOWN
    assert not np.any(out.observed_free & out.collision3d)


def test_history_motion_compensation_and_expiry():
    m = mapper(free_history_seconds=0.3)
    pose0 = np.eye(4)
    m.update(np.array([[3, 0, 0]]), np.empty((0, 3)), pose0, timestamp=0)
    pose1 = np.eye(4)
    pose1[0, 3] = 1.0
    moved = m.update(np.empty((0, 3)), np.empty((0, 3)), pose1, timestamp=0.1)
    idx = m.voxel_indices(np.array([[0.4, 0, 0]]))[0]
    assert moved.observed_free[tuple(idx)]
    expired = m.update(np.empty((0, 3)), np.empty((0, 3)), pose1, timestamp=0.5)
    assert not expired.observed_free.any()


def test_floor_gaussian_provenance_preserves_low_obstacle():
    from orbvi_occ.tt_backend import gaussian_voxelize
    m = mapper()
    floor = floor_points()
    # A 20 cm obstacle is outside the 6 cm floor residual band. Its Gaussian
    # overlaps floor Gaussians and must survive floor-source removal.
    low_obstacle = np.array([[1.0, 0.0, -0.8]])
    source = np.r_[floor, low_obstacle]
    idx, _ = gaussian_voxelize(source, bounds=np.array([m.config.bounds[:3], m.config.bounds[3:]]), voxel_size=0.25, sigma=0.3)
    centres = m.origin + (idx + 0.5) * 0.25
    out = m.update(source, centres, fused_source_points=source)
    assert out.summary["floor_source_returns"] == len(floor)
    assert out.occupied3d.sum() > out.collision3d.sum()
    assert out.traversability[cell(m, [1, 0])] == BLOCKED
    assert out.traversability[cell(m, [-2, -2])] != BLOCKED


def test_partial_clearance_not_promoted_to_strict_with_unknown_slice():
    m = mapper(footprint_length=0.2, footprint_width=0.2, safety_margin=0)
    def controlled_free(points, origin=None):
        free = np.ones(m.shape, bool)
        # One whole unknown height layer in the requested body volume.
        z_unknown = int(np.floor((-0.5 - m.origin[2]) / m.config.resolution))
        free[:, :, z_unknown] = False
        return free
    m.ray_free_mask = controlled_free
    out = m.update(floor_points(), np.empty((0, 3)))
    assert out.traversability[cell(m, [0, 0])] == PARTIAL_CLEARANCE
    assert not np.any(out.traversability == OBSERVED_CLEARANCE)
    assert out.summary["partial_is_verified_free"] is False


def test_clearance_above_grid_is_unknown_not_confirmed():
    m = mapper(vehicle_height=4.0)
    m.ray_free_mask = lambda points, origin=None: np.ones(m.shape, bool)
    out = m.update(floor_points(), np.empty((0, 3)))
    assert not np.any(out.traversability == OBSERVED_CLEARANCE)


def test_without_pose_history_is_not_blindly_reused():
    m = mapper()
    m.update(np.array([[3, 0, 0]]), np.empty((0, 3)), timestamp=0)
    out = m.update(np.empty((0, 3)), np.empty((0, 3)), timestamp=0.1)
    assert not out.observed_free.any()


def test_individual_ray_origins_do_not_assume_lidar_origin():
    m = mapper()
    points = np.array([[3, 2, 0], [3, -2, 0]], dtype=float)
    origins = np.array([[0, 2, 0], [0, -2, 0]], dtype=float)
    free = m.ray_free_mask(points, origins)
    on_ray = m.voxel_indices(np.array([[1, 2, 0]]))[0]
    between_rays = m.voxel_indices(np.array([[1, 0, 0]]))[0]
    assert free[tuple(on_ray)]
    assert not free[tuple(between_rays)]


def test_camera_predicted_depth_cannot_certify_full_clearance():
    m = mapper(footprint_length=0.2, footprint_width=0.2, safety_margin=0)
    m.ray_free_mask = lambda points, origin=None: np.ones(m.shape, bool)
    out = m.update(floor_points(), np.empty((0, 3)), free_evidence_source="predicted_camera_depth")
    assert out.traversability[cell(m, [0, 0])] == PARTIAL_CLEARANCE
    assert not np.any(out.traversability == OBSERVED_CLEARANCE)
    assert out.summary["free_evidence_is_sensor_confirmed"] is False


def test_default_collision_voxelizer_remains_the_unmodified_legacy_function(monkeypatch):
    from orbvi_occ import tt_backend
    legacy=tt_backend.gaussian_voxelize
    called=[]
    def traced(points,**kwargs):
        called.append(points.copy())
        return legacy(points,**kwargs)
    m=mapper();source=np.r_[floor_points(),[[1,0,-.8]]]
    idx,_=legacy(source,bounds=[m.config.bounds[:3],m.config.bounds[3:]],voxel_size=.25,sigma=.3)
    centres=m.origin+(idx+.5)*.25
    monkeypatch.setattr(tt_backend,'gaussian_voxelize',traced)
    out=m.update(source,centres,fused_source_points=source)
    assert len(called)==1
    expected,_=legacy(called[0],bounds=[m.config.bounds[:3],m.config.bounds[3:]],voxel_size=.25,sigma=.3)
    mask=np.zeros(m.shape,bool);mask[tuple(expected.T)]=True
    np.testing.assert_array_equal(out.collision3d,mask)
    assert 'collision_voxelizer' not in out.summary


@pytest.mark.parametrize('axis',[0,1,2])
def test_camera_collision_subset_uses_identical_indexing_at_nextafter_upper(axis):
    from orbvi_occ.camera_fusion import semantic_gaussian_voxelize,geometry_collision_voxelize
    config={'resolution':.3,'bounds':[-12,-12,-2,12,12,4],
        'ground_tolerance':.06,'min_ground_inliers':20,'gaussian_sigma':.3}
    mapper=LocalTraversability(config,collision_voxelizer=geometry_collision_voxelize)
    point=np.array([[.1,.1,.1]],np.float32)
    point[0,axis]=np.nextafter(np.float32(config['bounds'][axis+3]),np.float32(-np.inf))
    source=np.r_[floor_points().astype(np.float32),point]
    occ=semantic_gaussian_voxelize(source,np.ones((len(source),1),np.float32),
        bounds=[config['bounds'][:3],config['bounds'][3:]],voxel_size=.3,sigma=.3)
    result=mapper.update(source,occ['centers'],fused_source_points=source,free_evidence_source='predicted_camera_depth')
    assert result.summary['ground']['valid']
    assert result.collision3d.any()
    assert not np.any(result.collision3d & ~result.occupied3d)
    explicit,_=geometry_collision_voxelize(point,bounds=[config['bounds'][:3],config['bounds'][3:]],voxel_size=.3,sigma=.3)
    expected=np.zeros(mapper.shape,bool);expected[tuple(explicit.T)]=True
    np.testing.assert_array_equal(result.collision3d,expected)

