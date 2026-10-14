import numpy as np
import pytest

from orbvi_occ.tt_backend import TTBackend, gaussian_offsets, gaussian_voxelize, upstream_reference_voxelize


@pytest.mark.parametrize("voxel_size", [0.1, 0.2, 0.3, 0.5, 0.8])
def test_actual_upstream_gs2voxel_equivalence(voxel_size):
    """Run the pinned upstream implementation, not a second handwritten formula."""
    rng = np.random.default_rng(629)
    points = rng.uniform(-2.3, 2.3, (103, 3)).astype(np.float32)
    labels = rng.integers(0, 7, len(points), dtype=np.uint8)
    one_hot = np.eye(7, dtype=np.float32)[labels]
    bounds = np.array([[-3, -3, -3], [3, 3, 3]], np.float32)
    actual_idx, actual_cls = gaussian_voxelize(points, labels, bounds=bounds,
                                              voxel_size=voxel_size, indexing="upstream_trunc")
    reference_idx, reference_cls = upstream_reference_voxelize(points, one_hot, bounds, voxel_size)
    np.testing.assert_array_equal(actual_idx, reference_idx)
    np.testing.assert_array_equal(actual_cls, reference_cls)


def test_soft_semantic_votes_match_actual_upstream():
    rng = np.random.default_rng(799)
    points = rng.uniform(-1, 1, (40, 3)).astype(np.float32)
    semantic = rng.random((40, 4), dtype=np.float32)
    semantic /= semantic.sum(1, keepdims=True)
    bounds = np.array([[-2, -2, -2], [2, 2, 2]], np.float32)
    actual_idx, actual_cls = gaussian_voxelize(points, semantic, bounds=bounds, indexing="upstream_trunc")
    reference_idx, reference_cls = upstream_reference_voxelize(points, semantic, bounds)
    np.testing.assert_array_equal(actual_idx, reference_idx)
    np.testing.assert_array_equal(actual_cls, reference_cls)


def test_gaussian_footprint_is_not_point_only_voxelization():
    offsets, weight = gaussian_offsets(0.3, 0.3)
    assert len(offsets) == 7
    assert np.all(weight > 0.6065)
    idx, labels = gaussian_voxelize(np.array([[0.15, 0.15, 0.15]], np.float32), [2],
                                   bounds=np.array([[-3, -3, -3], [3, 3, 3]]))
    assert len(idx) == 7
    assert np.all(labels == 2)


def test_floor_fixes_negative_coordinate_collapse():
    points = np.array([[-0.05, 0.05, 0.05], [0.05, 0.05, 0.05]], np.float32)
    bounds = np.array([[-1, -1, -1], [1, 1, 1]], np.float32)
    fixed_idx, _ = gaussian_voxelize(points, [1, 2], bounds=bounds, sigma=0.001, voxel_size=0.25)
    upstream_idx, _ = gaussian_voxelize(points, [1, 2], bounds=bounds, sigma=0.001,
                                      voxel_size=0.25, indexing="upstream_trunc")
    assert len(fixed_idx) == 2
    assert len(upstream_idx) == 1


def test_causal_transform_expiry_and_dynamic_trail_removal():
    backend = TTBackend(dict(bounds=[[-5, -5, -3], [5, 5, 3]], history_seconds=0.3,
                             gaussian_sigma=0.001, voxel_size=0.2))
    original = np.array([[1.05, 0.05, 0.05], [2.05, 0.05, 0.05]], np.float32)
    first = backend.update(original, np.eye(4), 1_000_000_000, np.array([4, 8]), [False, True])
    assert len(first["current_centers"]) == 2
    moved_pose = np.eye(4); moved_pose[0, 3] = 0.4
    second = backend.update(np.array([[3.05, 0.05, 0.05]], np.float32), moved_pose, 1_100_000_000, [3])
    assert second["diagnostics"]["past_points"] == 1
    assert second["diagnostics"]["past_frames"] == 1
    np.testing.assert_allclose(second["fused_centers"][0], [0.7, 0.1, 0.1], atol=1e-5)
    assert 8 not in second["fused_labels"]
    expired = backend.update(np.empty((0, 3), np.float32), moved_pose, 2_000_000_000)
    assert expired["diagnostics"]["past_frames"] == 0
    assert len(expired["fused_centers"]) == 0
    with pytest.raises(ValueError, match="strictly increasing"):
        backend.update(original, np.eye(4), 1_900_000_000)


def test_history_never_reduces_occupied_geometry():
    backend = TTBackend(dict(bounds=[[-5, -5, -3], [5, 5, 3]], history_seconds=0.5))
    backend.update(np.array([[2, 1, 0]], np.float32), np.eye(4), 1_000_000_000)
    result = backend.update(np.array([[1, 1, 0]], np.float32), np.eye(4), 1_100_000_000)
    current = {tuple(x) for x in result["current_indices"]}
    fused = {tuple(x) for x in result["fused_indices"]}
    assert current <= fused
    assert result["diagnostics"]["iterations"] == 0
