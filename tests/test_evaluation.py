import numpy as np
import pytest

from orbvi_occ.evaluation import (FullBagAcceptance, GeometryMetricAccumulator,
                                  assert_no_source_leakage, evaluate_geometry,
                                  split_holdout)


def test_spatial_holdout_prevents_duplicate_point_leakage():
    rng = np.random.default_rng(10)
    original = rng.normal(size=(500, 3))
    original = np.r_[original, original[:100], [[np.nan, 0, 0]]]
    fit, held, fit_ids, held_ids = split_holdout(original, fraction=0.2)
    assert len(fit) + len(held) == 600
    assert len(held) > 0 and len(fit) > 0
    assert_no_source_leakage(fit_ids, held_ids)
    fit_cells = set(map(tuple, np.floor(fit / 0.3).astype(int)))
    held_cells = set(map(tuple, np.floor(held / 0.3).astype(int)))
    assert not fit_cells.intersection(held_cells)
    reversed_fit, reversed_held, _, _ = split_holdout(original[:-1][::-1], fraction=0.2)
    assert set(map(tuple, fit)) == set(map(tuple, reversed_fit))
    assert set(map(tuple, held)) == set(map(tuple, reversed_held))


def test_source_leakage_is_error():
    with pytest.raises(ValueError, match="overlap"):
        assert_no_source_leakage([1, 2, 3], [3, 4])


def test_empty_holdout_reports_unavailable_not_perfect():
    report = evaluate_geometry(np.array([[0, 0, 0]]), np.empty((0, 3)))
    assert report["holdout_returns"] == 0
    assert report["endpoint_hit_rate"] is None


def test_geometry_and_false_free_denominators_use_only_local_holdout():
    free = np.zeros((4, 4, 4), bool)
    free[2, 2, 2] = True
    pred = np.array([[0.1, 0.1, 0.1]])
    held = np.array([[0.1, 0.1, 0.1], [1.1, 1.1, 1.1], [9, 9, 9]])
    report = evaluate_geometry(pred, held, voxel_size=1, tolerance_m=0,
                               observed_free=free, bounds=[-1, -1, -1, 3, 3, 3])
    assert report["holdout_returns"] == 2
    assert report["endpoint_hit_rate"] == 0.5
    assert report["false_free_count"] == 1
    assert report["false_free_rate"] == 0.5


def test_accumulator_weights_returns_not_frame_averages():
    metrics = GeometryMetricAccumulator()
    metrics.add({"holdout_returns": 10, "endpoint_hits": 10})
    metrics.add({"holdout_returns": 90, "endpoint_hits": 0})
    assert metrics.summary()["weighted_endpoint_hit_rate"] == 0.1


def test_all_2348_frames_are_required_for_acceptance():
    acceptance = FullBagAcceptance(expected_frames=2348)
    for i in range(2348):
        acceptance.add(i, 1_000_000_000 + i * 100_000_000, success=True,
                       input_points=100, occupancy_voxels=10, traversability_cells=6400)
    report = acceptance.summary()
    assert report["accepted"]
    assert report["recorded_frames"] == 2348
    incomplete = FullBagAcceptance(expected_frames=2348)
    for r in acceptance.records[:-1]:
        incomplete.add(r["frame_id"], r["timestamp_ns"], True)
    assert not incomplete.summary()["accepted"]
    assert incomplete.summary()["missing_frame_ids"] == [2347]


def test_duplicate_or_failed_frame_rejects_complete_looking_run():
    run = FullBagAcceptance(expected_frames=2)
    run.add(0, 1, True)
    run.add(0, 2, True)
    assert not run.summary()["accepted"]
    failed = FullBagAcceptance(expected_frames=2)
    failed.add(0, 1, True)
    failed.add(1, 2, False, error="conversion failed")
    assert failed.summary()["failed_frame_ids"] == [1]
    assert not failed.summary()["accepted"]


def test_duplicate_timestamps_and_truncated_endpoints_are_rejected():
    duplicate = FullBagAcceptance(expected_frames=2)
    duplicate.add(0, 10, True)
    duplicate.add(1, 10, True)
    assert not duplicate.summary()["accepted"]
    boundaries = FullBagAcceptance(expected_frames=2, expected_start_ns=10, expected_end_ns=20)
    boundaries.add(0, 10, True)
    boundaries.add(1, 19, True)
    assert not boundaries.summary()["accepted"]
    assert not boundaries.summary()["last_timestamp_matches_expected"]
