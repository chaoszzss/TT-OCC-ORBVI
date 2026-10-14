"""Adversarial saved-artifact contract checks; no inference is mocked here."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('independent_semantic_acceptance', PROJECT / 'scripts/verify_camera_semantic.py')
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def artifact_contract():
    """Two-view correspondence example for checker corruption tests only."""
    registry = {'unknown_id': 255, 'classes': [{'id': 0, 'name': 'wall'}, {'id': 1, 'name': 'car'}]}
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    poses[1, 0, 3] = .2
    camera_points = np.array([[0., 0., 2.], [.2, 0., 2.], [0., 0., 3.]], np.float32)
    view_ids = np.array([0, 1, 1], np.uint8)
    labels = np.array([0, 1, 255], np.uint8)
    source = {'points': camera_points + poses[view_ids, :3, 3], 'origins': poses[view_ids, :3, 3],
              'source_uv': np.array([[0, 0], [0, 0], [1, 0]], np.int32),
              'semantic_uv': np.array([[0, 0], [0, 0], [1, 0]], np.int32),
              'source_view_ids': view_ids, 'source_image_shapes': np.array([[1, 2], [1, 2]]),
              'labels': labels, 'semantic_confidence': np.array([.8, .9, 0.], np.float32),
              'evidence': np.array([[.8, .1], [.05, .9], [0., 0.]], np.float32),
              'depth_confidence': np.ones(3, np.float32), 'source_camera_points': camera_points,
              'source_poses': poses}
    maps = [
        {'labels': np.array([[0, 255]], np.uint8), 'confidence': np.array([[.8, 0]], np.float32),
         'valid_mask': np.ones((1, 2), bool), 'topk_labels': np.array([[[0, 1], [0, 1]]], np.uint8),
         'topk_probabilities': np.array([[[.8, .1], [0, 0]]], np.float32)},
        {'labels': np.array([[1, 255]], np.uint8), 'confidence': np.array([[.9, 0]], np.float32),
         'valid_mask': np.ones((1, 2), bool), 'topk_labels': np.array([[[1, 0], [0, 1]]], np.uint8),
         'topk_probabilities': np.array([[[.9, .05], [0, 0]]], np.float32)}]
    return source, maps, registry


def test_independent_checker_validates_correspondence_and_pose_geometry():
    source, maps, registry = artifact_contract()
    result = checker.validate_source_mapping(source, maps, registry)
    assert result['point_count'] == 3 and result['known_point_count'] == 2
    assert result['class_points'] == {0: 1, 1: 1, 255: 1}


def test_checker_rejects_label_replaced_by_another_view():
    source, maps, registry = artifact_contract()
    source['labels'][1] = 0
    with pytest.raises(ValueError, match='actual source pixels'):
        checker.validate_source_mapping(source, maps, registry)


def test_checker_rejects_unknown_points_given_positive_class_evidence():
    source, maps, registry = artifact_contract()
    source['evidence'][2, 0] = .4
    with pytest.raises(ValueError, match='zero semantic evidence'):
        checker.validate_source_mapping(source, maps, registry)


def test_checker_rejects_confidence_detached_from_its_actual_pixel():
    source, maps, registry = artifact_contract()
    source['semantic_confidence'][1] = .8
    with pytest.raises(ValueError, match='confidence differs from source pixel'):
        checker.validate_source_mapping(source, maps, registry)


def test_checker_rejects_point_xyz_with_wrong_source_camera_transform():
    source, maps, registry = artifact_contract()
    source['points'][1, 0] += .01
    with pytest.raises(ValueError, match='calibrated source camera XYZ'):
        checker.validate_source_mapping(source, maps, registry)


def test_checker_rejects_point_ids_in_a_different_numeric_taxonomy():
    source, maps, registry = artifact_contract()
    source['labels'][1] = 20
    with pytest.raises(ValueError, match='outside the garage registry'):
        checker.validate_source_mapping(source, maps, registry)


def test_checker_rejects_shifted_uv_even_when_shape_is_plausible():
    source, maps, registry = artifact_contract()
    source['semantic_uv'][1, 0] = 1
    with pytest.raises(ValueError, match='source pixel'):
        checker.validate_source_mapping(source, maps, registry)


def test_checker_rejects_ego_geometry_even_if_its_semantics_is_unknown():
    source, maps, registry = artifact_contract()
    # The third point is unknown and has zero class evidence, but still must
    # not be emitted from a pixel explicitly masked as the acquisition body.
    maps[0]['ego_mask'] = np.zeros((1, 2), bool)
    maps[1]['ego_mask'] = np.array([[False, True]])
    maps[1]['valid_mask'][0, 1] = False
    with pytest.raises(ValueError, match='masked ego body'):
        checker.validate_source_mapping(source, maps, registry)


def rejected_voxel_contract():
    registry = {'classes': [{'id': 0, 'name': 'wall'}, {'id': 1, 'name': 'floor'}, {'id': 2, 'name': 'car'}]}
    source = {'points': np.array([[0, 0, 2]], np.float32)}
    occ = {'bounds': np.array([[-1, -1, 1], [1, 1, 3]], np.float32), 'voxel_size': .3, 'T_world_sensor': np.eye(4)}
    for prefix in ('current', 'fused'):
        occ[prefix + '_indices'] = np.array([[3, 3, 3]], np.int32)
        occ[prefix + '_labels'] = np.array([255], np.uint8)
        occ[prefix + '_confidence'] = np.array([.34], np.float32)
        occ[prefix + '_known_support'] = np.array([1.], np.float32)
        occ[prefix + '_evidence'] = np.array([[.34, .33, .33]], np.float32)
    return occ, source, registry


def test_rejected_unknown_voxel_may_retain_its_true_candidate_score():
    occ, source, registry = rejected_voxel_contract()
    result = checker.validate_occupancy(occ, source, registry, sigma=.1)
    assert result['current_known_voxels'] == 0


def test_unknown_voxel_cannot_store_arbitrary_confidence():
    occ, source, registry = rejected_voxel_contract()
    occ['current_confidence'][0] = 1.
    with pytest.raises(ValueError, match='recorded class evidence'):
        checker.validate_occupancy(occ, source, registry, sigma=.1)


def test_unknown_voxel_requires_actual_support_or_confidence_rejection():
    occ, source, registry = rejected_voxel_contract()
    occ['current_evidence'][0] = [.8, .1, .1]
    occ['current_confidence'][0] = .8
    with pytest.raises(ValueError, match='declared evidence/confidence gates'):
        checker.validate_occupancy(occ, source, registry, sigma=.1)


def hybrid_saved_contract():
    """Explicit saved-array contract, not a mock neural-network prediction."""
    registry = {'classes': [{'id': i, 'name': n} for i, n in enumerate(
        ('wall', 'floor', 'ceiling', 'column', 'car', 'person', 'door', 'vegetation', 'other', 'road', 'sky'))]}
    sem = {'valid_mask': np.ones((1, 5), bool), 'ego_mask': np.zeros((1, 5), bool),
           'rectification_valid_mask': np.ones((1, 5), bool)}
    original_labels = np.array([[0, 255, 4, 255, 0]], np.uint8)
    alternative_labels = np.array([[3, 1, 3, 4, 3]], np.uint8)
    original_conf = np.array([[.8, 0, .8, 0, .8]], np.float32)
    alternative_conf = np.array([[.9, .9, .9, .8, .9]], np.float32)
    native_b5 = np.array([[.9, .9, .9, .4, .6]], np.float32)
    for prefix, labels, confidence in [('official', original_labels, original_conf), ('b5', alternative_labels, alternative_conf)]:
        sem[prefix + '_labels'] = labels
        sem[prefix + '_confidence'] = confidence
        sem[prefix + '_valid_mask'] = np.ones((1, 5), bool)
        sem[prefix + '_topk_labels'] = np.stack([np.where(labels == 255, 0, labels), np.ones((1, 5), np.uint8), np.full((1, 5), 2, np.uint8)], -1)
        sem[prefix + '_topk_probabilities'] = np.stack([confidence, confidence * .1, confidence * .05], -1)
        sem[prefix + '_source_labels'] = labels.copy() if prefix == 'official' else np.array([[42, 3, 42, 20, 42]], np.uint8)
        sem[prefix + '_source_confidence'] = confidence.copy() if prefix == 'official' else native_b5
    sem['labels'] = np.array([[3, 1, 4, 255, 0]], np.uint8)
    sem['expert_id'] = np.array([[2, 1, 0, 255, 0]], np.uint8)
    for key in ('confidence', 'topk_labels', 'topk_probabilities', 'source_labels', 'source_confidence'):
        sem[key] = sem['official_' + key].copy()
        sem[key][:, :2] = sem['b5_' + key][:, :2]
    sem['hybrid_rules_json'] = np.array(json.dumps({'unknown_fill_confidence': .5, 'column_refine_confidence': .65}))
    sem['source_registry_per_expert_json'] = np.array(json.dumps({'0': 'OpenSeeD garage', '1': 'ADE20K', '2': 'ADE20K'}))
    return sem, registry


def test_independent_hybrid_rule_check_preserves_known_primary_and_double_gates():
    sem, registry = hybrid_saved_contract()
    result = checker.validate_hybrid_choice(sem, registry)
    assert result['unknown_filled_pixels'] == 1 and result['official_wall_subtyped_column_pixels'] == 1


def test_hybrid_cannot_replace_known_official_car_with_another_expert_column():
    sem, registry = hybrid_saved_contract()
    sem['labels'][0, 2] = 3
    sem['expert_id'][0, 2] = 2
    with pytest.raises(ValueError, match='declared fixed expert rules'):
        checker.validate_hybrid_choice(sem, registry)


def test_hybrid_cannot_fill_unknown_when_native_confidence_fails():
    sem, registry = hybrid_saved_contract()
    sem['labels'][0, 3] = 4
    sem['expert_id'][0, 3] = 1
    with pytest.raises(ValueError, match='declared fixed expert rules'):
        checker.validate_hybrid_choice(sem, registry)


def test_hybrid_selected_confidence_cannot_be_averaged_or_invented():
    sem, registry = hybrid_saved_contract()
    sem['confidence'][0, 0] = .85
    with pytest.raises(ValueError, match='fabricated or blended'):
        checker.validate_hybrid_choice(sem, registry)


def test_real_frame68_nextafter_model_member_does_not_cross_grid_upper_face():
    """Actual cached model point9984, source view0/UV156,184; no NN mock.

    Its positive-X Gaussian member is exactly the final representable FP32
    value below 12m. FP32 addition of the origin offset used to round to24m,
    generating an invalid x-index80 instead of79 on the 80-cell grid.
    """
    from orbvi_occ.camera_fusion import semantic_gaussian_voxelize
    point = np.array([[1094398770, 1066267021, 3215734218]], np.uint32).view(np.float32)
    evidence = np.array([[0., .6429516077041626, .01414030697196722, 0., 0., 0., 0., 0., 0.,
                          .1456748992204666, 0.]], np.float32)
    expanded_x = point[0, 0] + np.float32(.3)
    assert expanded_x == np.nextafter(np.float32(12), np.float32(-np.inf))
    result = semantic_gaussian_voxelize(point, evidence, bounds=[[-12, -12, -2], [12, 12, 4]],
                                        voxel_size=.3, sigma=.3)
    indices = result['indices']
    assert np.all((indices >= 0) & (indices < np.array([80, 80, 20])))
    assert [79, 43, 2] in indices.tolist()
    assert [80, 43, 2] not in indices.tolist()


def test_real_frame9_collision_callback_stays_inside_same_camera_occupancy_grid():
    """Actual model point56294 exposes an interior Y-boundary mismatch.

    The deterministic planar samples provide ground support for the downstream
    calculation; they are mathematical test inputs, not simulated NN outputs.
    The single non-floor member comes byte-exact from the real frame9 cache.
    """
    from orbvi_occ.camera_fusion import semantic_gaussian_voxelize, geometry_collision_voxelize
    from orbvi_occ.traversability import LocalTraversability
    real = np.array([[1092142525, 1091253043, 3214357589]], np.uint32).view(np.float32)
    axes = np.linspace(-2., 2., 17, dtype=np.float32)
    x, y = np.meshgrid(axes, axes, indexing='ij')
    floor = np.column_stack([x.ravel(), y.ravel(), np.full(x.size, -1.8, np.float32)])
    source = np.concatenate([floor, real])
    bounds = [[-12, -12, -2], [12, 12, 4]]
    occupied = semantic_gaussian_voxelize(source, np.zeros((len(source), 1), np.float32),
                                         bounds=bounds, voxel_size=.3, sigma=.3)
    result = LocalTraversability({'resolution': .3, 'bounds': [-12, -12, -2, 12, 12, 4]},
                                 collision_voxelizer=geometry_collision_voxelize).update(
        source, occupied['centers'], fused_source_points=source, free_evidence_source='predicted_camera_depth')
    assert result.summary['ground']['valid']
    assert not np.any(result.collision3d & ~result.occupied3d)
    assert result.collision3d[71, 68, 1]
    assert not result.collision3d[71, 69, 1]
    assert not np.any(result.traversability == 3)


def test_independent_checker_rejects_history_suppressing_an_actual_current_car():
    occ = {'bounds': np.array([[-1., -1., -1.], [1., 1., 1.]]), 'voxel_size': .3,
           'current_indices': np.array([[1, 1, 1]]), 'fused_indices': np.array([[1, 1, 1]]),
           'current_labels': np.array([4], np.uint8), 'fused_labels': np.array([0], np.uint8),
           'current_confidence': np.array([.9], np.float32)}
    row = {'index': 1, 'algorithm': {'past_frames': 1, 'past_points': 20, 'history_seconds': .6,
                                   'history_decay_seconds': .25},
           'auxiliary': {'camera_motion': {'accepted': True}}}
    with pytest.raises(ValueError, match='class-independent priority'):
        checker.validate_temporal_fusion(occ, row, {})


def test_independent_checker_rejects_history_surviving_a_camera_motion_reset():
    occ = {'bounds': np.array([[-1., -1., -1.], [1., 1., 1.]]), 'voxel_size': .3,
           'current_indices': np.array([[1, 1, 1]]), 'fused_indices': np.array([[1, 1, 1]]),
           'current_labels': np.array([4], np.uint8), 'fused_labels': np.array([4], np.uint8),
           'current_confidence': np.array([.9], np.float32)}
    row = {'index': 1, 'algorithm': {'past_frames': 1, 'past_points': 20, 'history_seconds': .6,
                                   'history_decay_seconds': .25},
           'auxiliary': {'camera_motion': {'accepted': False}}}
    with pytest.raises(ValueError, match='Rejected camera motion'):
        checker.validate_temporal_fusion(occ, row, {})


def camera_registry_contract():
    """Saved-metadata corruption fixture; no simulated neural inference."""
    cameras = (3, 2, 4, 5)
    names = [f'{camera}_v0' for camera in cameras]
    views = [{'physical_camera_id': camera, 'view_id': name, 'source_stamp_ns': 100}
             for camera, name in zip(cameras, names)]
    source = {'source_view_names': np.array(names), 'view_ids': np.array(names),
              'source_image_shapes': np.full((4, 2), 336), 'source_poses': np.repeat(np.eye(4)[None], 4, axis=0),
              'source_intrinsics': np.repeat(np.eye(3)[None], 4, axis=0), 'source_stamp_ns': np.full(4, 100),
              'camera_ids': np.arange(4, dtype=np.uint8), 'source_view_ids': np.arange(4, dtype=np.uint8)}
    frame = {'images': {str(camera): {'stamp_ns': 100} for camera in cameras}}
    return views, source, frame


def test_camera_registry_requires_each_of_four_physical_cameras_once():
    views, source, frame = camera_registry_contract()
    assert checker.validate_camera_view_registry(views, source, frame)['actual_camera_view_count'] == 4
    views[1]['physical_camera_id'] = 3
    with pytest.raises(ValueError, match='four distinct cameras'):
        checker.validate_camera_view_registry(views, source, frame)


def test_camera_registry_rejects_a_missing_fourth_view():
    views, source, frame = camera_registry_contract()
    with pytest.raises(ValueError, match='exactly four'):
        checker.validate_camera_view_registry(views[:3], source, frame)


def test_camera_registry_cannot_permute_sources_while_retaining_four_camera_count():
    views, source, frame = camera_registry_contract()
    views[0], views[1] = views[1], views[0]
    with pytest.raises(ValueError, match='actual ordered saved camera views'):
        checker.validate_camera_view_registry(views, source, frame)


def test_raw_camera_inventory_rejects_duplicate_files_counted_as_extra_messages(tmp_path):
    # A forged index could report two records while pointing to one retained
    # image. Check actual path uniqueness before accepting any aggregate count.
    index = {'2': [{'path': 'images/camera_2/000000.jpg', 'stamp_ns': 100},
                   {'path': 'images/camera_2/000000.jpg', 'stamp_ns': 101}], '3': [], '4': [], '5': []}
    (tmp_path / 'camera_index.json').write_text(json.dumps(index))
    topic = '/orbvi/raw/camera_2/image/compressed'
    manifest = {'counts': {topic: 2}, 'bag_counts_all_topics': {topic: 2}}
    with pytest.raises(ValueError, match='raw image paths are duplicated'):
        checker.validate_raw_camera_inventory(tmp_path, manifest, [])


def test_full_camera_count_cannot_claim_a_missing_acquisition_was_processed():
    summary = {'camera_selected_view_count': 9392, 'raw_camera_images_in_bag': 18787}
    inventory = {'raw_camera_images_in_bag': 18787}
    actual = [{'actual_camera_view_count': 4} for _ in range(2347)]
    with pytest.raises(ValueError, match='do not cover all 9392'):
        checker.validate_full_camera_counts(summary, inventory, actual)


def test_full_summary_must_match_the_actual_selected_and_retained_camera_counts():
    actual = [{'actual_camera_view_count': 4} for _ in range(2348)]
    inventory = {'raw_camera_images_in_bag': 18787}
    summary = {'camera_selected_view_count': 9392, 'raw_camera_images_in_bag': 18787}
    assert checker.validate_full_camera_counts(summary, inventory, actual)['actually_verified_selected_views'] == 9392
    summary['camera_selected_view_count'] = 9396
    with pytest.raises(ValueError, match='actual verified camera artifacts'):
        checker.validate_full_camera_counts(summary, inventory, actual)
    summary['camera_selected_view_count'] = 9392
    summary['raw_camera_images_in_bag'] = 9392
    with pytest.raises(ValueError, match='actual retained raw image inventory'):
        checker.validate_full_camera_counts(summary, inventory, actual)
