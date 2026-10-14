#!/usr/bin/env python3
"""Independent acceptance for saved, real camera semantic occupancy artifacts.

This checker performs no inference. A full success requires every source-bag
time step, source-pixel label/confidence consistency, explicit garage IDs,
causal timestamps, occupied geometry support and conservative clearance.
Numerical consistency is not a ground-truth accuracy or official mIoU claim.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree

PROJECT = Path(__file__).resolve().parents[1]
UNKNOWN = 255
PHYSICAL_CAMERAS = frozenset(('2', '3', '4', '5'))


def strict_json(data):
    def invalid(value):
        raise ValueError(f"Non-finite JSON constant: {value}")
    return json.loads(data, parse_constant=invalid)


def read_json(path):
    return strict_json(Path(path).read_text())


def read_rows(path):
    return [strict_json(line) for line in Path(path).read_text().splitlines() if line.strip()]


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def first(data, *names):
    for name in names:
        if name in data:
            return data[name]
    raise ValueError(f"Missing required array: {' / '.join(names)}")


def probability(array, shape, name):
    array = np.asarray(array)
    require(array.shape == shape, f"{name} shape must be {shape}, got {array.shape}")
    require(np.isfinite(array).all() and np.all(array >= 0) and np.all(array <= 1 + 1e-6),
            f"{name} must be finite within [0, 1]")
    return array


def labels_valid(labels, classes, name):
    labels = np.asarray(labels)
    require(np.issubdtype(labels.dtype, np.integer), f"{name} requires integer IDs")
    require(np.all(np.isin(labels, [*classes, UNKNOWN])), f"{name} contains IDs outside the garage registry")
    return labels


def validate_camera_view_registry(views, source, frame, prepared_dir=None):
    """Prove one saved calibrated view per actual acquisition camera."""
    require(isinstance(views, list) and len(views) == 4,
            'Each acquisition requires exactly four camera views')
    physical = [str(int(view['physical_camera_id'])) for view in views]
    require(len(set(physical)) == 4 and set(physical) == PHYSICAL_CAMERAS,
            'Physical camera views must be four distinct cameras {3, 2, 4, 5}')
    require(set(frame['images']) == PHYSICAL_CAMERAS,
            'Prepared acquisition is missing an actual physical camera')
    names = [view['view_id'] for view in views]
    require(all(isinstance(name, str) and name for name in names) and len(set(names)) == 4,
            'Saved camera view IDs are missing or duplicated')
    for key in ('source_view_names', 'view_ids'):
        actual = np.asarray(first(source, key))
        require(actual.shape == (4,) and actual.tolist() == names,
                f'{key} differs from the four actual ordered saved camera views')
    require(np.asarray(first(source, 'source_image_shapes')).shape == (4, 2),
            'Depth image shape table must have exactly four rows')
    require(np.asarray(first(source, 'source_poses')).shape == (4, 4, 4) and
            np.asarray(first(source, 'source_intrinsics')).shape == (4, 3, 3) and
            np.asarray(first(source, 'source_stamp_ns')).shape == (4,),
            'Source camera pose/intrinsic/timestamp tables must contain four views')
    # Legacy camera_ids names virtual input slots 0..3; actual physical camera
    # identity is resolved above through source_view_ids and view descriptors.
    slots = np.asarray(first(source, 'source_view_ids'))
    camera_slots = np.asarray(first(source, 'camera_ids'))
    require(camera_slots.shape == slots.shape and np.array_equal(camera_slots, slots),
            'Legacy camera slot IDs disagree with exact source-view lineage')
    for view, camera in zip(views, physical):
        require(int(view['source_stamp_ns']) == frame['images'][camera]['stamp_ns'],
                'Saved view belongs to a different physical camera acquisition')
        if prepared_dir is not None:
            expected = (Path(prepared_dir) / frame['images'][camera]['path']).resolve()
            require(Path(view['source_image_path']).resolve() == expected,
                    'Saved physical camera view points to a different actual raw RGB file')
    return {'physical_camera_views': physical, 'actual_camera_view_count': 4}


def validate_raw_camera_inventory(prepared, manifest, sources):
    """Count real extracted raw images and validate all selected source links."""
    prepared = Path(prepared)
    index = read_json(prepared / 'camera_index.json')
    require(set(index) == PHYSICAL_CAMERAS, 'Raw image index has different physical cameras')
    counts, lookup = {}, {}
    for camera in sorted(PHYSICAL_CAMERAS):
        records = index[camera]
        require(isinstance(records, list), 'Raw camera index is not a sequence')
        topic = f'/orbvi/raw/camera_{camera}/image/compressed'
        count = len(records)
        require(count == manifest['counts'][topic] == manifest['bag_counts_all_topics'][topic],
                f'Camera {camera}: extracted raw count differs from actual source bag topic count')
        paths = [record['path'] for record in records]
        require(len(set(paths)) == count, f'Camera {camera}: raw image paths are duplicated')
        folder = prepared / 'images' / f'camera_{camera}'
        files = {str(path.relative_to(prepared)) for path in folder.glob('*.jpg') if path.is_file()}
        require(files == set(paths), f'Camera {camera}: actual retained raw JPEG files differ from source index')
        require(all((prepared / path).stat().st_size > 0 for path in paths),
                f'Camera {camera}: retained raw image is empty')
        stamps = [record['stamp_ns'] for record in records]
        require(all(isinstance(stamp, int) for stamp in stamps) and
                all(left < right for left, right in zip(stamps, stamps[1:])),
                f'Camera {camera}: raw acquisition timestamps are invalid or unordered')
        stats = manifest['time_statistics'][topic]
        require(stats['count'] == count and stats['first_stamp_ns'] == stamps[0] and stats['last_stamp_ns'] == stamps[-1],
                f'Camera {camera}: raw timeline differs from source bag topic statistics')
        counts[camera] = count
        lookup[camera] = {record['path']: record['stamp_ns'] for record in records}
    for frame in sources:
        require(set(frame['images']) == PHYSICAL_CAMERAS, 'Prepared time step lacks one of four cameras')
        for camera in PHYSICAL_CAMERAS:
            image = frame['images'][camera]
            require(lookup[camera].get(image['path']) == image['stamp_ns'] <= frame['stamp_ns'],
                    'Selected camera source is absent from actual raw image inventory or is from the future')
    total = sum(counts.values())
    require(total == 18787, 'Raw camera inventory does not cover all 18787 messages in this source bag')
    return {'raw_camera_images_by_physical_camera': counts, 'raw_camera_images_in_bag': total,
            'prepared_selected_camera_views': sum(len(frame['images']) for frame in sources)}


def validate_full_camera_counts(summary, raw_inventory, verified_frames, expected_frames=2348):
    selected = sum(frame['actual_camera_view_count'] for frame in verified_frames)
    require(len(verified_frames) == expected_frames and selected == expected_frames * 4 == 9392,
            'Actual verified source views do not cover all 9392 selected camera inputs')
    require(summary['camera_selected_view_count'] == selected,
            'Full summary selected view count differs from actual verified camera artifacts')
    require(summary['raw_camera_images_in_bag'] == raw_inventory['raw_camera_images_in_bag'] == 18787,
            'Full summary raw camera count differs from actual retained raw image inventory')
    return {'actually_verified_selected_views': selected,
            'actually_retained_raw_camera_images': raw_inventory['raw_camera_images_in_bag']}


def validate_hybrid_choice(sem, registry, fill_threshold=.5, column_threshold=.65):
    """Recompute every frozen-expert choice without importing production helper."""
    if 'expert_id' not in sem:
        return None
    class_ids = [int(item['id']) for item in registry['classes']]
    names = {item['name']: int(item['id']) for item in registry['classes']}
    official = np.asarray(first(sem, 'official_labels'))
    b5 = np.asarray(first(sem, 'b5_labels'))
    shape = official.shape
    require(official.ndim == 2 and b5.shape == shape, 'Hybrid experts have incompatible actual pixel domains')
    official_conf = probability(first(sem, 'official_confidence'), shape, 'Official expert confidence')
    b5_conf = probability(first(sem, 'b5_confidence'), shape, 'B5 mapped confidence')
    b5_native_conf = probability(first(sem, 'b5_source_confidence'), shape, 'B5 native confidence')
    ego = np.asarray(first(sem, 'ego_mask'), bool)
    valid = np.asarray(first(sem, 'rectification_valid_mask'), bool) & ~ego
    valid &= np.asarray(first(sem, 'official_valid_mask'), bool)
    valid &= np.asarray(first(sem, 'b5_valid_mask'), bool)
    require(np.array_equal(sem['valid_mask'], valid), 'Hybrid altered actual expert/rectification/body validity domain')
    ade_known = np.isin(b5, class_ids) & (b5 != names['sky'])
    fill = valid & (official == UNKNOWN) & ade_known & (b5_conf >= fill_threshold) & (b5_native_conf >= fill_threshold)
    subtype = valid & (official == names['wall']) & (b5 == names['column']) & (b5_conf >= column_threshold) & (b5_native_conf >= column_threshold)
    chosen_b5 = fill | subtype
    chosen_labels = official.copy()
    chosen_labels[chosen_b5] = b5[chosen_b5]
    accepted = valid & np.isin(chosen_labels, class_ids) & (chosen_labels != names['sky'])
    chosen_labels[~accepted] = UNKNOWN
    require(np.array_equal(sem['labels'], chosen_labels), 'Hybrid class selection differs from its declared fixed expert rules')
    expected_expert = np.full(shape, UNKNOWN, np.uint8)
    expected_expert[accepted] = 0
    expected_expert[fill & accepted] = 1
    expected_expert[subtype & accepted] = 2
    require(np.array_equal(sem['expert_id'], expected_expert), 'Hybrid expert provenance IDs do not describe actual class selection')
    for key in ('confidence', 'topk_labels', 'topk_probabilities', 'source_labels', 'source_confidence'):
        expected = np.asarray(first(sem, 'official_' + key)).copy()
        alternate = np.asarray(first(sem, 'b5_' + key))
        require(expected.shape == alternate.shape and expected.shape[:2] == shape, 'Hybrid expert array dimensions disagree: ' + key)
        expected[chosen_b5] = alternate[chosen_b5]
        if key in ('confidence', 'topk_probabilities', 'source_confidence'):
            expected[~accepted] = 0
        elif key == 'source_labels':
            expected[~accepted] = UNKNOWN
        require(np.array_equal(sem[key], expected), 'Hybrid fabricated or blended selected expert array: ' + key)
    rules = json.loads(str(first(sem, 'hybrid_rules_json')))
    require(rules.get('unknown_fill_confidence') == fill_threshold and rules.get('column_refine_confidence') == column_threshold,
            'Hybrid stored confidence gates differ from the accepted configuration')
    native = json.loads(str(first(sem, 'source_registry_per_expert_json')))
    require('ADE20K' in native.get('1', '') and 'ADE20K' in native.get('2', '') and 'OpenSeeD' in native.get('0', ''),
            'Hybrid expert-dependent native IDs are not explicitly named')
    return {'expert_pixels': {int(a): int(b) for a, b in zip(*np.unique(expected_expert, return_counts=True))},
            'unknown_filled_pixels': int(fill.sum()), 'official_wall_subtyped_column_pixels': int(subtype.sum())}


def validate_source_mapping(source, semantic_maps, registry):
    """Validate source-UV evidence numerically, without calling production lifting."""
    classes = [int(item['id']) for item in registry['classes']]
    points = np.asarray(first(source, 'points', 'xyz'), float)
    require(points.ndim == 2 and points.shape[1] == 3 and np.isfinite(points).all(), 'Source XYZ must be finite Nx3')
    n = len(points)
    require(n > 0, 'A committed camera frame has no source points')
    origins = np.asarray(first(source, 'origins'), float)
    require(origins.shape == points.shape and np.isfinite(origins).all(), 'Source origins must be finite Nx3')
    uv = np.asarray(first(source, 'source_uv'))
    vids = np.asarray(first(source, 'source_view_ids', 'view_ids'))
    shapes = np.asarray(first(source, 'source_image_shapes'))
    require(uv.shape == (n, 2) and np.issubdtype(uv.dtype, np.integer), 'Source UV must be integer Nx2')
    require(vids.shape == (n,) and np.issubdtype(vids.dtype, np.integer), 'Source view IDs must be integer N')
    require(shapes.shape == (len(semantic_maps), 2) and np.all(shapes > 0), 'Depth source image shapes must match view count')
    require(np.all((vids >= 0) & (vids < len(semantic_maps))), 'Point references a nonexistent source view')
    labels = labels_valid(first(source, 'labels', 'semantic_labels'), classes, 'Point semantics')
    require(labels.shape == (n,), 'Point labels must have one row per point')
    confidence = probability(first(source, 'semantic_confidence', 'confidence'), (n,), 'Point semantic confidence')
    evidence = probability(first(source, 'evidence', 'semantic_evidence'), (n, len(classes)), 'Point class evidence')
    depth_conf = np.asarray(first(source, 'depth_confidence', 'source_confidence'))
    require(depth_conf.shape == (n,) and np.isfinite(depth_conf).all() and np.all(depth_conf >= 0),
            'Depth confidence must have one finite nonnegative value per point')
    semantic_uv = np.asarray(first(source, 'semantic_uv'))
    require(semantic_uv.shape == (n, 2) and np.issubdtype(semantic_uv.dtype, np.integer), 'Semantic UV must be integer Nx2')
    known = labels != UNKNOWN
    require(np.all(confidence[~known] == 0) and np.all(evidence[~known] == 0),
            'Unknown semantic points must retain geometry with zero semantic evidence')
    require(np.all(evidence[known].sum(1) > 0), 'Known points must have positive semantic evidence')
    class_counts = Counter()
    view_quality = []
    uses_experts = any('expert_id' in sem for sem in semantic_maps)
    point_experts = None
    if uses_experts:
        point_experts = np.asarray(first(source, 'semantic_expert_id'))
        require(point_experts.shape == (n,) and np.issubdtype(point_experts.dtype, np.integer),
                'Hybrid point expert provenance must be integer N')
    for vid, sem in enumerate(semantic_maps):
        idx = np.flatnonzero(vids == vid)
        dh, dw = map(int, shapes[vid])
        require(np.all((uv[idx, 0] >= 0) & (uv[idx, 0] < dw) & (uv[idx, 1] >= 0) & (uv[idx, 1] < dh)),
                f'View {vid}: point UV outside the actual depth image')
        label_map = labels_valid(first(sem, 'labels'), classes, f'View {vid} pixel labels')
        require(label_map.ndim == 2, f'View {vid}: pixel semantics must be HxW')
        h, w = label_map.shape
        mask = np.asarray(first(sem, 'valid_mask'), bool)
        require(mask.shape == (h, w), f'View {vid}: valid mask shape mismatch')
        ego = None
        if 'ego_mask' in sem:
            ego = np.asarray(sem['ego_mask'])
            require(ego.shape == (h, w) and ego.dtype == np.bool_, f'View {vid}: ego mask must be boolean in the actual semantic domain')
            require(not np.any(mask & ego), f'View {vid}: ego body still marked a valid scene pixel')
            require(np.all(label_map[ego] == UNKNOWN), f'View {vid}: masked ego image pixels gained scene semantics')
        # Scaling integer pixel centres is deliberately computed independently.
        su = np.floor((uv[idx, 0] + 0.5) * w / dw).astype(int)
        sv = np.floor((uv[idx, 1] + 0.5) * h / dh).astype(int)
        expected_uv = np.column_stack([su, sv])
        require(np.array_equal(semantic_uv[idx], expected_uv), f'View {vid}: stored semantic UV does not represent its source pixel')
        require(np.array_equal(labels[idx], label_map[sv, su]), f'View {vid}: point labels differ from their actual source pixels')
        if point_experts is not None:
            require('expert_id' in sem and np.array_equal(point_experts[idx], sem['expert_id'][sv, su]),
                    f'View {vid}: point selected expert differs from its actual source pixel')
        require(np.all(mask[sv[known[idx]], su[known[idx]]]), f'View {vid}: invalid image pixels produced known semantics')
        conf_map = probability(first(sem, 'confidence'), (h, w), f'View {vid} semantic confidence')
        if ego is not None:
            require(np.all(conf_map[ego] == 0), f'View {vid}: masked ego pixels gained semantic confidence')
            require(np.all(np.asarray(first(sem, 'topk_probabilities'))[ego] == 0), f'View {vid}: masked ego pixels gained class evidence')
            require(not np.any(ego[sv, su]), f'View {vid}: actual sampled XYZ originates in the masked ego body')
        expected_conf = np.where(known[idx], conf_map[sv, su], 0)
        require(np.allclose(confidence[idx], expected_conf, atol=2e-6, rtol=0), f'View {vid}: point confidence differs from source pixel')
        top_ids = np.asarray(first(sem, 'topk_labels'))[sv, su]
        top_probs = np.asarray(first(sem, 'topk_probabilities'))[sv, su]
        require(top_ids.shape == top_probs.shape and top_ids.ndim == 2, f'View {vid}: top-k evidence shape mismatch')
        expected_evidence = np.zeros((len(idx), len(classes)), np.float32)
        for k in range(top_ids.shape[-1]):
            good = known[idx] & np.isin(top_ids[:, k], classes)
            np.add.at(expected_evidence, (np.flatnonzero(good), top_ids[good, k]), top_probs[good, k])
        require(np.allclose(evidence[idx], expected_evidence, atol=2e-6, rtol=0), f'View {vid}: point evidence differs from source top-k evidence')
        class_counts.update(int(label) for label in labels[idx])
        view_stats = {'view_index': vid, 'semantic_grid_pixels': h * w,
                      'semantic_grid_class_pixels': {int(a): int(b) for a, b in zip(*np.unique(label_map, return_counts=True))},
                      'sampled_point_count': len(idx), 'sampled_class_points': {int(a): int(b) for a, b in zip(*np.unique(labels[idx], return_counts=True))}}
        valid_key = next((key for key in ('source_valid_masks', 'depth_valid_masks') if key in source), None)
        if valid_key:
            depth_valid = np.asarray(source[valid_key][vid])
            require(depth_valid.shape == (dh, dw) and depth_valid.dtype == np.bool_, f'View {vid}: saved depth validity grid differs from its image domain')
            require(np.all(depth_valid[uv[idx, 1], uv[idx, 0]]), f'View {vid}: sampled source points originate in rejected depth pixels')
            yy, xx = np.indices((dh, dw))
            mx = np.floor((xx + .5) * w / dw).astype(int)
            my = np.floor((yy + .5) * h / dh).astype(int)
            depth_labels = label_map[my, mx]
            view_stats['depth_grid_class_pixels'] = {int(a): int(b) for a, b in zip(*np.unique(depth_labels, return_counts=True))}
            view_stats['depth_valid_class_pixels'] = {int(a): int(b) for a, b in zip(*np.unique(depth_labels[depth_valid], return_counts=True))}
            view_stats['depth_valid_pixels'] = int(depth_valid.sum())
        view_quality.append(view_stats)
    camera_points = np.asarray(first(source, 'source_camera_points'), float)
    transforms = np.asarray(first(source, 'T_rig_camera', 'source_poses'), float)
    require(camera_points.shape == points.shape and transforms.shape == (len(semantic_maps), 4, 4),
            'Camera source XYZ/rig transform dimensions invalid')
    reproduced = np.einsum('nij,nj->ni', transforms[vids, :3, :3], camera_points) + transforms[vids, :3, 3]
    require(np.allclose(reproduced, points, atol=3e-5, rtol=0), 'Saved source XYZ differs from calibrated source camera XYZ')
    require(np.allclose(origins, transforms[vids, :3, 3], atol=3e-6, rtol=0), 'Source ray origins differ from their actual camera origins')
    if 'source_depth' in source:
        require(np.allclose(source['source_depth'], camera_points[:, 2], atol=2e-5, rtol=0),
                'Saved source depth differs from its source camera Z')
    ranges = np.linalg.norm(points - origins, axis=1)
    car_id = next((int(item['id']) for item in registry['classes'] if item['name'] == 'car'), None)
    cars = labels == car_id if car_id is not None else np.zeros(n, bool)
    return {'point_count': n, 'known_point_count': int(known.sum()), 'class_points': dict(class_counts),
            'source_view_quality': view_quality,
            'car_points_under_2m': int(np.count_nonzero(cars & (ranges <= 2))),
            'car_points_over_2m': int(np.count_nonzero(cars & (ranges > 2))),
            'mean_known_confidence': float(confidence[known].mean()) if known.any() else None}


def reference_current_indices(source_points, bounds, size, sigma):
    """Independent physical member reference, including exact internal faces.

    The upstream 0.6065 Gaussian cutoff selects integer offsets inside the
    radius sphere (boundary exp(-0.5) is strictly larger than the cutoff).
    Preserve FP32 physical expansion, then assign its cells in FP64. No
    production rasterizer or its flattened-address path is called here.
    """
    bounds = np.asarray(bounds, np.float64)
    points = np.asarray(source_points, np.float32)
    points = points[np.isfinite(points).all(1) & (points >= bounds[0]).all(1) & (points < bounds[1]).all(1)]
    radius = int(np.clip(np.rint(np.float32(sigma / size + 1e-3)), 0, 3))
    axis = np.arange(-radius, radius + 1, dtype=np.int32)
    delta = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), -1).reshape(-1, 3)
    delta = delta[np.sum(delta * delta, axis=1) <= radius * radius]
    offsets = delta.astype(np.float32) * np.float32(size)
    shape = np.ceil((bounds[1] - bounds[0]) / size).astype(np.int64)
    parts = []
    for start in range(0, len(points), 20000):
        expanded = points[start:start + 20000, None, :] + offsets[None, :, :]
        inside = np.all((expanded >= bounds[0]) & (expanded < bounds[1]), axis=-1)
        indices = np.floor((expanded[inside].astype(np.float64) - bounds[0]) / size).astype(np.int64)
        indices = indices[np.all((indices >= 0) & (indices < shape), axis=1)]
        parts.append((indices[:, 0] * shape[1] + indices[:, 1]) * shape[2] + indices[:, 2])
    unique = np.unique(np.concatenate(parts)) if parts else np.empty(0, np.int64)
    return np.column_stack([unique // (shape[1] * shape[2]), (unique // shape[2]) % shape[1], unique % shape[2]])


def validate_occupancy(occ, source, registry, sigma, min_confidence=.35, min_support=.05):
    classes = [int(item['id']) for item in registry['classes']]
    bounds = np.asarray(first(occ, 'bounds'), float)
    size = float(first(occ, 'voxel_size'))
    require(bounds.shape == (2, 3) and np.isfinite(bounds).all() and np.all(bounds[1] > bounds[0]) and size > 0,
            'Occupancy bounds/voxel size invalid')
    shape = np.ceil((bounds[1] - bounds[0]) / size).astype(int)
    pose = np.asarray(first(occ, 'T_world_sensor'), float)
    require(pose.shape == (4, 4) and np.isfinite(pose).all() and np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6),
            'Occupancy pose must be a finite rigid 4x4 transform')
    require(np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-3) and
            np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-3), 'Occupancy pose rotation is not proper')
    result = {}
    for prefix in ('current', 'fused'):
        idx = np.asarray(first(occ, prefix + '_indices'))
        label = labels_valid(first(occ, prefix + '_labels'), classes, prefix + ' voxel labels')
        excluded_surface_ids = [int(item['id']) for item in registry['classes']
                                if item.get('occupancy_semantics_valid') is False]
        require(not np.any(np.isin(label, excluded_surface_ids)), prefix + ' labels declare a class excluded from finite occupied surfaces')
        require(idx.ndim == 2 and idx.shape[1] == 3 and np.issubdtype(idx.dtype, np.integer), prefix + ' indices must be integer Nx3')
        require(len(idx) > 0 and label.shape == (len(idx),), prefix + ' occupancy must be nonempty with aligned labels')
        require(np.all((idx >= 0) & (idx < shape)) and len(np.unique(idx, axis=0)) == len(idx), prefix + ' voxel indices duplicated or outside bounds')
        conf = probability(first(occ, prefix + '_confidence', prefix + '_semantic_confidence'), (len(idx),), prefix + ' semantic confidence')
        support = np.asarray(first(occ, prefix + '_known_support', prefix + '_known_support_weight'))
        require(support.shape == (len(idx),) and np.isfinite(support).all() and np.all(support >= 0), prefix + ' known support malformed')
        require(np.all(support[label != UNKNOWN] > 0), prefix + ' known voxel without positive known evidence')
        # Saved votes and the declared normalization operator are FP32. Keep
        # that numerical contract at the gate boundary; FP64 re-division can
        # put an exactly-threshold candidate on the opposite side by one ULP.
        accumulated = np.asarray(first(occ, prefix + '_evidence'), np.float32)
        require(accumulated.shape == (len(idx), len(classes)) and np.isfinite(accumulated).all() and np.all(accumulated >= 0),
                prefix + ' accumulated class evidence invalid')
        require(np.allclose(support, accumulated.sum(1), atol=1e-4, rtol=3e-6), prefix + ' known support differs from accumulated evidence')
        total = accumulated.sum(1)
        expected_score = np.divide(accumulated.max(1), total, out=np.zeros_like(total), where=total > 0)
        require(np.allclose(conf, expected_score, atol=2e-6, rtol=0), prefix + ' candidate confidence differs from recorded class evidence')
        expected_labels = accumulated.argmax(1).astype(np.uint8)
        rejected = (support < min_support) | (expected_score < min_confidence)
        expected_labels[rejected] = UNKNOWN
        require(np.array_equal(label, expected_labels), prefix + ' voxel labels do not implement declared evidence/confidence gates')
        centers = bounds[0] + (idx + .5) * size
        if prefix + '_centers' in occ:
            require(np.allclose(occ[prefix + '_centers'], centers, atol=1e-6, rtol=0), prefix + ' centers differ from their voxel indices')
        if prefix == 'current':
            expected_indices = reference_current_indices(first(source, 'points', 'xyz'), bounds, size, sigma)
            require(np.array_equal(idx, expected_indices),
                    'Current geometry differs from exact actual FP32 Gaussian members with FP64 cell assignment')
            # Independent conservative support envelope: any Gaussian-expanded
            # cell must lie near a real input point, including its half-diagonal.
            pts = np.asarray(first(source, 'points', 'xyz'), float)
            effective = np.clip(np.rint(np.float32(sigma / size + 1e-3)), 0, 3) * size
            nearest, _ = cKDTree(pts).query(centers, workers=1)
            require(np.all(nearest <= effective + np.sqrt(3) * size / 2 + 2e-5),
                    'Current occupied center has no geometrically plausible source member')
            result['current_exact_real_source_members_verified'] = True
        result[prefix + '_class_voxels'] = {int(a): int(b) for a, b in zip(*np.unique(label, return_counts=True))}
        result[prefix + '_known_voxels'] = int(np.count_nonzero(label != UNKNOWN))
    return result


def validate_traversability(trav, occ):
    """Ensure the safety artifact still corresponds to complete saved geometry."""
    bounds = np.asarray(first(trav, 'bounds'), float)
    resolution = float(first(trav, 'resolution'))
    require(bounds.shape == (6,) and np.isfinite(bounds).all() and resolution > 0, 'Invalid local traversal grid bounds')
    shape = np.ceil((bounds[3:] - bounds[:3]) / resolution).astype(int)
    require(np.all(shape > 0), 'Invalid traversal grid dimensions')
    occupied = np.asarray(first(trav, 'occupied3d'))
    free = np.asarray(first(trav, 'observed_free'))
    collision = np.asarray(first(trav, 'collision3d'))
    require(occupied.shape == tuple(shape) and free.shape == occupied.shape and collision.shape == occupied.shape,
            'Traversal evidence dimensions differ from declared grid')
    require(occupied.dtype == free.dtype == collision.dtype == np.bool_, 'Traversal evidence must be boolean')
    require(not np.any(free & collision), 'Occupied collision was also certified free')
    require(not np.any(collision & ~occupied), 'Collision geometry absent from occupancy')
    indices = occ['fused_indices']
    centers = occ['bounds'][0] + (indices + .5) * float(occ['voxel_size'])
    projected = np.floor((centers - bounds[:3]) / resolution).astype(int)
    inside = np.all((projected >= 0) & (projected < shape), axis=1)
    expected = np.zeros(tuple(shape), bool)
    expected[tuple(projected[inside].T)] = True
    require(np.array_equal(expected, occupied), 'Saved traversal occupancy differs from complete fused occupancy')
    grid = np.asarray(first(trav, 'traversability'))
    require(grid.shape == tuple(shape[:2]) and np.issubdtype(grid.dtype, np.integer) and np.all(np.isin(grid, [0, 1, 2])),
            'Predicted camera depth falsely certified measured full clearance or traversal IDs invalid')


def temporal_conflicts(occ):
    """Descriptive current-vs-fused same-cell category changes, not accuracy."""
    ci, fi = occ['current_indices'], occ['fused_indices']
    shape = np.ceil((occ['bounds'][1] - occ['bounds'][0]) / float(occ['voxel_size'])).astype(np.int64)
    current = np.ravel_multi_index(ci.T, shape); fused = np.ravel_multi_index(fi.T, shape)
    _, a, b = np.intersect1d(current, fused, return_indices=True)
    cl, fl = occ['current_labels'][a], occ['fused_labels'][b]
    known = (cl != UNKNOWN) & (fl != UNKNOWN)
    return {'shared_current_fused_cells': len(a), 'shared_both_known_cells': int(known.sum()),
            'current_fused_known_category_conflicts': int(np.count_nonzero(known & (cl != fl))),
            'current_known_lost_to_unknown_in_fusion': int(np.count_nonzero((cl != UNKNOWN) & (fl == UNKNOWN)))}


def validate_temporal_fusion(occ, row, config):
    """Independently enforce the saved causal policy and current-class priority.

    This does not infer an object's velocity or establish pose accuracy.
    A consistent class vote may still be a mistaken pretrained prediction.
    """
    shape = np.ceil((occ['bounds'][1].astype(np.float64) - occ['bounds'][0]) /
                    float(occ['voxel_size'])).astype(np.int64)
    current = np.ravel_multi_index(occ['current_indices'].T, shape)
    fused = np.ravel_multi_index(occ['fused_indices'].T, shape)
    shared, current_rows, fused_rows = np.intersect1d(current, fused, return_indices=True)
    require(len(shared) == len(current), 'Time fusion omitted actual current occupied geometry')
    threshold = config.get('current_priority_min_confidence', .45)
    strong = (occ['current_labels'][current_rows] != UNKNOWN) & (occ['current_confidence'][current_rows] >= threshold)
    require(np.array_equal(occ['current_labels'][current_rows[strong]], occ['fused_labels'][fused_rows[strong]]),
            'Time fusion replaced a current class above the declared class-independent priority gate')
    diagnostic = row['algorithm']
    count = diagnostic['past_frames']
    require(isinstance(count, int) and 0 <= count <= min(config.get('max_history_frames', 6), row['index']),
            'Time fusion history count exceeds its causal bounded prefix')
    require(diagnostic['history_seconds'] == config.get('history_seconds', .6) and
            diagnostic['history_decay_seconds'] == config.get('history_decay_seconds', .25),
            'Time fusion declared horizon/decay differs from executed configuration')
    if not row['auxiliary']['camera_motion']['accepted']:
        require(count == 0 and diagnostic['past_points'] == 0, 'Rejected camera motion still injected temporal history')
    if count == 0:
        require(np.array_equal(occ['current_indices'], occ['fused_indices']) and
                np.array_equal(occ['current_labels'], occ['fused_labels']) and
                np.array_equal(occ['current_evidence'], occ['fused_evidence']),
                'Time fusion without historical source points fabricated geometry or semantic votes')
    return {'current_geometry_preserved_in_fusion': True, 'current_priority_cells_verified': int(strong.sum()),
            'bounded_causal_history_frames': count}


def verify(run_dir, prepared_dir=None, expected_frames=2348, allow_partial=False, frame_indices=None):
    run_dir, prepared = Path(run_dir).resolve(), Path(prepared_dir or PROJECT / 'prepared').resolve()
    checks, errors, frames = [], [], []
    def check(name, ok, detail=None):
        checks.append({'name': name, 'passed': bool(ok), 'detail': detail})
    report = {'run_directory': str(run_dir), 'scope': 'Camera semantic engineering consistency; no ground-truth mIoU',
              'expected_full_recording_frames': expected_frames, 'allow_partial': allow_partial, 'checks': checks, 'artifact_errors': errors}
    try:
        # Bind this report to the exact executed run and committed journal
        # snapshot. A later rebuild may use the same folder name; its report
        # must never inherit a previous run's acceptance by path alone.
        run_bytes = (run_dir / 'run.json').read_bytes()
        journal_bytes = (run_dir / 'frames.jsonl').read_bytes()
        metadata = strict_json(run_bytes)
        require(not journal_bytes or journal_bytes.endswith(b'\n'), 'Frame journal has an uncommitted partial tail')
        rows = [strict_json(line) for line in journal_bytes.splitlines() if line.strip()]
        report['run_identity'] = {'config_sha256': metadata.get('config_sha256'),
            'bag_sha256': metadata.get('bag_sha256'), 'inference_cache_key': metadata.get('inference_cache_key'),
            'run_json_sha256': hashlib.sha256(run_bytes).hexdigest(),
            'frames_jsonl_sha256': hashlib.sha256(journal_bytes).hexdigest()}
        registry = metadata.get('semantic_registry') or read_json(run_dir / 'semantic_registry.json')
        sources = read_rows(prepared / 'frames.jsonl')
        manifest = read_json(prepared / 'manifest.json')
    except (OSError, ValueError, KeyError) as exc:
        check('required_documents', False, str(exc)); report['accepted'] = False; return report
    ids = [r.get('index') for r in rows]
    check('camera_route', metadata.get('route') == 'camera')
    check('camera_only_runtime_inputs', metadata.get('runtime_lidar_input') is False and
          metadata.get('runtime_vio_input') is False and metadata.get('pose_source') == 'camera_depth_only')
    check('full_source_manifest', manifest.get('complete') is True and len(sources) == expected_frames)
    check('source_index_contiguous', [frame.get('index') for frame in sources] == list(range(expected_frames)))
    raw_inventory = None
    try:
        raw_inventory = validate_raw_camera_inventory(prepared, manifest, sources)
        report['camera_source_inventory'] = raw_inventory
        check('actual_raw_camera_inventory_and_selection_complete',
              raw_inventory['prepared_selected_camera_views'] == expected_frames * 4, raw_inventory)
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        check('actual_raw_camera_inventory_and_selection_complete', False, str(exc))
    full = ids == list(range(expected_frames)) and metadata.get('limited_development_run') is False and frame_indices is None
    check('all_full_bag_time_steps_processed', full, {'recorded': len(ids), 'expected': expected_frames})
    check('recorded_indices_unique_ordered', ids == sorted(set(ids)))
    if frame_indices is not None:
        selected = sorted(set(frame_indices))
        check('requested_actual_frames_are_committed', bool(selected) and all(index in ids for index in selected))
        rows = [row for row in rows if row['index'] in selected]
        report['artifact_selection'] = {'mode': 'explicit_read_only_sample', 'indices': selected,
            'complete_recording_claim': False}
    summary = None
    if full:
        try:
            summary = read_json(run_dir / 'summary.json'); progress = read_json(run_dir / 'progress.json')
            check('runner_final_state_committed', summary.get('complete') is True and progress.get('complete') is True and
                  progress.get('processed_frames') == expected_frames)
            coverage = summary.get('acceptance', {})
            check('full_summary_sensor_timestamp_coverage', coverage.get('recorded_frames') == expected_frames and
                  coverage.get('first_timestamp_ns') == sources[0]['stamp_ns'] and
                  coverage.get('last_timestamp_ns') == sources[-1]['stamp_ns'] and
                  coverage.get('timestamps_monotonic') is True and coverage.get('missing_frame_ids') == [])
            check('no_unavailable_semantic_truth_score', summary.get('semantic_miou') is None)
        except (OSError, ValueError) as exc:
            check('runner_final_state_committed', False, str(exc))
    check('run_source_bag_fingerprint', metadata.get('bag_sha256') == manifest.get('source_bag_sha256'))
    provenance = metadata.get('input_provenance_sha256', {})
    config = metadata.get('config', {})
    fingerprint = hashlib.sha256(json.dumps({'config': config, 'sources': provenance}, sort_keys=True).encode()).hexdigest()
    check('configuration_and_source_fingerprint', bool(provenance) and metadata.get('config_sha256') == fingerprint)
    changed = []
    for name, digest in provenance.items():
        archived = run_dir / 'source_snapshot' / name
        if not archived.is_file() or hashlib.sha256(archived.read_bytes()).hexdigest() != digest:
            changed.append(name)
    check('executed_source_snapshot_matches', bool(provenance) and not changed, changed)
    classes = registry.get('classes', [])
    class_ids = [int(item['id']) for item in classes]
    check('explicit_garage_registry', bool(classes) and class_ids == list(range(len(classes))) and registry.get('unknown_id') == UNKNOWN)
    names = {item['name']: int(item['id']) for item in classes}
    check('required_garage_meanings', all(name in names for name in ('wall', 'floor', 'ceiling', 'column', 'car', 'person')))
    check('wall_zero_not_official_noise', bool(classes) and classes[0].get('name') == 'wall')
    sem_provider = metadata.get('semantics', metadata.get('semantic_provider', {}))
    report['actual_semantic_provider'] = sem_provider
    report['actual_depth_provider'] = metadata.get('camera_depth_provider', metadata.get('depth_provider', {}))
    check('untrained_on_orbvi_declared', sem_provider.get('trained_on_orbvi') is False)
    check('semantic_provider_identity_explicit', bool(sem_provider.get('provider')))
    check('real_frozen_semantic_weights_identity', bool(sem_provider.get('repository')) and bool(sem_provider.get('revision')) and
          sem_provider.get('weights_loaded_strict') is True)
    if sem_provider.get('experts'):
        for name, identity in sem_provider['experts'].items():
            check('frozen_hybrid_expert_identity_' + name, bool(identity.get('repository')) and bool(identity.get('revision')) and
                  identity.get('weights_loaded_strict') is True and identity.get('trained_on_orbvi') is False)
        check('hybrid_method_deviation_disclosed', sem_provider.get('original_paper_reproduction') is False and
              sem_provider.get('cross_expert_probability_comparison') is False)
    depth_provider = report['actual_depth_provider']
    check('camera_depth_has_no_lidar_input', depth_provider.get('input_lidar') is False and depth_provider.get('input_depth') is False)
    check('real_frozen_depth_weights_loaded', depth_provider.get('weights_loaded_strict') is True and bool(depth_provider.get('revision')))
    sigma = config.get('ttocc', {}).get('gaussian_sigma', .3)
    for row in rows:
        i = row.get('index')
        try:
            require(isinstance(i, int) and 0 <= i < len(sources), 'Invalid committed frame index')
            frame = sources[i]
            require(row['stamp_ns'] == frame['stamp_ns'], 'Frame timestamp differs from source manifest')
            source = load_npz(run_dir / 'source_points' / f'{i:06d}.npz')
            occ = load_npz(run_dir / 'occupancy' / f'{i:06d}.npz')
            trav = load_npz(run_dir / 'traversability' / f'{i:06d}.npz')
            require(int(first(source, 'stamp_ns')) == frame['stamp_ns'] == int(first(occ, 'stamp_ns')), 'Saved point/voxel timestamps differ from acquisition')
            view_document = read_json(run_dir / 'views' / f'{i:06d}.json')
            view_list = view_document if isinstance(view_document, list) else view_document['views']
            view_quality = validate_camera_view_registry(view_list, source, frame, prepared)
            require(row['auxiliary']['camera_maps'] == view_quality['actual_camera_view_count'],
                    'Journal camera view count differs from actual four-camera source artifacts')
            maps = [load_npz(run_dir / view['semantic_maps_path']) for view in view_list]
            source_stamps = np.asarray(first(source, 'source_stamp_ns'))
            require(source_stamps.shape == (len(view_list),), 'Point lineage camera timestamp lookup dimensions invalid')
            source_poses = np.asarray(first(source, 'source_poses'))
            source_intrinsics = np.asarray(first(source, 'source_intrinsics'))
            expert_quality = []
            for vidx, (view, sem) in enumerate(zip(view_list, maps)):
                vid = view['view_id']; cid = str(view['physical_camera_id'])
                stamp = int(first(sem, 'source_stamp_ns'))
                require(stamp == frame['images'][cid]['stamp_ns'] and stamp <= frame['stamp_ns'], f'View {vid} semantic timestamp is not its causal source acquisition')
                require(str(first(sem, 'pixel_domain')) == 'rectified_virtual_pinhole', f'View {vid} cache is not in the calibrated rectified domain')
                require(stamp == int(view['source_stamp_ns']), f'View {vid} registry and map timestamps disagree')
                require(stamp == int(source_stamps[vidx]), f'View {vid} point lineage camera timestamp disagrees')
                require(np.allclose(source_poses[vidx], view['T_rig_camera'], atol=3e-6, rtol=0), f'View {vid} source rig transform differs from actual view descriptor')
                k = np.asarray(view['K'], float).copy()
                h, w = maps[vidx]['labels'].shape
                dh, dw = source['source_image_shapes'][vidx]
                k[0] *= dw / w; k[1] *= dh / h
                require(np.allclose(source_intrinsics[vidx], k, atol=3e-5, rtol=0), f'View {vid} depth K was not resized with its actual RGB domain')
                expert_report = validate_hybrid_choice(sem, registry,
                    config.get('semantics', {}).get('unknown_fill_confidence', .5),
                    config.get('semantics', {}).get('column_refine_confidence', .65))
                if expert_report is not None:
                    expert_quality.append({'view_id': vid, **expert_report})
            quality = validate_source_mapping(source, maps, registry)
            quality.update(view_quality)
            if expert_quality:
                quality['hybrid_expert_quality'] = expert_quality
            quality.update(validate_occupancy(occ, source, registry, sigma,
                config.get('ttocc', {}).get('min_semantic_confidence', .35),
                config.get('ttocc', {}).get('min_semantic_support', .05)))
            validate_traversability(trav, occ)
            quality.update(temporal_conflicts(occ))
            quality.update(validate_temporal_fusion(occ, row, config.get('ttocc', {})))
            require(len(occ['current_indices']) == row['current_occupied_voxels'] and len(occ['fused_indices']) == row['fused_occupied_voxels'],
                    'Frame journal counts differ from actual semantic occupancy')
            require(abs(float(trav['timestamp']) - frame['stamp_ns'] / 1e9) < 1e-6, 'Traversal timestamp differs from acquisition')
            quality.update(index=i, stamp_ns=frame['stamp_ns'])
            frames.append(quality)
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            errors.append({'index': i, 'error': str(exc)})
            if len(errors) >= 128:
                break
    check('all_committed_semantic_artifacts_valid', not errors and len(frames) == len(rows), {'verified': len(frames), 'committed': len(rows), 'errors': len(errors)})
    report['frames'] = frames
    report['statistics'] = {'verified_frames': len(frames), 'source_point_count': sum(f['point_count'] for f in frames),
                            'source_known_point_count': sum(f['known_point_count'] for f in frames),
                            'actual_selected_camera_views': sum(f['actual_camera_view_count'] for f in frames)}
    if full:
        try:
            require(summary is not None and raw_inventory is not None, 'Complete camera source inventory/summary is unavailable')
            counts = validate_full_camera_counts(summary, raw_inventory, frames, expected_frames)
            check('full_camera_counts_match_actual_selected_and_retained_artifacts', True, counts)
        except (ValueError, KeyError, TypeError) as exc:
            check('full_camera_counts_match_actual_selected_and_retained_artifacts', False, str(exc))
    report['engineering_consistency_accepted'] = all(c['passed'] for c in checks if c['name'] != 'all_full_bag_time_steps_processed')
    report['accepted'] = report['engineering_consistency_accepted'] and full
    report['exit_success'] = report['accepted'] or (allow_partial and report['engineering_consistency_accepted'])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=PROJECT / 'runs/camera_semantic_full')
    parser.add_argument('--prepared', type=Path, default=PROJECT / 'prepared')
    parser.add_argument('--expected-frames', type=int, default=2348)
    parser.add_argument('--allow-partial', action='store_true', help='Development only: full-recording accepted remains false')
    parser.add_argument('--frames', type=int, nargs='+', help='Read-only sampled indices; full-recording accepted always remains false')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = verify(args.run, args.prepared, args.expected_frames, args.allow_partial, args.frames)
    output = args.output or PROJECT / 'diagnostics/camera_semantic_qa/semantic_acceptance.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps({'accepted': result.get('accepted'), 'engineering_consistency_accepted': result.get('engineering_consistency_accepted'),
                      'checks': len(result['checks']), 'artifact_errors': len(result['artifact_errors']), 'report': str(output)}, ensure_ascii=False))
    sys.exit(0 if result.get('exit_success') else 2)


if __name__ == '__main__':
    main()
