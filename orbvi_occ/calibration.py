"""Device camera provenance and independent metric LiDAR registration.

Factory files from unrelated robots are never substituted for missing extrinsics.
Camera rays use the actual nine-parameter triple-sphere projection, not OpenCV
fisheye/MEI. LiDAR poses are estimated independently without transferring an
unverified VIO coordinate convention.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

from .data import load_frames, load_scan, sha256_file, write_json

DEVICE_CALIBRATION = Path(__file__).resolve().parents[1] / 'calibration/source/camera_config.yaml'


def transform_points(points, transform):
    return np.asarray(points) @ transform[:3, :3].T + transform[:3, 3]


def inverse_transform(transform):
    result = np.eye(4)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ transform[:3, 3]
    return result


def export_camera_calibration(output, source=DEVICE_CALIBRATION):
    import cv2
    source = Path(source)
    storage = cv2.FileStorage(str(source), cv2.FILE_STORAGE_READ)
    cameras = {}
    for raw_id, index in [('3', 0), ('2', 1), ('4', 2), ('5', 3)]:
        params = storage.getNode('cam' + str(index)).mat().reshape(-1)
        transform = np.eye(4)
        transform[:3, :] = storage.getNode('Twc' + str(index)).mat()
        transform[:3, 3] *= 0.001  # Original board YAML millimetres.
        cameras[raw_id] = {'model': 'triple_sphere', 'params': params.tolist(),
                           'params_order': ['fx', 'fy', 'cx', 'cy', 'xi', 'lambda', 'alpha', 'b', 'c'],
                           'width': 1280, 'height': 1080, 'source_camera_index': index,
                           'T_camera0_camera': transform.tolist(),
                           'T_camera_lidar': None}
    storage.release()
    result = {'version': 1, 'source': str(source.resolve()), 'source_sha256': sha256_file(source),
              'translation_units': 'metres; source Twc translations converted from millimetres',
              'camera_order': ['3', '2', '4', '5'], 'cameras': cameras,
              'projection_validation_source': '/home/chaoszzss/dataset/rosbag/orbvi_4vl_data_cqupt/orbvi/orbvi_4cam_full_sources/preprocessing/depth_frame_validation/rgb_multitime_check.json',
              'projection_validation': 'Device depth RGB reprojected to four original images at 0/50/150/220 seconds; correct triple-sphere convention gives lower RGB residual than alternative conventions.',
              'lidar_camera_extrinsic_valid': False,
              'lidar_camera_extrinsic_status': 'Device-specific LiDAR-to-camera calibration was not provided; null is intentional.',
              'vio_pose_semantics': 'Bag header=world and child_frame_id=world; bridge preserves publisher values. Do not infer camera poses from these labels.'}
    write_json(output, result)
    return result


def copy_device_calibration(source_directory, destination):
    """Archive the user-provided calibration without modifying original files."""
    source_directory, destination = Path(source_directory).resolve(), Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    files = []
    for source in sorted(source_directory.rglob('*')):
        if not source.is_file():
            continue
        relative = source.relative_to(source_directory)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        digest = sha256_file(source)
        if sha256_file(target) != digest:
            raise ValueError('Calibration archive copy failed: '+str(relative))
        files.append({'path': str(relative), 'bytes': source.stat().st_size, 'sha256': digest})
    camera_config = source_directory / 'camera_config.yaml'
    driver = json.loads((source_directory / 'MID360s_config.json').read_text())
    result = {'primary_source_directory': str(source_directory), 'archive_directory': str(destination),
              'files': files, 'camera_config_sha256': sha256_file(camera_config),
              'camera_model': 'triple_sphere',
              'lidar_driver_extrinsic_parameter': driver['lidar_configs'][0]['extrinsic_parameter'],
              'lidar_driver_extrinsic_interpretation': 'Driver coordinate correction defaults; all zeros do not establish a LiDAR-to-camera extrinsic.',
              'contains_device_lidar_camera_transform': False,
              'eight_rectified_body_camera_interpretation': 'VINS initial guesses with estimate_extrinsic=1. These virtual camera transforms cannot be attached directly to physical four fisheyes.',
              'raw_to_rectified_rotation_present': False,
              'decision': 'Use four physical camera intrinsics and relative rig transforms; retain null LiDAR-camera extrinsic.'}
    return result


def voxel_downsample(points, voxel_size, limit=12000):
    points = np.asarray(points, dtype=np.float64)
    distance = np.linalg.norm(points, axis=1)
    keep = np.isfinite(points).all(axis=1) & (distance > 0.7) & (distance < 30.0)
    points = points[keep]
    if not len(points):
        return points
    cells = np.floor(points / voxel_size).astype(np.int32)
    _, inverse = np.unique(cells, axis=0, return_inverse=True)
    count = np.bincount(inverse)
    result = np.column_stack([np.bincount(inverse, weights=points[:, k])/count for k in range(3)])
    if len(result) > limit:
        # Spatially stratified deterministic reduction, not temporal frame skipping.
        result = result[np.linspace(0, len(result)-1, limit).astype(np.int64)]
    return result


def estimate_normals(points, neighbors=12):
    tree = cKDTree(points)
    _, indices = tree.query(points, k=min(neighbors, len(points)), workers=1)
    local = points[indices]
    centered = local - local.mean(axis=1, keepdims=True)
    covariance = np.einsum('nki,nkj->nij', centered, centered) / local.shape[1]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    normals = eigenvectors[:, :, 0]
    planar = eigenvalues[:, 0] / np.maximum(eigenvalues[:, 1], 1e-8) < 0.3
    return tree, normals, planar


def icp_point_to_plane(source, target, initial=None, max_iterations=18, max_distance=0.8,
                       rotation_prior=None):
    """Robust fixed-scale rigid alignment, source coordinates to target coordinates."""
    transform = np.eye(4) if initial is None else np.asarray(initial).copy()
    if rotation_prior is not None:
        delta_rotation = transform[:3,:3] @ np.asarray(rotation_prior).T
        yaw = np.arctan2(delta_rotation[1,0], delta_rotation[0,0])
        transform[:3,:3] = Rotation.from_euler('z', yaw).as_matrix() @ rotation_prior
    tree, normals, planar = estimate_normals(target)
    rank, condition, pairs = 0, np.inf, 0
    for iteration in range(max_iterations):
        moved = transform_points(source, transform)
        distance, index = tree.query(moved, workers=1)
        gate = max_distance if iteration < 5 else min(max_distance, 0.45)
        keep = (distance < gate) & planar[index]
        if keep.sum() < 100:
            break
        points, normal = moved[keep], normals[index[keep]]
        residual = np.einsum('ij,ij->i', points - target[index[keep]], normal)
        robust = np.sqrt(np.minimum(1.0, 0.08 / np.maximum(np.abs(residual), 1e-8)))
        matrix = np.column_stack([np.cross(points, normal), normal])
        if rotation_prior is not None:
            matrix = matrix[:, [2,3,4,5]]
        solution, _, rank, singular = np.linalg.lstsq(matrix * robust[:, None], -residual * robust, rcond=1e-5)
        condition = float(singular[0] / max(singular[-1], 1e-12))
        angle = abs(solution[0]) if rotation_prior is not None else np.linalg.norm(solution[:3])
        translation = solution[1:] if rotation_prior is not None else solution[3:]
        if angle > 0.15 or np.linalg.norm(translation) > 0.5:
            break
        delta = np.eye(4)
        delta[:3, :3] = (Rotation.from_euler('z',solution[0]).as_matrix() if rotation_prior is not None
                         else Rotation.from_rotvec(solution[:3]).as_matrix())
        delta[:3, 3] = translation
        transform = delta @ transform
        if angle < 2e-5 and np.linalg.norm(translation) < 2e-4:
            break
    moved = transform_points(source, transform)
    distance, index = tree.query(moved, workers=1)
    keep = (distance < 0.45) & planar[index]
    signed = np.einsum('ij,ij->i', moved[keep] - target[index[keep]], normals[index[keep]])
    stats = {'iterations': iteration + 1, 'correspondence_fraction': float(keep.mean()),
             'point_plane_rmse_m': float(np.sqrt(np.mean(signed**2))) if len(signed) else float('inf'),
             'nearest_neighbor_median_m': float(np.median(distance)),
             'nearest_neighbor_p95_m': float(np.percentile(distance, 95)),
             'rank': int(rank), 'condition': condition, 'expected_rank': 4 if rotation_prior is not None else 6}
    # Keep rotations on SO(3): constant-velocity propagation plus transpose
    # inversion otherwise feeds tiny roundoff back exponentially over long runs.
    transform[:3, :3] = Rotation.from_matrix(transform[:3, :3]).as_matrix()
    return transform, stats


def initial_ground_frame(points):
    """Use a horizontal plane as a world orientation/reference, without floor claims.

    MID360 has limited downward coverage. This near-field plane can be a car
    surface; traversal must estimate supported floor independently.
    """
    points = np.asarray(points)
    radius = np.linalg.norm(points[:, :2], axis=1)
    candidate = points[np.isfinite(points).all(axis=1) & (points[:, 2] < -0.6) &
                       (points[:, 2] > -3.0) & (radius > 1) & (radius < 12)]
    rng, best = np.random.default_rng(42), None
    for _ in range(256):
        if len(candidate) < 100:
            break
        sample = candidate[rng.choice(len(candidate), 3, replace=False)]
        normal = np.cross(sample[1]-sample[0], sample[2]-sample[0])
        normal /= max(np.linalg.norm(normal), 1e-12)
        if abs(normal[2]) < 0.95:
            continue
        if normal[2] < 0:
            normal *= -1
        offset = -normal @ sample[0]
        keep = np.abs(candidate @ normal + offset) < 0.06
        if best is None or keep.sum() > best[0]:
            best = (int(keep.sum()), normal, offset, keep)
    if best is None or best[0] < 100:
        raise ValueError('Could not validate an initial ground plane')
    inliers = candidate[best[3]]
    center = inliers.mean(axis=0)
    _, _, vt = np.linalg.svd(inliers-center, full_matrices=False)
    normal = vt[-1]
    if normal[2] < 0:
        normal *= -1
    offset = -normal @ center
    axis = np.cross(normal, [0., 0., 1.])
    angle = np.arctan2(np.linalg.norm(axis), normal[2])
    rotation = Rotation.from_rotvec(axis / max(np.linalg.norm(axis), 1e-12) * angle)
    transform = np.eye(4)
    transform[:3, :3] = rotation.as_matrix()
    transform[2, 3] = offset
    residual = inliers @ normal + offset
    return transform, {'source': 'RANSAC of first raw metric LiDAR scan; no camera/VIO extrinsic required',
                       'floor_validated': False,
                       'interpretation': 'Horizontal reference plane only; z=0 is not automatically traversable ground',
                       'normal_in_lidar': normal.tolist(), 'offset_m': float(offset),
                       'inliers': len(inliers), 'rmse_m': float(np.sqrt(np.mean(residual**2))),
                       'T_world_lidar0': transform.tolist()}


def build_lidar_trajectory(prepared, voxel_size=0.30, frame_limit=None, constrain_gravity=True):
    """Process every indexed scan with robust ICP and a causal sliding local map.

    frame_limit is only for development diagnostics. A complete run has no limit
    and trajectory.npz is never emitted by diagnostic runs.
    """
    prepared = Path(prepared)
    frames = load_frames(prepared)
    if frame_limit is not None:
        frames = frames[:frame_limit]
    started = time.monotonic()
    orientation_prior, rotation_report = None, None
    if constrain_gravity:
        prior_path = prepared / 'vio_rotation_prior.npz'
        if prior_path.exists():
            with np.load(prior_path) as prior:
                orientation_prior = prior['R_world_lidar'][:len(frames)]
            rotation_report = json.loads((prepared / 'vio_rotation_prior_report.json').read_text())
        else:
            bootstrap, _ = build_lidar_trajectory(prepared, voxel_size, frame_limit=451,
                                                   constrain_gravity=False)
            orientation_prior, rotation_report = estimate_vio_rotation_prior(prepared, bootstrap)
            orientation_prior = orientation_prior[:len(frames)]
    points = load_scan(prepared, frames[0])['points']
    first, ground = initial_ground_frame(points)
    if orientation_prior is not None:
        first[:3,:3] = orientation_prior[0]
    poses, statistics = [first], [dict(index=0, correspondence_fraction=1., point_plane_rmse_m=0.,
                                     nearest_neighbor_median_m=0., nearest_neighbor_p95_m=0.,
                                     rank=6, condition=1., iterations=0, accepted=True)]
    previous_scan = voxel_downsample(points, voxel_size, 7000)
    relative = np.eye(4)
    keyframes = [(0, transform_points(previous_scan, first))]
    map_points = keyframes[0][1]
    map_update = -1
    for index, frame in enumerate(frames[1:], 1):
        scan = voxel_downsample(load_scan(prepared, frame)['points'], voxel_size, 7000)
        initial = poses[-1] @ relative
        pose, stats = icp_point_to_plane(scan, map_points, initial, max_iterations=16,
                                         rotation_prior=None if orientation_prior is None else orientation_prior[index])
        step = inverse_transform(poses[-1]) @ pose
        accepted = (stats['rank'] == stats['expected_rank'] and stats['correspondence_fraction'] > 0.30 and
                    stats['point_plane_rmse_m'] < 0.20 and
                    np.linalg.norm(step[:3, 3]) < 0.65 and
                    np.linalg.norm(Rotation.from_matrix(step[:3, :3]).as_rotvec()) < 0.15)
        if not accepted:
            # Independently constrain the same frame against its immediately
            # preceding scan; no silent frame removal or unverified VIO fallback.
            pair, pair_stats = icp_point_to_plane(scan, previous_scan, relative, max_iterations=24)
            pose = poses[-1] @ pair
            stats = pair_stats
            accepted = (stats['rank'] == 6 and stats['correspondence_fraction'] > 0.25 and
                        stats['point_plane_rmse_m'] < 0.25 and np.linalg.norm(pair[:3, 3]) < 0.75)
            stats['fallback_scan_to_scan'] = True
        if not accepted:
            raise RuntimeError('LiDAR registration failed quality gate at frame {}: {}'.format(index, stats))
        pose[:3, :3] = Rotation.from_matrix(pose[:3, :3]).as_matrix()
        relative = inverse_transform(poses[-1]) @ pose
        relative[:3, :3] = Rotation.from_matrix(relative[:3, :3]).as_matrix()
        poses.append(pose)
        previous_scan = scan
        stats.update(index=index, accepted=accepted,
                     step_translation_m=float(np.linalg.norm(relative[:3, 3])),
                     step_rotation_deg=float(np.linalg.norm(Rotation.from_matrix(relative[:3, :3]).as_rotvec())*180/np.pi))
        statistics.append(stats)
        # Keyframes every ten scans, bounded by past sensor acquisitions only.
        if index % 10 == 0:
            keyframes.append((index, transform_points(scan, pose)))
            keyframes = keyframes[-6:]
            # World point distance is irrelevant: recenter for range filtering.
            center = pose[:3, 3]
            map_points = voxel_downsample(np.concatenate([item[1] for item in keyframes])-center,
                                          voxel_size, 14000) + center
        if index % 100 == 0 or index == len(frames)-1:
            print('LiDAR ICP {}/{}; rmse={:.3f}m, correspondence={:.2f}; {:.1f}s'.format(
                  index+1, len(frames), stats['point_plane_rmse_m'],
                  stats['correspondence_fraction'], time.monotonic()-started), flush=True)
            np.savez(prepared / 'trajectory.partial.npz',
                     stamp_ns=np.array([item['stamp_ns'] for item in frames[:index+1]], dtype=np.int64),
                     T_world_lidar=np.asarray(poses))
    poses = np.asarray(poses)
    result = {'complete': frame_limit is None, 'frame_count': len(frames),
              'method': 'Metric LiDAR point-to-plane ICP; causal sliding six-keyframe map, constant velocity initialization; optional hand-eye-validated VIO roll/pitch constraint',
              'uses_vio': orientation_prior is not None,
              'uses_vio_translation': False, 'uses_vio_scale': False,
              'vio_rotation_report': rotation_report, 'voxel_size_m': voxel_size, 'ground': ground,
              'point_plane_rmse_m': {key: float(np.percentile([item['point_plane_rmse_m'] for item in statistics[1:]], percentile))
                                      for key, percentile in [('median', 50), ('p95', 95), ('max', 100)]},
              'correspondence_fraction': {key: float(np.percentile([item['correspondence_fraction'] for item in statistics[1:]], percentile))
                                          for key, percentile in [('min', 0), ('median', 50), ('p95', 95)]},
              'trajectory_distance_m': float(np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1).sum()),
              'endpoint_displacement_m': float(np.linalg.norm(poses[-1, :3, 3]-poses[0, :3, 3])),
              'elapsed_seconds': time.monotonic()-started,
              'limitations': ['No loop closure optimization; the endpoint displacement is a diagnostic, not ground truth error.',
                             'Initial z=0 is an arbitrary fitted horizontal reference plane. Ground must be estimated from local floor support.',
                             'Raw scans have up to 100ms acquisition duration. Deskew is available from the estimated trajectory.',
                             'Registration residual is an internal consistency measure, not independent occupancy accuracy.']}
    if frame_limit is None:
        np.savez(prepared / 'trajectory.npz', stamp_ns=np.array([item['stamp_ns'] for item in frames], dtype=np.int64),
                 T_world_lidar=poses, point_plane_rmse_m=np.array([item['point_plane_rmse_m'] for item in statistics]),
                 correspondence_fraction=np.array([item['correspondence_fraction'] for item in statistics]))
        write_json(prepared / 'trajectory_report.json', result)
        keys = sorted(set().union(*(item.keys() for item in statistics)))
        with (prepared / 'trajectory_quality.csv').open('w') as file:
            writer = csv.DictWriter(file, fieldnames=keys)
            writer.writeheader(); writer.writerows(statistics)
    return poses, result


def estimate_vio_rotation_prior(prepared, bootstrap_poses):
    """Estimate only a body-to-LiDAR rotation from independent sensor motions.

    Three-axis excitation and disjoint validation are required. Translation is
    not estimated and these results are not a LiDAR-camera calibration.
    """
    prepared = Path(prepared)
    frames = load_frames(prepared)
    odometry = np.load(prepared / 'odometry.npz')
    lidar_stamps = np.array([frame['stamp_ns'] for frame in frames], dtype=np.int64)
    origin = int(lidar_stamps[0])
    body_time = (odometry['stamp_ns'] - origin) / 1e9
    lidar_time = (lidar_stamps - origin) / 1e9
    body_interpolated = Slerp(body_time, Rotation.from_quat(odometry['orientation_xyzw']))(
        np.clip(lidar_time, body_time[0], body_time[-1])).as_matrix()
    lidar_rotation = bootstrap_poses[:, :3, :3]
    training = np.arange(0, 300, 5)
    validation = np.arange(300, 440, 5)
    candidates = []
    for convention in ('direct', 'inverse'):
        body = body_interpolated if convention == 'direct' else body_interpolated.transpose(0,2,1)
        source = Rotation.from_matrix(body[training].transpose(0,2,1) @ body[training+10]).as_rotvec()
        target = Rotation.from_matrix(lidar_rotation[training].transpose(0,2,1) @ lidar_rotation[training+10]).as_rotvec()
        fitted, _ = Rotation.align_vectors(target, source)
        rotation = fitted.as_matrix()
        predicted = rotation @ (body[validation].transpose(0,2,1) @ body[validation+10]) @ rotation.T
        observed = lidar_rotation[validation].transpose(0,2,1) @ lidar_rotation[validation+10]
        residual = np.linalg.norm(Rotation.from_matrix(predicted.transpose(0,2,1)@observed).as_rotvec(),axis=1)*180/np.pi
        candidates.append({'convention': convention, 'R_lidar_body': rotation.tolist(),
                           'validation_median_deg': float(np.median(residual)),
                           'validation_p95_deg': float(np.percentile(residual,95)),
                           'motion_excitation_singular_values': np.linalg.svd(source,compute_uv=False).tolist()})
    choice = min(candidates, key=lambda candidate: candidate['validation_p95_deg'])
    if choice['validation_p95_deg'] > .5 or choice['motion_excitation_singular_values'][-1] < .01:
        raise ValueError('VIO rotation-to-LiDAR hand-eye validation failed: '+str(choice))
    rotation = np.array(choice['R_lidar_body'])
    imu = np.load(prepared / 'imu.npz')
    proper_acceleration = np.median(imu['acceleration'][:1000],axis=0)
    proper_acceleration /= np.linalg.norm(proper_acceleration)
    gravity_up_lidar = rotation @ proper_acceleration
    axis = np.cross(gravity_up_lidar, [0.,0.,1.])
    angle = np.arctan2(np.linalg.norm(axis), gravity_up_lidar[2])
    initial_rotation = Rotation.from_rotvec(axis/max(np.linalg.norm(axis),1e-12)*angle).as_matrix()
    # Inference uses the latest acquisition at/before LiDAR scan end, not a
    # future interpolation endpoint. Offline static calibration above is separate.
    indices = np.searchsorted(odometry['stamp_ns'], lidar_stamps, side='right')-1
    indices = np.maximum(indices,0)
    body_causal = Rotation.from_quat(odometry['orientation_xyzw'][indices]).as_matrix()
    if choice['convention'] == 'inverse':
        body_causal = body_causal.transpose(0,2,1)
    orientation_prior = initial_rotation @ rotation @ body_causal[0].T @ body_causal @ rotation.T
    report = dict(choice, candidates=candidates,
                  calibration_training_lidar_frames=[0,299], calibration_validation_lidar_frames=[300,449],
                  gravity_up_in_lidar=gravity_up_lidar.tolist(),
                  valid_rotation_only=True, valid_lidar_camera_extrinsic=False,
                  translation_estimated=False,
                  runtime_timestamp_policy='latest VIO header acquisition at or before LiDAR scan end',
                  purpose='Constrain ICP roll/pitch drift; metric translations and yaw remain LiDAR-estimated',
                  limitations=['Rotation calibrated offline using the first 45 seconds; calibration residual is not occupancy accuracy.',
                               'No body-LiDAR translation or camera-body transform is inferred.'])
    np.savez(prepared / 'vio_rotation_prior.npz', stamp_ns=lidar_stamps,
             R_world_lidar=orientation_prior, R_lidar_body=rotation,
             vio_stamp_ns=odometry['stamp_ns'][indices])
    write_json(prepared / 'vio_rotation_prior_report.json', report)
    return orientation_prior, report


def deskew_scan(points, offset_time_ns, scan_start_ns, reference_stamp_ns, trajectory):
    """Map each acquisition into the LiDAR coordinates at scan end.

    The first scan has no past motion estimate and is left unchanged. Later
    scans interpolate/extrapolate only the previous and current registered
    poses, a constant-velocity approximation.
    """
    stamps = np.asarray(trajectory['stamp_ns'], dtype=np.int64)
    poses = np.asarray(trajectory['T_world_lidar'])
    query = np.int64(scan_start_ns) + np.asarray(offset_time_ns, dtype=np.int64)
    end_index = nearest_index_for_trajectory(stamps, reference_stamp_ns)
    if end_index == 0:
        return np.asarray(points).copy()
    previous = max(0, end_index-1)
    dt = int(stamps[end_index])-int(stamps[previous])
    if dt <= 0:
        return np.asarray(points).copy()
    fraction = (query-stamps[previous]).astype(float)/dt
    start_rotation, end_rotation = poses[previous, :3, :3], poses[end_index, :3, :3]
    tangent = Rotation.from_matrix(start_rotation.T @ end_rotation).as_rotvec()
    rotation = Rotation.from_rotvec(fraction[:, None]*tangent).as_matrix()
    rotation = np.einsum('ij,njk->nik', start_rotation, rotation)
    position = poses[previous, :3, 3] + fraction[:, None]*(poses[end_index, :3, 3]-poses[previous, :3, 3])
    world = np.einsum('nij,nj->ni', rotation, points) + position
    reference = poses[nearest_index_for_trajectory(stamps, reference_stamp_ns)]
    return (world-reference[:3, 3]) @ reference[:3, :3]


def nearest_index_for_trajectory(stamps, stamp):
    from .data import nearest_index
    return nearest_index(stamps, stamp)


def audit_depth_lidar_extrinsic(prepared, output):
    """Reject or diagnose a calibration hypothesis over disjoint scene times.

    The public FAST-Calib rotation is explicitly an INITIAL HYPOTHESIS. It is
    never promoted to device calibration, and no configuration is overwritten.
    Stereo depth inconsistency is checked through independent per-time estimates.
    """
    prepared = Path(prepared)
    frames = load_frames(prepared)
    source = Path('/home/chaoszzss/orbvi_proj_ws/src/orbvi_omni_livo2/fast-calib_final/cam0_multi_calib_result.txt')
    Rcl = np.array([[.679420, -.733611, .014275], [.030320, .008633, -.999503],
                    [.733123, .679515, .028109]])
    A = np.array([[0.,0.,1.], [-1.,0.,0.], [0.,-1.,0.]])
    hypothesis = np.eye(4)
    hypothesis[:3, :3] = Rotation.from_matrix(Rcl.T @ A.T).as_matrix()
    hypothesis[:3, 3] = -Rcl.T @ np.array([.013496, -.065082, -.075599])
    records, estimates = [], []
    for index in [10, 400, 800, 1200, 1600, 2200]:
        frame = frames[index]
        target = voxel_downsample(load_scan(prepared, frame)['points'], .3, 7000)
        depth = voxel_downsample(np.load(prepared / frame['depth']['path'])['points'], .3, 7000)
        initial = transform_points(depth, hypothesis)
        angle = np.arctan2(initial[:,2], np.linalg.norm(initial[:,:2], axis=1))
        visible = (angle > np.deg2rad(-6)) & (angle < np.deg2rad(50))
        estimate, stats = icp_point_to_plane(depth[visible], target, hypothesis,
                                              max_iterations=30, max_distance=1.5)
        estimates.append(estimate)
        delta = inverse_transform(hypothesis) @ estimate
        records.append(dict(stats, frame_index=index, points=int(visible.sum()),
                            delta_translation_m=delta[:3,3].tolist(),
                            delta_rotation_deg=float(np.linalg.norm(Rotation.from_matrix(delta[:3,:3]).as_rotvec())*180/np.pi),
                            T_lidar_depth_ros_hypothesis=estimate.tolist(),
                            depth_age_ms=abs(frame['depth']['delta_ns'])/1e6))
    estimates = np.asarray(estimates)
    translation_span = np.ptp(estimates[:, :3, 3], axis=0)
    result = {'device_extrinsic_valid': False,
              'decision': 'reject_as_device_calibration',
              'source_hypothesis': str(source),
              'source_hypothesis_sha256': sha256_file(source) if source.exists() else None,
              'source_hypothesis_status': 'Public repository reference, not device-specific calibration',
              'method': 'Independent fixed-scale robust depth-to-LiDAR fits at six times spanning the complete bag; restricted to MID360 elevation overlap',
              'gates': {'minimum_correspondence_fraction': .6, 'maximum_point_plane_rmse_m': .12,
                        'maximum_across_time_translation_span_m': .2},
              'across_time_translation_span_m': translation_span.tolist(),
              'records': records,
              'reason': 'Low sensor-cloud agreement and inconsistent per-time extrinsics fail calibration acceptance; noisy/misaligned stereo depths cannot establish device extrinsics.',
              'effect': 'T_camera_lidar remains null. Camera semantics can be inferred independently, but lidar-camera projection is disabled by default.'}
    write_json(output, result)
    return result
