"""Full-recording camera semantic occupancy, with source-pixel audit artifacts.

The recording index is only a comparison clock. This runner never reads LiDAR
points, LiDAR depth or time-varying LiDAR/VIO poses. Both perception models are
frozen. Test-time Gaussian rasterization is the documented zero-iteration
TT-Occ adaptation, not the original optimized 3DGS/RAFT implementation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback
import zipfile

import cv2
import numpy as np
import psutil

from .camera_depth import MapAnythingDepth
from .camera_fusion import CameraSemanticFusion, geometry_collision_voxelize
from .camera_run import QualityCameraMotion, json_finite
from .camera_semantic import CameraSemanticProvider, build_virtual_views, lift_source_semantics
from .data import load_frames
from .evaluation import FullBagAcceptance
from .projection import load_camera_rig
from .run import PROJECT, resolve, load_images, write_json
from .taxonomy import load_registry, class_histogram
from .traversability import LocalTraversability
from .tt_backend import UPSTREAM_COMMIT
from .visualization import save_local_map, save_summary_plot


def atomic_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_suffix('.npz.tmp')
    # Actual-frame profiling showed ZIP level 1 halves serialization time with
    # about 2-6% larger files. Standard NPY members retain identical tensors.
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for name, value in arrays.items():
            with archive.open(name+'.npy', 'w', force_zip64=True) as member:
                np.lib.format.write_array(member, np.asarray(value), allow_pickle=False)
    os.replace(temporary, path)


def read_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def read_committed_journal(path):
    """A newline is the commit boundary; preserve a interrupted trailing write."""
    path = Path(path)
    payload = path.read_bytes()
    boundary = payload.rfind(b'\n') + 1
    rows = [json.loads(line) for line in payload[:boundary].splitlines() if line.strip()]
    if boundary < len(payload):
        tail = payload[boundary:]
        digest = hashlib.sha256(tail).hexdigest()[:16]
        path.with_name(f'uncommitted_journal_tail_{digest}.bin').write_bytes(tail)
        with path.open('r+b') as handle:
            handle.truncate(boundary)
    return rows


def link_artifact(source, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


def provenance(config):
    paths = [PROJECT / 'configs/calibration.json', PROJECT / 'taxonomy/garage.json']
    if config['virtual_views'].get('ego_mask_path'):
        paths.append(resolve(config['virtual_views']['ego_mask_path']))
    paths.extend(sorted((PROJECT / 'orbvi_occ').glob('*.py')))
    for component in ('depth', 'semantics'):
        model = resolve(config[component]['model_dir'])
        paths.extend(p for p in (model/'provenance.json', model/'config.json') if p.exists())
    if config['semantics'].get('ade_model_dir'):
        model = resolve(config['semantics']['ade_model_dir'])
        paths.extend(p for p in (model/'provenance.json',model/'config.json') if p.exists())
    if config['semantics'].get('provider') in ('openseed','hybrid'):
        paths.append(PROJECT/'scripts/openseed_resident_python.sh')
        paths.extend(p for p in (PROJECT/'models/openseed-source/source-provenance.json',
            PROJECT/'models/openseed-source/compatibility-patches.json', PROJECT/'models/clip-tokenizer/provenance.json') if p.exists())
    return {str(path.relative_to(PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def inference_key(config, manifest):
    files = ['configs/calibration.json', 'taxonomy/garage.json', 'orbvi_occ/camera_depth.py',
             'orbvi_occ/camera_semantic.py', 'orbvi_occ/projection.py', 'orbvi_occ/taxonomy.py']
    if config['semantics'].get('provider') in ('openseed','hybrid'):
        files.append('orbvi_occ/openseed_provider.py')
    if config['semantics'].get('provider') == 'hybrid':
        files.append('orbvi_occ/camera_semantic_hybrid.py')
    signatures = {p: hashlib.sha256((PROJECT/p).read_bytes()).hexdigest() for p in files if (PROJECT/p).exists()}
    if config['virtual_views'].get('ego_mask_path'):
        p = resolve(config['virtual_views']['ego_mask_path'])
        signatures[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    for component in ('depth', 'semantics'):
        p = resolve(config[component]['model_dir'])/'provenance.json'
        if p.exists():
            signatures[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    if config['semantics'].get('ade_model_dir'):
        p=resolve(config['semantics']['ade_model_dir'])/'provenance.json'
        signatures[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    value = {'bag': manifest['source_bag_sha256'], 'depth': config['depth'],
             'semantics': config['semantics'], 'virtual_views': config['virtual_views'], 'sources': signatures}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest(), value


class FrameInference:
    """Shared exact-configuration cache for pilot, full run and causal replay."""
    def __init__(self, config, prepared, manifest, registry):
        self.config, self.prepared, self.registry = config, prepared, registry
        self.key, key_data = inference_key(config, manifest)
        self.cache = resolve(config.get('inference_cache', 'cache/camera_semantic'))/self.key
        for sub in ('source_points', 'semantic_maps', 'semantic_images', 'views', 'committed'):
            (self.cache/sub).mkdir(parents=True, exist_ok=True)
        write_json(self.cache/'cache.json', key_data)
        self.rig = load_camera_rig(PROJECT/'configs/calibration.json')
        self.depth = self.semantics = None
        self.metadata = {}

    def _models(self):
        if self.depth is None:
            self.depth = MapAnythingDepth(**{**self.config['depth'],
                                           'model_dir': str(resolve(self.config['depth']['model_dir']))})
            # Keep the existing mixed-precision tensors unchanged. Depth and
            # image semantics run in separate GPU phases on the 8 GB host.
            if self.depth.device.startswith('cuda'):
                self.depth.model.to('cpu')
                import torch
                torch.cuda.empty_cache()
            sem = {k:v for k,v in self.config['semantics'].items() if k != 'enabled'}
            sem['model_dir'] = str(resolve(sem['model_dir']))
            if sem.get('ade_model_dir'):
                sem['ade_model_dir'] = str(resolve(sem['ade_model_dir']))
            if sem.get('provider') in ('openseed', 'hybrid'):
                sem.setdefault('python', str(PROJECT/'scripts/openseed_resident_python.sh'))
            if sem.get('provider') == 'hybrid':
                from .camera_semantic_hybrid import HybridCameraSemanticProvider
                self.semantics = HybridCameraSemanticProvider(**sem, registry=self.registry)
            elif sem.get('provider') == 'openseed':
                from .openseed_provider import OpenSeeDSubprocessProvider
                self.semantics = OpenSeeDSubprocessProvider(**sem, registry=self.registry)
            else:
                self.semantics = CameraSemanticProvider(**sem, registry=self.registry)
            self.metadata = {'camera_depth_provider': self.depth.metadata(),
                             'semantics': self.semantics.metadata()}
            write_json(self.cache/'model_runtime.json', json_finite(self.metadata))

    def _perceive(self, views):
        """Move frozen tensors between phases without converting their dtypes."""
        import torch
        ade = getattr(self.semantics, 'ade', self.semantics)
        ade_model = getattr(ade, 'model', None)
        depth_cuda = self.depth.device.startswith('cuda')
        ade_cuda = str(getattr(ade, 'device', '')).startswith('cuda')
        if depth_cuda:
            self.depth.model.to('cpu')
            torch.cuda.empty_cache()
        if ade_model is not None and ade_cuda:
            ade_model.to(ade.device)
        semantic_maps = self.semantics.predict_views(views)
        if ade_model is not None and ade_cuda:
            ade_model.to('cpu')
            torch.cuda.empty_cache()
        if depth_cuda:
            self.depth.model.to(self.depth.device)
        try:
            prediction = self.depth.predict(views)
        finally:
            if depth_cuda:
                self.depth.model.to('cpu')
                torch.cuda.empty_cache()
        return semantic_maps, prediction

    def get(self, frame, output):
        index, stamp = frame['index'], frame['stamp_ns']
        commit = self.cache/'committed'/f'{index:06d}.json'
        cached = commit.exists()
        if cached:
            record = json.loads(commit.read_text())
            if record['stamp_ns'] != stamp or record['cache_key'] != self.key:
                raise RuntimeError('Inference cache provenance mismatch')
            if not self.metadata:
                self.metadata = json.loads((self.cache/'model_runtime.json').read_text())
            source = read_npz(self.cache/'source_points'/f'{index:06d}.npz')
            views = json.loads((self.cache/'views'/f'{index:06d}.json').read_text())
        else:
            self._models()
            images = load_images(self.prepared, frame)
            if len(images) != 4:
                raise RuntimeError(f'All four synchronized causal cameras required at frame {index}')
            actual_views = build_virtual_views(images, self.rig, self.config['virtual_views'])
            semantic_maps, prediction = self._perceive(actual_views)
            lifted = lift_source_semantics(prediction, semantic_maps, registry=self.registry)
            source = {k: np.asarray(prediction[k]) for k in (
                'points', 'origins', 'colors', 'camera_ids', 'source_uv', 'source_view_ids',
                'source_image_shapes', 'source_depth', 'source_confidence', 'source_camera_points',
                'source_poses', 'source_intrinsics', 'source_view_names', 'source_validity_sampling')}
            source.update(labels=lifted['labels'], semantic_confidence=lifted['semantic_confidence'],
                          evidence=lifted['evidence'], semantic_uv=lifted['semantic_uv'],
                          depth_confidence=prediction['source_confidence'], stamp_ns=np.int64(stamp),
                          source_valid_masks=np.asarray(prediction['valid_masks']),
                          depth_maps=np.asarray(prediction['depth_maps']),
                          depth_confidence_maps=np.asarray(prediction['confidence_maps']),
                          view_ids=np.asarray([v['view_id'] for v in actual_views]),
                          source_stamp_ns=np.asarray([frame['images'][v['camera_id']]['stamp_ns'] for v in actual_views], np.int64),
                          near_range_m=np.linalg.norm(prediction['points']-prediction['origins'], axis=1).astype(np.float32),
                          pixel_domain=np.asarray('rectified_virtual_pinhole'))
            if all('expert_id' in sem for sem in semantic_maps):
                experts = np.full(len(source['points']), 255, np.uint8)
                for vid, sem in enumerate(semantic_maps):
                    selected = np.flatnonzero(source['source_view_ids'] == vid)
                    uv = source['semantic_uv'][selected]
                    experts[selected] = sem['expert_id'][uv[:,1],uv[:,0]]
                source['semantic_expert_id'] = experts
            views = []
            for view, sem in zip(actual_views, semantic_maps):
                name = f"{index:06d}_{view['view_id']}"
                image_path, maps_path = f'semantic_images/{name}.jpg', f'semantic_maps/{name}.npz'
                if not cv2.imwrite(str(self.cache/image_path), cv2.cvtColor(view['image'], cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise RuntimeError('Failed writing actual rectified source RGB')
                sem_arrays = {k: np.asarray(v) for k,v in sem.items()}
                sem_arrays.update(source_stamp_ns=np.int64(frame['images'][view['camera_id']]['stamp_ns']),
                                  K=np.asarray(view['K']), T_rig_camera=np.asarray(view['T_rig_camera']))
                if 'ego_mask' in view:
                    sem_arrays['ego_mask'] = np.asarray(view['ego_mask'], bool)
                atomic_npz(self.cache/maps_path, **sem_arrays)
                views.append({'view_id': view['view_id'], 'physical_camera_id': view['physical_camera_id'],
                    'image_path': image_path, 'semantic_image_path': image_path, 'semantic_maps_path': maps_path,
                    'K': np.asarray(view['K']).tolist(), 'T_rig_camera': np.asarray(view['T_rig_camera']).tolist(),
                    'source_stamp_ns': frame['images'][view['camera_id']]['stamp_ns'],
                    'source_image_path': str(self.prepared/frame['images'][view['camera_id']]['path']),
                    'pixel_domain': 'rectified_virtual_pinhole', 'image_shape': list(view['image'].shape[:2]),
                    'virtual_yaw_deg': view['virtual_yaw_deg'], 'virtual_pitch_deg': view['virtual_pitch_deg']})
                if 'ego_mask_metadata' in view:
                    views[-1]['ego_mask_metadata'] = view['ego_mask_metadata']
                if 'ego_mask_scope' in view:
                    views[-1]['ego_mask_scope'] = view['ego_mask_scope']
            source_path = self.cache/'source_points'/f'{index:06d}.npz'
            atomic_npz(source_path, **source)
            write_json(self.cache/'views'/f'{index:06d}.json', views)
            record = {'index': index, 'stamp_ns': stamp, 'cache_key': self.key,
                'depth': prediction['diagnostics'], 'semantics': self.semantics.last_info,
                'semantic_lifting': lifted['diagnostics']}
            write_json(commit, json_finite(record))
        for relative in [f'source_points/{index:06d}.npz', f'views/{index:06d}.json',
                         *[v[key] for v in views for key in ('image_path', 'semantic_maps_path')]]:
            link_artifact(self.cache/relative, output/relative)
        return source, record, cached

    def close(self):
        if self.semantics is not None and hasattr(self.semantics, 'close'):
            self.semantics.close()


def run(config_path, output_override=None, limit=None, resume=False):
    import torch
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    config = json.loads(Path(config_path).read_text())
    if config.get('route') != 'camera':
        raise ValueError('Dedicated camera semantic runner accepts camera route only')
    prepared, output = resolve(config['prepared']), resolve(output_override or config['output'])
    manifest = json.loads((prepared/'manifest.json').read_text())
    all_frames = load_frames(prepared)
    if not manifest['complete'] or [f['index'] for f in all_frames] != list(range(len(all_frames))):
        raise RuntimeError('Source extraction is not a complete contiguous recording')
    expected = len(all_frames)
    frames = all_frames if limit is None else all_frames[:limit]
    if not frames:
        raise ValueError('The requested run has no acquisitions')
    registry = load_registry()
    sources = provenance(config)
    fingerprint = hashlib.sha256(json.dumps({'config': config, 'sources': sources}, sort_keys=True).encode()).hexdigest()
    metadata = {'route': 'camera', 'pipeline': 'camera_semantic_v1', 'config': config,
        'config_sha256': fingerprint, 'input_provenance_sha256': sources,
        'bag_sha256': manifest['source_bag_sha256'], 'upstream_commit': UPSTREAM_COMMIT,
        'semantic_registry': registry, 'expected_full_recording_frames': expected,
        'limited_development_run': limit is not None, 'sensor_timestamp_basis': manifest['timestamp_basis'],
        'pose_source': 'camera_depth_only', 'runtime_lidar_input': False, 'runtime_vio_input': False,
        'device_cross_sensor_calibration_valid': False,
        'camera_reference_frame': 'camera_3 optical origin, x forward/y left/z up by fixed optical-to-FLU rotation',
        'recording_sampling': {'reference': 'existing 10 Hz acquisition comparison clock; timestamps only',
            'camera_source_rate_hz': 20, 'selected_views_per_time_step': 4,
            'scope': 'all aligned time steps across the complete recording; unselected 20 Hz images remain in prepared'},
        'method_scope': 'Frozen metric depth and frozen image semantics, exact source UV lifting, causal semantic Gaussian occupancy; zero optimization iterations',
        'collision_voxelizer': 'orbvi_occ.camera_fusion.geometry_collision_voxelize; same camera occupancy grid indexing for non-floor sources',
        'runtime_memory_schedule': 'OpenSeeD, ADE and depth use separate GPU inference phases; registered model tensors reside on CPU between phases without dtype conversion. OpenSeeD keeps its isolated process and unchanged small text/evidence caches, but parks its weights on CPU after prediction. Models, input pixels and inference arithmetic are unchanged. Cached source artifacts retain their original recorded runtime metadata.',
        'openseed_worker_entry': 'scripts/openseed_resident_python.sh -> orbvi_occ.openseed_residency, wrapping the unchanged pinned OpenSeeDWorker inference',
        'ego_body_mask': config['virtual_views'].get('ego_mask_path'),
        'presentation': {'color_mode': 'semantic', 'theme': 'official', 'default_garage_view': 'ceiling_cut',
            'scope': 'Display-only ceiling class hiding; all height occupied geometry and traversal evidence are saved unchanged'},
        'official_method_differences': ['garage taxonomy extends official street taxonomy',
            'zero-iteration Gaussian adapter; original 3DGS test-time optimization is not executed',
            'camera depth ICP with rejection gates; original RAFT/instance tracking is not executed'],
        'footprint_scope': 'demonstration dimensions, not a supplied measured vehicle envelope',
        'artifact_compression': 'standard NPZ/NPY, lossless ZIP DEFLATE level 1',
        'software': {'python': os.sys.version.split()[0], 'torch': torch.__version__, 'numpy': np.__version__, 'opencv': cv2.__version__}}
    if config['semantics'].get('provider') == 'hybrid':
        metadata['official_method_differences'].append(
            'Frozen ADE20K expert fills only panoptic semantic void and refines wall to column under preset source confidence gates')
    if (output/'run.json').exists():
        previous = json.loads((output/'run.json').read_text())
        if not resume:
            raise RuntimeError('Output already exists; choose a new output or --resume')
        if previous['config_sha256'] != fingerprint or previous['bag_sha256'] != metadata['bag_sha256']:
            raise RuntimeError('Resume configuration/source fingerprint mismatch')
    output.mkdir(parents=True, exist_ok=True)
    for folder in ('occupancy', 'traversability', 'source_points', 'views', 'semantic_images', 'semantic_maps', 'visualizations', 'source_snapshot'):
        (output/folder).mkdir(exist_ok=True)
    for relative in sources:
        target = output/'source_snapshot'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(PROJECT/relative, target)
    write_json(output/'semantic_registry.json', registry)
    previous_rows = []
    if resume and (output/'frames.jsonl').exists():
        previous_rows = read_committed_journal(output/'frames.jsonl')
        if [r['index'] for r in previous_rows] != list(range(len(previous_rows))):
            raise RuntimeError('Resume requires a contiguous committed frame prefix')
        for row in previous_rows:
            for sub in ('occupancy', 'traversability', 'source_points', 'views'):
                suffix = '.json' if sub == 'views' else '.npz'
                if not (output/sub/f"{row['index']:06d}{suffix}").exists():
                    raise RuntimeError('Committed resume artifact missing')
    restart = len(previous_rows)
    if restart > len(frames):
        raise RuntimeError('Resume limit is shorter than the already committed recording prefix')
    inference = FrameInference(config, prepared, manifest, registry)
    metadata['inference_cache_key'] = inference.key
    write_json(output/'run.json', metadata)
    fusion = CameraSemanticFusion(config['ttocc'], registry)
    local_builder = LocalTraversability(config['traversability'],
        collision_voxelizer=geometry_collision_voxelize)
    motion = QualityCameraMotion()
    acceptance = FullBagAcceptance(expected, all_frames[0]['stamp_ns'], all_frames[-1]['stamp_ns'])
    for row in previous_rows:
        acceptance.add(row['index'], row['stamp_ns'], True, row['raw_source_points'], row['fused_occupied_voxels'], row['traversability_cells'])
    rows = previous_rows.copy()
    started = time.perf_counter()
    peak_ram = psutil.Process().memory_info().rss
    log = (output/'frames.jsonl').open('a' if resume else 'w', buffering=1)
    try:
        for frame in frames:
            index, tick = frame['index'], time.perf_counter()
            source, record, cached = inference.get(frame, output)
            if 'semantics' not in metadata:
                metadata.update(inference.metadata)
                write_json(output/'run.json', json_finite(metadata))
            pose, motion_info = motion.update(source['points'])
            if not motion_info['accepted']:
                fusion.reset()
                local_builder = LocalTraversability(config['traversability'],
                    collision_voxelizer=geometry_collision_voxelize)
            result = fusion.update(source['points'], pose, frame['stamp_ns'], evidence=source['evidence'],
                semantic_confidence=source['semantic_confidence'], origins=source['origins'], depth_confidence=source['depth_confidence'])
            local = local_builder.update(source['points'], result['fused_centers'], pose=pose,
                timestamp=frame['stamp_ns']/1e9, ray_origins=source['origins'],
                fused_source_points=result['fused_source_points'], free_evidence_source='predicted_camera_depth')
            if index < restart:
                continue
            occupancy = {k:v for k,v in result.items() if k.startswith(('current_', 'fused_')) and isinstance(v, np.ndarray) and 'source_points' not in k}
            occupancy.update(stamp_ns=np.int64(frame['stamp_ns']), T_world_sensor=pose,
                bounds=fusion.bounds, voxel_size=fusion.config.voxel_size,
                segment_id=np.int32(motion_info['segment_id']),
                pixel_domain=np.array('rectified_virtual_pinhole'),
                semantic_registry=np.array(registry['name']),
                free_space_semantics=np.array('unobserved cells remain unknown; camera depth rays are predictions'))
            atomic_npz(output/'occupancy'/f'{index:06d}.npz', **occupancy)
            local.save(str(output/'traversability'/f'{index:06d}.npz'))
            point_labels = source['labels']
            beyond_ego = source['near_range_m'] > 2
            row = {'index': index, 'stamp_ns': frame['stamp_ns'], 'timestamp': frame['stamp_ns']/1e9,
                'raw_source_points': frame['point_count'], 'route_input_points': len(source['points']),
                'current_occupied_voxels': len(result['current_indices']), 'fused_occupied_voxels': len(result['fused_indices']),
                'new_voxels_from_history': result['diagnostics']['new_voxels_from_history'],
                'traversability_cells': int(local.traversability.size), 'ground': local.summary['ground'],
                'class_area_m2': local.summary['class_area_m2'],
                'current_semantic_histogram': class_histogram(result['current_labels'], registry),
                'fused_semantic_histogram': class_histogram(result['fused_labels'], registry),
                'source_semantic_histogram': class_histogram(point_labels, registry),
                'source_semantics_beyond_2m': class_histogram(point_labels[beyond_ego], registry),
                'source_semantics_within_2m': class_histogram(point_labels[~beyond_ego], registry),
                'processing_seconds': time.perf_counter()-tick, 'inference_cache_hit': cached,
                'algorithm': result['diagnostics'],
                'auxiliary': {'camera_motion': motion_info, 'camera_depth': record['depth'],
                    'camera_semantics': record['semantics'], 'semantic_lifting': record['semantic_lifting'],
                    'camera_maps': len(source['source_image_shapes']), 'runtime_lidar_input': False},
                'rss_mib': psutil.Process().memory_info().rss/2**20}
            log.write(json.dumps(json_finite(row), ensure_ascii=False, allow_nan=False)+'\n')
            rows.append(row)
            acceptance.add(index, frame['stamp_ns'], True, frame['point_count'], len(result['fused_indices']), local.traversability.size)
            peak_ram = max(peak_ram, psutil.Process().memory_info().rss)
            if index % config.get('evaluation', {}).get('visualization_stride', 100) == 0 or index == len(frames)-1:
                save_local_map(local, output/'visualizations'/f'{index:06d}.png',
                    title=f'Camera semantic | acquisition {index}/{expected-1}',
                    source_npz=output/'traversability'/f'{index:06d}.npz')
            if index % 25 == 0 or index == len(frames)-1:
                write_json(output/'progress.json', {'processed_frames': len(rows), 'expected_frames': expected,
                    'last_frame': index, 'elapsed_seconds': time.perf_counter()-started, 'route': 'camera', 'complete': False})
                print(f'camera semantic: {len(rows)}/{expected} frame={index} points={len(source["points"])} '
                      f'voxels={len(result["fused_indices"])} {row["processing_seconds"]:.2f}s cache={cached}', flush=True)
    except Exception as error:
        write_json(output/'failure.json', {'frame': frame['index'] if 'frame' in locals() else None,
            'error': str(error), 'traceback': traceback.format_exc(), 'complete': False})
        raise
    finally:
        log.close()
        inference.close()
    poses, segments, registered = [], [], []
    for row in rows:
        data = read_npz(output/'occupancy'/f"{row['index']:06d}.npz")
        poses.append(data['T_world_sensor'])
        segments.append(row['auxiliary']['camera_motion']['segment_id'])
        registered.append(row['auxiliary']['camera_motion']['accepted'])
    atomic_npz(output/'trajectory.npz', T_world_camera=np.asarray(poses),
        stamp_ns=np.asarray([r['stamp_ns'] for r in rows], np.int64),
        segment_id=np.asarray(segments, np.int32), registration_accepted=np.asarray(registered, bool))
    times = np.asarray([r['processing_seconds'] for r in rows])
    aggregate = {prefix: Counter() for prefix in ('current', 'fused', 'source')}
    beyond_counts = Counter()
    for row in rows:
        for prefix in aggregate:
            aggregate[prefix].update(row[prefix+'_semantic_histogram'])
        beyond_counts.update(row['source_semantics_beyond_2m'])
    segment_count = max(segments)+1 if segments else 0
    summary = {'complete': acceptance.summary()['accepted'] and limit is None,
        'acceptance': acceptance.summary(), 'route': 'camera', 'processed_frames': len(rows), 'expected_frames': expected,
        'camera_selected_view_count': sum(r['auxiliary']['camera_maps'] for r in rows),
        'raw_camera_images_in_bag': sum(manifest['counts'].get(f'/orbvi/raw/camera_{cid}/image/compressed', 0) for cid in ('2','3','4','5')),
        'camera_source_point_count': sum(r['route_input_points'] for r in rows),
        'camera_segments': segment_count, 'camera_history_resets': sum(not a for a in registered),
        'global_pose_continuity_valid': segment_count <= 1, 'runtime_lidar_input': False, 'runtime_vio_input': False,
        'semantic_registry': registry['name'], 'semantic_histograms': {k:dict(v) for k,v in aggregate.items()},
        'source_semantics_beyond_2m': dict(beyond_counts),
        'latency_seconds': {'median': float(np.median(times)), 'p95': float(np.percentile(times, 95)), 'max': float(times.max())},
        'elapsed_seconds_this_invocation': time.perf_counter()-started, 'sum_frame_processing_seconds': float(times.sum()),
        'peak_process_rss_mib': peak_ram/2**20, 'frames_with_ground': sum(bool(r['ground'].get('valid')) for r in rows),
        'frames_with_partial_clearance': sum(r['class_area_m2'].get('partial_clearance',0)>0 for r in rows),
        'semantics': metadata.get('semantics'), 'camera_depth_provider': metadata.get('camera_depth_provider'),
        'semantic_miou': None, 'occupancy_ground_truth_iou': None,
        'limitations': ['No labelled garage ground truth; class counts and model scores do not establish semantic accuracy.',
            'True garage surfaces, occlusion, lighting and field of view differ from official street examples.',
            'Near-range predicted car includes possible ego body; a >2m diagnostic is not an instance annotation.',
            'Original TT-Occ test-time 3DGS optimization and RAFT instance tracking are not executed in this adapter.',
            'Camera depth free-space evidence cannot certify measured full clearance.',
            'Camera ICP consistency is not independent metric pose accuracy; rejected registrations start separate local segments.']}
    write_json(output/'summary.json', json_finite(summary))
    write_json(output/'progress.json', {'processed_frames': len(rows), 'expected_frames': expected,
        'last_frame': rows[-1]['index'], 'route': 'camera', 'complete': summary['complete']})
    failure = output/'failure.json'
    if failure.exists():
        failure.unlink()
    save_summary_plot(rows, output/'timeline.png')
    print(json.dumps({'complete': summary['complete'], 'processed': len(rows), 'expected': expected, 'output':str(output)}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(PROJECT/'configs/camera_semantic.json'))
    parser.add_argument('--output')
    parser.add_argument('--limit', type=int, help='development prefix only; never a full-recording result')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    summary = run(args.config, args.output, args.limit, args.resume)
    if args.limit is None and not summary['complete']:
        raise SystemExit('Full camera recording acceptance failed')


if __name__ == '__main__':
    main()
