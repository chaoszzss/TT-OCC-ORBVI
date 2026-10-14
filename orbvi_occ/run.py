"""Process every indexed acquisition with auditable per-frame artifacts."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import cv2
import numpy as np
import psutil

from .data import load_frames, load_scan
from .projection import PinholeRectifier, load_camera_rig, transform_points, label_lidar_points
from .tt_backend import TTBackend, gaussian_voxelize, UPSTREAM_COMMIT
from .traversability import LocalTraversability
from .evaluation import FullBagAcceptance, evaluate_geometry, split_holdout, GeometryMetricAccumulator
from .visualization import save_local_map, save_summary_plot, save_geometry_xy

PROJECT = Path(__file__).resolve().parents[1]
OPTICAL_TO_FORWARD_LEFT_UP = np.array([[0,0,1,0],[-1,0,0,0],[0,-1,0,0],[0,0,0,1.]],dtype=float)


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))
    os.replace(temp,path)


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else PROJECT/p


def load_images(prepared, frame):
    images={}
    for camid,item in frame['images'].items():
        if item is None or not item.get('valid_sync',False):
            continue
        if item['stamp_ns'] > frame['stamp_ns']:
            raise RuntimeError('Future camera acquisition entered inference')
        bgr = cv2.imread(str(prepared/item['path']))
        if bgr is None:
            raise RuntimeError(f"Unreadable camera acquisition {item['path']}")
        images[camid] = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    return images


class CameraMotion:
    """Camera-only depth-cloud ICP; no LiDAR/VIO poses or depths are inputs."""
    def __init__(self):
        self.pose=np.eye(4)
        self.previous=None
        self.relative=np.eye(4)
        self.records=[]

    def update(self,points):
        from .calibration import voxel_downsample,icp_point_to_plane
        from scipy.spatial.transform import Rotation
        current=voxel_downsample(points,.3,6000)
        if self.previous is None:
            info={'accepted':True,'initial_frame':True,'pose_source':'camera_depth_only'}
        else:
            relative,info=icp_point_to_plane(current,self.previous,self.relative,max_iterations=20,max_distance=1.0)
            rotation=float(np.linalg.norm(Rotation.from_matrix(relative[:3,:3]).as_rotvec()))
            accepted=(info['rank']==6 and info['correspondence_fraction']>.25 and
                      info['point_plane_rmse_m']<.35 and np.linalg.norm(relative[:3,3])<1.0 and rotation<.25)
            if not accepted:
                # An unreliable temporal alignment cannot be silently used for
                # fusion. Caller clears the historical occupancy in this case.
                info.update(accepted=False,pose_source='camera_depth_only',history_reset=True)
                self.relative=np.eye(4)
            else:
                self.pose=self.pose@relative
                self.relative=relative
                info.update(accepted=True,pose_source='camera_depth_only')
        self.previous=current
        self.records.append(info)
        return self.pose.copy(),info


def run(config_path,limit=None,resume=False,output_override=None):
    import torch
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    config=json.loads(Path(config_path).read_text())
    route=config['route']
    if route not in ('lidar','camera'):
        raise ValueError('route must be lidar or camera')
    prepared=resolve(config['prepared'])
    output=resolve(output_override or config['output'])
    manifest=json.loads((prepared/'manifest.json').read_text())
    frames=load_frames(prepared)
    if not manifest['complete'] or len(frames)!=manifest['counts']['/orbvi/lidar/custom']:
        raise RuntimeError('Incomplete original-bag extraction')
    if [f['index'] for f in frames]!=list(range(len(frames))):
        raise RuntimeError('Noncontiguous recording index')
    total_expected=len(frames)
    if limit is not None:
        frames=frames[:limit]
    camera_cfg=load_camera_rig(PROJECT/'configs/calibration.json')
    provenance_files=[PROJECT/'configs/calibration.json']
    if route=='lidar':
        provenance_files.append(prepared/'trajectory.npz')
    for component in ('semantics','depth'):
        if component in config and 'model_dir' in config[component]:
            model_provenance=resolve(config[component]['model_dir'])/'provenance.json'
            if model_provenance.exists():
                provenance_files.append(model_provenance)
    provenance_files.extend(sorted((PROJECT/'orbvi_occ').glob('*.py')))
    provenance={str(p.relative_to(PROJECT)) if p.is_relative_to(PROJECT) else str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in provenance_files if p.exists()}
    fingerprint=hashlib.sha256(json.dumps({'config':config,'inputs':provenance},sort_keys=True).encode()).hexdigest()
    metadata={'route':route,'config':config,'config_sha256':fingerprint,
              'bag_sha256':manifest['source_bag_sha256'],'upstream_commit':UPSTREAM_COMMIT,'input_provenance_sha256':provenance,
              'expected_full_recording_frames':total_expected,'limited_development_run':limit is not None,
              'sensor_timestamp_basis':manifest['timestamp_basis'],
              'device_cross_sensor_calibration_valid':False,
              'footprint_scope':'configurable demonstration dimensions; actual vehicle dimensions were not supplied',
              'software':{'python':os.sys.version.split()[0],'torch':torch.__version__,'numpy':np.__version__,'opencv':cv2.__version__}}
    if (output/'run.json').exists():
        old=json.loads((output/'run.json').read_text())
        if not resume:
            raise RuntimeError('Output already exists; choose a new output directory or --resume')
        if old['config_sha256']!=fingerprint or old['bag_sha256']!=metadata['bag_sha256']:
            raise RuntimeError('Resume fingerprint mismatch; old outputs cannot be reused')
    output.mkdir(parents=True,exist_ok=True)
    for folder in ['occupancy','traversability','visualizations','semantic_maps']:
        (output/folder).mkdir(exist_ok=True)
    logpath=output/'frames.jsonl'
    previous_rows=[]
    if resume and logpath.exists():
        previous_rows=[json.loads(line) for line in logpath.read_text().splitlines() if line.strip()]
        if [r['index'] for r in previous_rows]!=list(range(len(previous_rows))):
            raise RuntimeError('Resume requires a contiguous successful output prefix')
        for row in previous_rows:
            for sub in ['occupancy','traversability']:
                if not (output/sub/f"{row['index']:06d}.npz").exists():
                    raise RuntimeError('Resume artifact missing')
    restart=len(previous_rows)
    # Recompute warm-up frames from original source to reconstruct causal state.
    # Camera poses require prefix replay; local LiDAR histories are bounded.
    warm_start=0 if route=='camera' else max(0,restart-config['ttocc']['max_history_frames']-4)
    backend=TTBackend(config['ttocc'])
    traversability=LocalTraversability(config['traversability'])
    semantics=None
    if config.get('semantics',{}).get('enabled'):
        from .semantics import SegformerSemantics
        sem_cfg=config['semantics']
        semantics=SegformerSemantics(resolve(sem_cfg['model_dir']),sem_cfg.get('device','auto'))
        metadata['semantics']=semantics.metadata()
    camera_depth=None
    if route=='lidar':
        if not (prepared/'trajectory.npz').exists():
            raise RuntimeError('Complete LiDAR-only trajectory is required; run prepare --trajectory')
        trajectory=dict(np.load(prepared/'trajectory.npz',allow_pickle=False))
        if len(trajectory['stamp_ns'])!=total_expected:
            raise RuntimeError('Trajectory does not cover all original scans')
        if not np.array_equal(trajectory['stamp_ns'],np.array([f['stamp_ns'] for f in load_frames(prepared)])):
            raise RuntimeError('Trajectory acquisition index mismatch')
        metadata['pose_source']=json.loads((prepared/'trajectory_report.json').read_text())['method']
        metadata['deskew']='past/current trajectory constant-velocity scan-end approximation'
    else:
        from .camera_depth import MapAnythingDepth
        camera_depth=MapAnythingDepth(**{**config['depth'],'model_dir':str(resolve(config['depth']['model_dir']))})
        motion=CameraMotion()
        rectifiers={cid:PinholeRectifier(c['params'],source_size=(c['width'],c['height'])) for cid,c in camera_cfg['cameras'].items()}
        metadata['pose_source']='ICP of predicted camera depth only; no LiDAR/VIO pose input'
        metadata['camera_reference_frame']='camera_3 optical origin, x optical-forward, y optical-left, z optical-up'
        metadata['camera_scale_source']='official frozen metric MapAnything checkpoint plus device camera rig calibration; no LiDAR depth alignment'
    write_json(output/'run.json',metadata)
    all_frames=load_frames(prepared)
    acceptance=FullBagAcceptance(total_expected,all_frames[0]['stamp_ns'],all_frames[-1]['stamp_ns'])
    for row in previous_rows:
        acceptance.add(row['index'],row['stamp_ns'],True,row['raw_source_points'],row['fused_occupied_voxels'],row['traversability_cells'])
    metric_acc=GeometryMetricAccumulator()
    rows=previous_rows.copy()
    world_display=[]
    poses=[]
    metric_path=output/'heldout_metrics.jsonl'
    metrics=[]
    if resume and metric_path.exists():
        metrics=[json.loads(line) for line in metric_path.read_text().splitlines() if line.strip()]
        metrics=[m for m in metrics if m['index']<restart]
        metric_ids=[m['index'] for m in metrics]
        if len(metric_ids)!=len(set(metric_ids)):
            raise RuntimeError('Resume metric journal contains duplicate committed frame IDs')
        for metric in metrics:
            metric_acc.add(metric)
    started=time.perf_counter()
    peak_ram=psutil.Process().memory_info().rss
    log=logpath.open('a' if resume else 'w',buffering=1)
    # The metric can be persisted just before a frame journal write is
    # interrupted. Remove that uncommitted suffix before replaying the frame.
    metric_log=metric_path.open('w',buffering=1)
    for metric in metrics:
        metric_log.write(json.dumps(metric,allow_nan=False)+'\n')
    try:
        for frame in frames[warm_start:]:
            index=frame['index']; tick=time.perf_counter()
            scan=load_scan(prepared,frame)
            raw_points=np.asarray(scan['points'])
            valid_returns=np.isfinite(raw_points).all(1)&(np.linalg.norm(raw_points,axis=1)>=.2)
            measured_points=raw_points[valid_returns]
            measured_offsets=scan['offset_time_ns'][valid_returns]
            image_maps={}; sem_time=0.
            images=load_images(prepared,frame) if semantics is not None or route=='camera' else {}
            for cid,image in images.items():
                cache=output/'semantic_maps'/f"{index:06d}_{cid}.npz"
                if semantics is None:
                    break
                if cache.exists():
                    with np.load(cache,allow_pickle=False) as cached:
                        image_maps[cid]=cached['labels']
                else:
                    a=time.perf_counter(); labels=semantics.predict(image); sem_time+=time.perf_counter()-a
                    np.savez_compressed(cache,labels=labels,source_stamp_ns=frame['images'][cid]['stamp_ns'])
                    image_maps[cid]=labels
            dynamic=None
            if route=='lidar':
                from .calibration import deskew_scan
                points=deskew_scan(measured_points,measured_offsets,frame['scan_start_ns'],frame['stamp_ns'],trajectory)
                pose=trajectory['T_world_lidar'][index]
                # Cross-sensor extrinsics are deliberately not manufactured.
                labels=np.full(len(points),255,np.uint8)
                aux={'semantic_lidar_association_valid':False,'camera_maps':len(image_maps)}
                origins=deskew_scan(np.zeros_like(measured_points),measured_offsets,frame['scan_start_ns'],frame['stamp_ns'],trajectory)
                evidence='lidar_measurement'
            else:
                if len(images)!=4:
                    raise RuntimeError(f'Camera route requires all four causal synchronized images at frame {index}')
                views=[]
                for cid in camera_cfg['camera_order']:
                    rect=rectifiers[cid]
                    views.append({'camera_id':cid,'image':rect.rectify(images[cid]),'K':rect.K,
                                  'T_rig_camera':OPTICAL_TO_FORWARD_LEFT_UP@np.asarray(camera_cfg['cameras'][cid]['T_camera0_camera']),
                                  'valid_mask':rect.valid})
                predicted=camera_depth.predict(views)
                points=predicted['points'];origins=predicted['origins']
                labels,_=label_lidar_points(points,camera_cfg,image_maps,np.linalg.inv(OPTICAL_TO_FORWARD_LEFT_UP),validated=True)
                if semantics is not None:
                    dynamic=np.isin(labels,semantics.dynamic_class_ids)
                pose,motion_info=motion.update(points)
                if not motion_info['accepted']:
                    backend.reset()
                    traversability=LocalTraversability(config['traversability'])
                aux={'camera_depth':predicted['diagnostics'],'camera_motion':motion_info,'camera_maps':len(image_maps)}
                evidence='predicted_camera_depth'
            result=backend.update(points,pose,frame['stamp_ns'],labels,dynamic)
            local=traversability.update(points,result['fused_centers'],pose=pose,timestamp=frame['stamp_ns']/1e9,
                                       ray_origins=origins,fused_source_points=result['fused_source_points'],
                                       free_evidence_source=evidence)
            if index<restart:
                continue
            occ_path=output/'occupancy'/f'{index:06d}.npz'
            map_path=output/'traversability'/f'{index:06d}.npz'
            np.savez_compressed(occ_path,stamp_ns=frame['stamp_ns'],T_world_sensor=pose,
                current_indices=result['current_indices'],current_labels=result['current_labels'],
                fused_indices=result['fused_indices'],fused_labels=result['fused_labels'],
                bounds=backend.bounds,voxel_size=backend.config.voxel_size,
                free_space_semantics=np.array('absent cells are unknown unless ray evidence exists'))
            local.save(str(map_path))
            ev_cfg=config.get('evaluation',{})
            # This shadow check removes whole held-out source voxels before
            # Gaussian construction, floor estimation and free-ray inference.
            if index%ev_cfg.get('holdout_stride',50)==0 or index==len(frames)-1:
                if route=='lidar':
                    construction,held,ids,hids=split_holdout(measured_points,ev_cfg.get('holdout_fraction',.1),seed=42+index,voxel_size=.3)
                    # Undeskewed scan-local validation avoids any pose-fit input.
                    shadow=TTBackend({**config['ttocc'],'history_seconds':0,'max_history_frames':0})
                    pred=shadow.update(construction,np.eye(4),frame['stamp_ns'])
                    safe=LocalTraversability({**config['traversability'],'free_history_frames':1}).update(
                        construction,pred['current_centers'],fused_source_points=pred['current_source_points'])
                    metric=evaluate_geometry(pred['current_centers'],held,.3,0.,safe.observed_free,
                        config['traversability']['bounds'],ids,hids)
                    metric['index']=index
                    metric['validation_scope']='strict scan-local spatial holdout before occupancy and ray construction; un-deskewed; no pose fitting'
                    metric['raw_single_frame']=evaluate_geometry(construction,held,.3,0.,bounds=config['traversability']['bounds'])
                    metric_acc.add(metric);metrics.append(metric)
                    metric_log.write(json.dumps(metric,allow_nan=False)+'\n')
            row={'index':index,'stamp_ns':frame['stamp_ns'],'timestamp':frame['stamp_ns']/1e9,
                 'raw_source_points':int(scan['point_count']) if 'point_count' in scan else int(len(scan['points'])),
                 'route_input_points':len(points),'fused_occupied_voxels':len(result['fused_indices']),
                 'invalid_raw_lidar_returns':int((~valid_returns).sum()),
                 'current_occupied_voxels':len(result['current_indices']),
                 'new_voxels_from_history':result['diagnostics']['new_voxels_from_history'],
                 'traversability_cells':int(local.traversability.size),
                 'class_area_m2':local.summary['class_area_m2'],'ground':local.summary['ground'],
                 'processing_seconds':time.perf_counter()-tick,'semantics_seconds':sem_time,
                 'algorithm':result['diagnostics'],'auxiliary':aux,
                 'rss_mib':psutil.Process().memory_info().rss/2**20}
            peak_ram=max(peak_ram,psutil.Process().memory_info().rss)
            log.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            rows.append(row)
            acceptance.add(index,frame['stamp_ns'],True,len(scan['points']),len(result['fused_indices']),local.traversability.size)
            poses.append(pose)
            if index%10==0:
                world_display.append(transform_points(result['current_centers'],pose).astype(np.float32))
            if index%ev_cfg.get('visualization_stride',50)==0 or index==len(frames)-1:
                save_local_map(local,output/'visualizations'/f'{index:06d}.png',
                               title=f'{route} | acquisition {index}/{total_expected-1}',source_npz=map_path)
            if index%25==0 or index==len(frames)-1:
                write_json(output/'progress.json',{'processed_frames':len(rows),'expected_frames':total_expected,
                    'last_frame':index,'elapsed_seconds':time.perf_counter()-started,'route':route,'complete':False})
                print(f'{route}: {len(rows)}/{total_expected}, frame={index}, voxels={len(result["fused_indices"])}, '
                      f'{row["processing_seconds"]:.2f}s/frame, elapsed={time.perf_counter()-started:.1f}s',flush=True)
    except Exception as error:
        write_json(output/'failure.json',{'frame':frame['index'] if 'frame' in locals() else None,
            'error':str(error),'traceback':traceback.format_exc(),'complete':False})
        raise
    finally:
        log.close()
        metric_log.close()
    times=np.array([r['processing_seconds'] for r in rows])
    areas={k:[r['class_area_m2'][k] for r in rows] for k in rows[0]['class_area_m2']}
    summary={'complete':acceptance.summary()['accepted'],'acceptance':acceptance.summary(),
             'source_lidar_points_accounted':sum(r['raw_source_points'] for r in rows),
             'source_bag_lidar_points':manifest['lidar_point_count'],
             'invalid_raw_lidar_returns':sum(r['invalid_raw_lidar_returns'] for r in rows),
             'route':route,'elapsed_seconds_this_invocation':time.perf_counter()-started,
             'sum_frame_processing_seconds':float(times.sum()),
             'latency_seconds':{'median':float(np.median(times)),'p95':float(np.percentile(times,95)),'max':float(times.max())},
             'peak_process_rss_mib':peak_ram/2**20,
             'class_area_m2':{k:{'mean':float(np.mean(v)),'max':float(np.max(v))} for k,v in areas.items()},
             'frames_with_ground':sum(bool(r['ground'].get('valid')) for r in rows),
             'heldout_geometry':metric_acc.summary(),'heldout_metrics':metrics,
             'semantic_lidar_association_valid':False,'semantic_miou':None,
             'limitations':['Cross-sensor LiDAR-camera extrinsics are unavailable; pixel semantics are real, LiDAR semantic labels are unknown.',
                'Geometry quality and complete-processing acceptance are distinct; no labelled occupancy ground truth is available.',
                'Demo footprint is configurable and is not a measured autonomous vehicle envelope.',
                'Full clearance remains unknown wherever sensor rays do not cover the complete required body volume.']}
    # Read the actual persisted outputs, including any resumed prefix. These
    # artifacts must never silently describe only the tail of a complete run.
    poses=[];world_display=[]
    for row in rows:
        with np.load(output/'occupancy'/f"{row['index']:06d}.npz",allow_pickle=False) as stored:
            pose=stored['T_world_sensor'];poses.append(pose)
            if row['index']%10==0:
                centers=stored['bounds'][0]+(stored['current_indices']+.5)*float(stored['voxel_size'])
                world_display.append(transform_points(centers,pose).astype(np.float32))
    if route=='camera':
        np.savez_compressed(output/'trajectory.npz',T_world_camera=np.asarray(poses),stamp_ns=np.array([r['stamp_ns'] for r in rows]))
        summary['camera_history_resets']=sum(not r['auxiliary']['camera_motion']['accepted'] for r in rows)
        summary['limitations'].append('Camera free-ray evidence comes from predicted depth; it cannot certify measured free space.')
    write_json(output/'summary.json',summary)
    write_json(output/'progress.json',{'processed_frames':len(rows),'expected_frames':total_expected,'complete':summary['complete'],'route':route})
    save_summary_plot(rows,output/'timeline.png')
    if world_display:
        global_points=np.concatenate(world_display)
        keys=np.floor(global_points/.3).astype(np.int64)
        _,first=np.unique(keys,axis=0,return_index=True)
        global_points=global_points[first]
        np.savez_compressed(output/'global_occupied_preview.npz',points=global_points,source_frame_stride=10)
        save_geometry_xy(global_points,output/'global_occupied_preview.png',np.array(poses)[:,:3,3])
    print(json.dumps({'route':route,'complete':summary['complete'],'processed':len(rows),'expected':total_expected}),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(PROJECT/'configs/lidar.json'))
    parser.add_argument('--output')
    parser.add_argument('--limit',type=int,help='development only; never accepted as a full run')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    summary=run(args.config,args.limit,args.resume,args.output)
    if args.limit is None and not summary['complete']:
        raise SystemExit('Full-recording acceptance failed')


if __name__=='__main__':
    main()
