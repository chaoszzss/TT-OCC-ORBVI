#!/usr/bin/env python3
"""Profile actual cached camera artifacts and CPU fusion without neural models.

NPZ compression candidates must round-trip exactly. Temporary files belong to
this script only; original artifacts and configurations are never modified.
"""
from __future__ import annotations
import argparse
import cProfile
import io
import json
from pathlib import Path
import pstats
import sys
import tempfile
import time
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from orbvi_occ.camera_fusion import CameraSemanticFusion,ray_inconsistent_history,_ray_index
from orbvi_occ.tt_backend import transform_points
from orbvi_occ.taxonomy import load_registry


def read_npz(path):
    with np.load(path,allow_pickle=False) as data:
        return {k:data[k].copy() for k in data.files}


def write_npz(path,arrays,level):
    if level==6:
        np.savez_compressed(path,**arrays)
        return
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=level) as archive:
        for key,arr in arrays.items():
            data=io.BytesIO()
            np.lib.format.write_array(data,np.asanyarray(arr),allow_pickle=False)
            archive.writestr(key+'.npy',data.getvalue())


def io_profile(arrays,temporary,name):
    results=[]
    for level in [6,1,3,0]:
        path=temporary/f'{name}_level{level}.npz'
        start=time.perf_counter();write_npz(path,arrays,level);seconds=time.perf_counter()-start
        start=time.perf_counter();actual=read_npz(path);read_seconds=time.perf_counter()-start
        exact=set(actual)==set(arrays) and all(v.dtype==actual[k].dtype and v.shape==actual[k].shape and np.array_equal(v,actual[k],equal_nan=v.dtype.kind in 'fc') for k,v in arrays.items())
        if not exact:
            raise RuntimeError('Compression candidate changed array data or dtype')
        results.append({'compresslevel':level,'write_seconds':seconds,'read_seconds':read_seconds,
            'bytes':path.stat().st_size,'exact_roundtrip':exact})
        path.unlink()
    return results


def bounded_ray_reference(past_points,current_points,*,angular_tolerance_deg=1.,range_margin=.6,
                          ray_origins=None,min_range=0.,min_neighbors=1,_prepared_index=None):
    old=np.asarray(past_points,float)
    index=_ray_index(current_points,ray_origins,min_neighbors) if _prepared_index is None else _prepared_index
    angle_limit=2*np.sin(np.deg2rad(angular_tolerance_deg)/2)
    inconsistent=np.zeros(len(old),bool)
    for origin,new_r,tree in index:
        local_old=old-origin;old_r=np.linalg.norm(local_old,axis=1)
        eligible=np.flatnonzero(old_r>max(.25,min_range))
        if not len(eligible):
            continue
        directions=local_old[eligible]/old_r[eligible,None]
        chord,idx=tree.query(directions,k=min_neighbors,
                             distance_upper_bound=np.nextafter(angle_limit,np.inf))
        if min_neighbors==1:
            chord,idx=chord[:,None],idx[:,None]
        covered=np.all(chord<angle_limit,axis=1)
        if np.any(covered):
            selected=eligible[covered]
            nearest_surface=new_r[idx[covered]].min(1)
            inconsistent[selected] |= old_r[selected]+range_margin<nearest_surface
    return inconsistent


def compare_bounded_rays(frames,config):
    fusion=CameraSemanticFusion(config,load_registry())
    rows=[]
    for index,source,occ in frames:
        pose=occ['T_world_sensor'];inverse=np.linalg.inv(pose)
        ray_index=_ray_index(source['points'],source['origins'],fusion.config.ray_min_neighbors)
        for past in fusion.history:
            local=transform_points(past['points_world'],inverse).astype(np.float32)
            kwargs={'angular_tolerance_deg':fusion.config.ray_angular_tolerance_deg,
                'range_margin':fusion.config.ray_range_margin,'ray_origins':source['origins'],
                'min_range':fusion.config.ray_min_range,'min_neighbors':fusion.config.ray_min_neighbors,
                '_prepared_index':ray_index}
            start=time.perf_counter();original=ray_inconsistent_history(local,source['points'],**kwargs);old_seconds=time.perf_counter()-start
            start=time.perf_counter();bounded=bounded_ray_reference(local,source['points'],**kwargs);new_seconds=time.perf_counter()-start
            rows.append({'current_index':index,'past_stamp_ns':past['stamp_ns'],'past_points':len(local),
                'original_seconds':old_seconds,'bounded_seconds':new_seconds,'removed':int(original.sum()),
                'difference_count':int(np.sum(original!=bounded))})
        fusion.update(source['points'],pose,int(source['stamp_ns']),evidence=source['evidence'],
            semantic_confidence=source['semantic_confidence'],origins=source['origins'],depth_confidence=source['depth_confidence'])
    old=sum(r['original_seconds'] for r in rows);new=sum(r['bounded_seconds'] for r in rows)
    return {'rows':rows,'original_seconds':old,'bounded_seconds':new,'speedup':old/new,
        'all_boolean_results_exact':all(r['difference_count']==0 for r in rows),
        'tested_history_points':sum(r['past_points'] for r in rows)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',default='runs/camera_semantic_b5_pilot')
    parser.add_argument('--frame',type=int,default=4)
    parser.add_argument('--replay-frames',type=int,default=8)
    parser.add_argument('--output',default='diagnostics/camera_semantic_backend/runtime_profile.json')
    parser.add_argument('--bounded-ray-check',action='store_true')
    args=parser.parse_args()
    run=Path(args.run);run=run if run.is_absolute() else ROOT/run
    metadata=json.loads((run/'run.json').read_text())
    report={'scope':'Actual persisted pilot artifacts, exact NPZ roundtrip and causal CPU-only fusion replay. No model/GPU loaded, no original outputs changed.','run':str(run),'frame':args.frame,'io':{}}
    name=f'{args.frame:06d}'
    sources={'source_points':run/'source_points'/f'{name}.npz','occupancy':run/'occupancy'/f'{name}.npz'}
    for view in json.loads((run/'views'/f'{name}.json').read_text()):
        sources['semantic_'+view['view_id']]=run/view['semantic_maps_path']
    with tempfile.TemporaryDirectory(prefix='ttocc-io-profile-') as directory:
        tmp=Path(directory)
        for name,path in sources.items():
            data=read_npz(path)
            report['io'][name]={'source':str(path),'array_uncompressed_bytes':sum(v.nbytes for v in data.values()),'candidates':io_profile(data,tmp,name)}
    frames=[]
    for index in range(args.replay_frames):
        points=run/'source_points'/f'{index:06d}.npz';occ=run/'occupancy'/f'{index:06d}.npz'
        if not points.exists() or not occ.exists():
            break
        frames.append((index,read_npz(points),read_npz(occ)))
    fusion=CameraSemanticFusion(metadata['config']['ttocc'],load_registry())
    rows=[];profiler=cProfile.Profile();profiler.enable()
    for index,source,occ in frames:
        start=time.perf_counter()
        actual=fusion.update(source['points'],occ['T_world_sensor'],int(source['stamp_ns']),
            evidence=source['evidence'],semantic_confidence=source['semantic_confidence'],
            origins=source['origins'],depth_confidence=source['depth_confidence'])
        elapsed=time.perf_counter()-start
        equal=np.array_equal(actual['fused_indices'],occ['fused_indices']) and np.array_equal(actual['fused_labels'],occ['fused_labels'])
        rows.append({'index':index,'seconds':elapsed,'stored_geometry_labels_exact':equal,'diagnostics':actual['diagnostics']})
    profiler.disable()
    stream=io.StringIO();pstats.Stats(profiler,stream=stream).strip_dirs().sort_stats('cumtime').print_stats(35)
    report['fusion_replay']={'rows':rows,'profile':stream.getvalue(),'all_geometry_labels_exact':all(r['stored_geometry_labels_exact'] for r in rows)}
    if args.bounded_ray_check:
        report['bounded_ray_check']=compare_bounded_rays(frames,metadata['config']['ttocc'])
    output=Path(args.output);output=output if output.is_absolute() else ROOT/output
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2))
    print(json.dumps({'output':str(output),'io':report['io'],'fusion_replay_rows':[{'index':r['index'],'seconds':r['seconds'],'exact':r['stored_geometry_labels_exact']} for r in rows]},indent=2))
    print(stream.getvalue())


if __name__=='__main__':
    main()
