"""Camera-only full-recording entry point with pose conditioning and segments.

The common Gaussian/safety pipeline is reused unchanged. A rejected camera-depth
registration starts a new local segment; its origin is not claimed to have an
independently measured transform to other segments.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

from . import run as pipeline
from .calibration import voxel_downsample,icp_point_to_plane


def json_finite(value):
    if isinstance(value,dict):
        return {str(k):json_finite(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [json_finite(v) for v in value]
    if isinstance(value,(float,np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value,np.integer):
        return int(value)
    if isinstance(value,np.bool_):
        return bool(value)
    return value


class QualityCameraMotion:
    def __init__(self):
        self.pose=np.eye(4);self.relative=np.eye(4);self.previous=None
        self.segment_id=0;self.records=[]

    def update(self,points):
        current=voxel_downsample(points,.3,6000)
        if len(current)<50:
            self.segment_id+=1;self.previous=None;self.relative=np.eye(4)
            info={'accepted':False,'reason':'insufficient_camera_depth_geometry','history_reset':True}
        elif self.previous is None:
            info={'accepted':True,'initial_frame':True}
            self.previous=current
        else:
            relative,info=icp_point_to_plane(current,self.previous,self.relative,max_iterations=20,max_distance=1.)
            finite=np.isfinite(relative).all() and all(np.isfinite(info.get(k,np.inf)) for k in
                      ('condition','point_plane_rmse_m','correspondence_fraction'))
            rotation=float(np.linalg.norm(Rotation.from_matrix(relative[:3,:3]).as_rotvec())) if finite else np.inf
            accepted=bool(finite and info['rank']==6 and info['condition']<1e4 and
                          info['correspondence_fraction']>.25 and info['point_plane_rmse_m']<.35 and
                          np.linalg.norm(relative[:3,3])<1. and rotation<.25)
            info.update(accepted=accepted,conditioning_limit=1e4)
            if accepted:
                self.pose=self.pose@relative;self.relative=relative
            else:
                self.segment_id+=1;self.relative=np.eye(4)
                info.update(history_reset=True,reason='camera_registration_quality_gate')
            self.previous=current
        info.update(segment_id=self.segment_id,pose_source='camera_depth_only',
                    global_segment_registration_valid=self.segment_id==0)
        info=json_finite(info)
        self.records.append(info)
        return self.pose.copy(),info


def main():
    import argparse
    from . import camera_depth
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(pipeline.PROJECT/'configs/camera.json'))
    parser.add_argument('--output')
    parser.add_argument('--limit',type=int,help='development only; never a complete recording result')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    pipeline.CameraMotion=QualityCameraMotion
    original_class=camera_depth.MapAnythingDepth
    original_write=pipeline.write_json
    provider_metadata={}
    class RecordedMapAnything(original_class):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw)
            provider_metadata.update(self.metadata())
    camera_depth.MapAnythingDepth=RecordedMapAnything
    def recorded_write(path,value):
        if Path(path).name=='run.json':
            value={**value,'camera_depth_provider':provider_metadata,
                   'camera_motion_quality_gate':'finite, rank=6, condition<10000, correspondence>.25, rmse<.35m, step<1m/.25rad',
                   'cross_segment_global_registration':'unconstrained after any rejected ICP; local results remain available'}
        return original_write(path,json_finite(value))
    pipeline.write_json=recorded_write
    summary=pipeline.run(args.config,args.limit,args.resume,args.output)
    cfg=json.loads(Path(args.config).read_text())
    output=pipeline.resolve(args.output or cfg['output'])
    rows=[json.loads(line) for line in (output/'frames.jsonl').read_text().splitlines()]
    segment_ids=np.array([r['auxiliary']['camera_motion']['segment_id'] for r in rows],dtype=np.int32)
    with np.load(output/'trajectory.npz',allow_pickle=False) as data:
        trajectory={k:data[k] for k in data.files}
    np.savez_compressed(output/'trajectory.npz',**trajectory,segment_id=segment_ids,
                        registration_accepted=np.array([r['auxiliary']['camera_motion']['accepted'] for r in rows]))
    segments=int(segment_ids.max()+1) if len(segment_ids) else 0
    summary['camera_segments']=segments
    summary['global_pose_continuity_valid']=segments<=1
    summary['camera_depth_provider']=provider_metadata
    original_write(output/'summary.json',summary)
    if segments>1:
        for suffix in ('.png','.json','.npz'):
            path=output/('global_occupied_preview'+suffix)
            if path.exists():
                path.rename(output/('unregistered_segments_preview'+suffix))
        original_write(output/'segments.json',{'segments':segments,
              'scope':'local causal fusion works within each segment; segment anchors are not globally registered',
              'preview':'unregistered_segments_preview is diagnostic and does not claim a continuous camera world map'})
    if args.limit is None and not summary['complete']:
        raise SystemExit('Camera full-recording acceptance failed')


if __name__=='__main__':
    main()
