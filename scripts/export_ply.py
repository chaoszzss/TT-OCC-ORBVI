#!/usr/bin/env python3
"""Export actual occupied voxel centres for CloudCompare/Open3D inspection."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--route',choices=['lidar','camera'],required=True)
    parser.add_argument('--frame',type=int,required=True)
    parser.add_argument('--run',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--world',action='store_true')
    args=parser.parse_args()
    run=args.run or ROOT/'runs'/f'{args.route}_full'
    source=run/'occupancy'/f'{args.frame:06d}.npz'
    if args.world and args.route=='camera':
        summary=json.loads((run/'summary.json').read_text())
        if not summary.get('global_pose_continuity_valid'):
            raise SystemExit('Camera segments are not globally registered; export local coordinates instead.')
    with np.load(source,allow_pickle=False) as data:
        xyz=data['bounds'][0]+(data['fused_indices']+.5)*float(data['voxel_size'])
        labels=data['fused_labels'].copy()
        pose=data['T_world_sensor'].copy()
        stamp=int(data['stamp_ns']);voxel=float(data['voxel_size'])
    if args.world:
        xyz=xyz@pose[:3,:3].T+pose[:3,3]
    palette=np.random.default_rng(0).integers(32,256,(256,3),dtype=np.uint8)
    palette[255]=[160,160,160]
    record=np.empty(len(xyz),dtype=[('x','<f4'),('y','<f4'),('z','<f4'),
                    ('red','u1'),('green','u1'),('blue','u1'),('semantic_id','u1')])
    for k,axis in enumerate('xyz'):
        record[axis]=xyz[:,k]
    for k,channel in enumerate(('red','green','blue')):
        record[channel]=palette[labels,k]
    record['semantic_id']=labels
    output=args.output or ROOT/'reports/pointclouds'/f'{args.route}_{args.frame:06d}.ply'
    output.parent.mkdir(parents=True,exist_ok=True)
    header=('ply\nformat binary_little_endian 1.0\n'
       'comment Actual TT-Occ occupied voxel centres; missing cells are not confirmed free\n'
       f'element vertex {len(record)}\nproperty float x\nproperty float y\nproperty float z\n'
       'property uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar semantic_id\nend_header\n')
    with output.open('wb') as file:
        file.write(header.encode('ascii'));file.write(record.tobytes())
    output.with_suffix('.json').write_text(json.dumps({'source_npz':str(source),'route':args.route,
         'frame':args.frame,'stamp_ns':stamp,'vertices':len(record),'voxel_size_m':voxel,
         'coordinate_frame':'estimated world' if args.world else 'current route sensor/rig',
         'semantic_255':'unknown_semantic; all exported vertices are geometrically occupied',
         'source_npz_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
         'scope':'voxel centres, no fabricated surfaces or unobserved free space'},indent=2))
    expected_bytes=len(header.encode('ascii'))+16*len(record)
    if output.stat().st_size!=expected_bytes:
        raise RuntimeError('PLY binary size verification failed')
    print(f'{output}: {len(record)} actual occupied voxel centres')


if __name__=='__main__':
    main()
