#!/usr/bin/env python3
"""Actual frozen OpenSeeD panoptic predictions on representative bag RGB views."""
from pathlib import Path
import argparse
import json
import sys
import time

import numpy as np
from PIL import Image

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from orbvi_occ.openseed_provider import OpenSeeDSubprocessProvider
from orbvi_occ.camera_semantic import build_virtual_views
from orbvi_occ.data import load_frames
from orbvi_occ.run import load_images
from orbvi_occ.taxonomy import palette,class_histogram


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames',default='0,135,1174,2347')
    parser.add_argument('--output',default='diagnostics/openseed_panoptic_probe')
    parser.add_argument('--min-confidence',type=float,default=0.)
    parser.add_argument('--image-size',type=int,default=512)
    args=parser.parse_args()
    frames=load_frames(PROJECT/'prepared')
    indices=[int(i) for i in args.frames.split(',')]
    rig=json.loads((PROJECT/'configs/calibration.json').read_text())
    views_config={'size':[518,518],'fov_deg':110.,'angles':[[0.,0.]],'ego_mask_path':'configs/camera_ego_mask.json'}
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    provider=OpenSeeDSubprocessProvider(PROJECT/'models/openseed-swint',image_size=args.image_size,
        min_confidence=args.min_confidence,output_mode='panoptic')
    report={'metadata':provider.metadata(),'virtual_views':views_config,'frames':[]}
    try:
        for index in indices:
            frame=frames[index]
            views=build_virtual_views(load_images(PROJECT/'prepared',frame),rig,views_config)
            started=time.perf_counter();results=provider.predict_views(views)
            record={'index':index,'stamp_ns_text':str(frame['stamp_ns']),
                'runtime_s':time.perf_counter()-started,'inference':provider.last_info,'views':[]}
            directory=output/f'{index:06d}';directory.mkdir(exist_ok=True)
            for view,result in zip(views,results):
                view_id=view['view_id'];np.savez_compressed(directory/(view_id+'.npz'),**result)
                Image.fromarray(view['image']).save(directory/(view_id+'_rgb.jpg'))
                Image.fromarray(palette()[result['labels']]).save(directory/(view_id+'_panoptic.png'))
                Image.fromarray((result['ego_mask'].astype(np.uint8)*255)).save(directory/(view_id+'_ego_mask.png'))
                confidence=(np.clip(result['confidence'],0,1)*255).astype(np.uint8)
                Image.fromarray(confidence).save(directory/(view_id+'_query_confidence.png'))
                labelled=result['labels']!=255
                pointwise_top=np.asarray(result['topk_labels'])[...,0]
                assert np.array_equal(pointwise_top[labelled],result['labels'][labelled]), 'Panoptic label/evidence winner contradiction'
                assert np.all(np.asarray(result['topk_probabilities'])[~labelled]==0), 'Rejected pixel has semantic evidence'
                record['views'].append({'view_id':view_id,'physical_camera':view['physical_camera_id'],
                    'pixel_domain':result['pixel_domain'],'class_counts':class_histogram(result['labels']),
                    'valid_pixels':int(view['valid_mask'].sum()),'masked_ego_pixels':int(result['ego_mask'].sum()),
                    'labelled_pixels':int(labelled.sum()),'labelled_confidence_median':float(np.median(result['confidence'][labelled])) if labelled.any() else None,
                    'label_evidence_match':True,'unassigned_evidence_zero':True,
                    'panoptic_query_ids_saved':True,'rgb':str(directory/(view_id+'_rgb.jpg')),
                    'semantic':str(directory/(view_id+'_panoptic.png'))})
            report['frames'].append(record)
            print(f'frame {index}: actual panoptic outputs saved',flush=True)
    finally:
        provider.close()
    (output/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'report':str(output/'report.json'),'frame_count':len(report['frames']),'GPU_worker_closed':True}),flush=True)


if __name__=='__main__':main()
