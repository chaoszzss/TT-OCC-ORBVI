"""Optional official OpenSeeD Swin-T inference in a separate dependency environment.

The main runtime never imports Detectron2. A persistent isolated worker loads
all 793 official checkpoint keys strictly, then supplies native garage labels,
soft evidence and panoptic IDs from calibrated RGB views. No training occurs.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
import traceback
from types import MethodType

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _panoptic_with_evidence(model, mask_cls, mask_pred):
    """Call canonical panoptic code unchanged; record its actual winning query.

    Query choice uses the same keep gate and cur_scores * sigmoid(mask) ranking
    as upstream. Candidate class evidence comes from that winning query, rather
    than a different semantic decoder distribution. Nonassigned pixels are zero.
    """
    torch = __import__('torch')
    F = torch.nn.functional
    pano, segments = model._canonical_panoptic_inference(mask_cls,mask_pred)
    raw_scores, raw_labels = mask_cls.sigmoid().max(-1)
    keep = raw_labels.ne(model.sem_seg_head.num_classes) & (raw_scores > model.object_mask_threshold)
    class_scores = mask_cls.sigmoid()
    if model.transform_eval:
        class_scores = F.softmax(class_scores/model.pano_temp,dim=-1)
    scores = class_scores.max(-1).values[keep]
    masks = mask_pred.sigmoid()[keep]
    query_ids = torch.arange(mask_cls.shape[0],device=mask_cls.device)[keep]
    h,w = pano.shape
    confidence = torch.zeros((h,w),dtype=torch.float32,device=pano.device)
    query_map = torch.full((h,w),-1,dtype=torch.int32,device=pano.device)
    evidence = torch.zeros((class_scores.shape[1],h,w),dtype=torch.float32,device=pano.device)
    if len(masks):
        winners = (scores[:,None,None]*masks).argmax(0)
        y,x = torch.meshgrid(torch.arange(h,device=pano.device),torch.arange(w,device=pano.device),indexing='ij')
        mask_scores = masks[winners,y,x]
        assigned = pano>0
        confidence[assigned] = (scores[winners]*mask_scores)[assigned].float()
        query_map[assigned] = query_ids[winners[assigned]].to(torch.int32)
        probabilities = class_scores[keep][winners] * mask_scores[:,:,None]
        evidence = probabilities.permute(2,0,1).contiguous().float()
        evidence[:,~assigned] = 0
    model._last_panoptic_evidence = {'confidence':confidence,'query_ids':query_map,'evidence':evidence}
    return pano,segments


class OpenSeeDWorker:
    """Runs only inside .openseed-venv with real Detectron2/CUDA kernels."""
    def __init__(self, model_dir, *, image_size=512, min_confidence=0., topk=3, device="cuda", output_mode="panoptic"):
        from PIL import Image
        if not hasattr(Image,"LINEAR"):
            Image.LINEAR = Image.BILINEAR  # Detectron2 0.6 uses the old Pillow alias.
        import torch
        import yaml
        from .taxonomy import load_registry
        self.torch = torch
        self.device = device
        if device != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('Official OpenSeeD adapter requires CUDA; CLIP text encoder uses CUDA explicitly')
        self.image_size = int(image_size)
        self.min_confidence = float(min_confidence)
        self.topk = min(int(topk), 11)
        if output_mode not in ("panoptic","semantic"):
            raise ValueError("output_mode must be panoptic or semantic")
        self.output_mode = output_mode
        self.registry = load_registry()
        self.model_dir = Path(model_dir).resolve()
        self.source = PROJECT/'models/openseed-source'
        self.checkpoint = self.model_dir/'model_state_dict_swint_51.2ap.pt'
        sys.path.insert(0,str(self.source))
        from openseed import build_model
        from detectron2.data import MetadataCatalog
        random.seed(0);np.random.seed(0);torch.manual_seed(0)
        cfg_path = self.source/'configs/openseed/openseed_swint_lang.yaml'
        cfg = yaml.safe_load(cfg_path.read_text())
        # The complete official checkpoint supplies pretrained Swin parameters;
        # no redundant initialization checkpoint or uninitialized keys are used.
        cfg['MODEL']['BACKBONE']['LOAD_PRETRAINED'] = False
        cfg['MODEL']['TEXT']['PRETRAINED_TOKENIZER'] = str(PROJECT/'models/clip-tokenizer')
        self.model = build_model(cfg)
        checkpoint = torch.load(self.checkpoint,map_location='cpu',weights_only=True)
        self.model.load_state_dict(checkpoint,strict=True)
        self.model = self.model.eval().to(device)
        self.class_names = [c['name'] for c in self.registry['classes']]
        self.vocabulary = ['wall','floor','ceiling','column','car','person','door','tree','obstacle','road','sky']
        stuff_ids = [i for i in range(11) if i not in (4,5)]
        metadata = MetadataCatalog.get('orbvi_openseed_garage').set(
            thing_classes=['car','person'], thing_dataset_id_to_contiguous_id={4:4,5:5},
            stuff_classes=[self.vocabulary[i] for i in stuff_ids],
            stuff_dataset_id_to_contiguous_id={i:i for i in stuff_ids})
        with torch.inference_mode():
            self.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(self.vocabulary,is_eval=False)
        self.model.metadata = metadata
        self.model.sem_seg_head.num_classes = 11
        self.model._canonical_panoptic_inference = self.model.panoptic_inference
        self.model.panoptic_inference = MethodType(_panoptic_with_evidence,self.model)
        self.last_info = {}
        self._metadata = {
            'provider':'Official OpenSeeD Swin-T pretrained checkpoint',
            'repository':'IDEA-Research/OpenSeeD',
            'revision':'release-openseed-asset-105743964',
            'release_asset_id':105743964,
            'checkpoint_url':'https://github.com/IDEA-Research/OpenSeeD/releases/download/openseed/model_state_dict_swint_51.2ap.pt',
            'source_commit':json.loads((self.source/'source-provenance.json').read_text())['commit'],
            'checkpoint':str(self.checkpoint),'checkpoint_sha256':_sha256(self.checkpoint),
            'checkpoint_bytes':self.checkpoint.stat().st_size,
            'weights_loaded_strict':True,'checkpoint_keys':len(checkpoint),
            'open_vocabulary':True,'trained_on_orbvi':False,'registry':self.registry['name'],
            'vocabulary':self.vocabulary,'vocabulary_ids':list(range(11)),
            'image_size':self.image_size,'device':device,'dtype':'float32',
            'text_prompt_method':'official random single template, Python/NumPy/Torch seed 0',
            'output_mode':self.output_mode,
            'confidence_definition':('actual winning panoptic query class score times per-pixel sigmoid mask score; uncalibrated' if self.output_mode=='panoptic' else 'class-normalized OpenSeeD weighted mask mass; uncalibrated'),
            'evidence_definition':('actual winning panoptic query topk class score times pixel mask score; rejected segment pixels zero' if self.output_mode=='panoptic' else 'class-normalized semantic mask mass topk'),
            'semantic_model_substitution':False,
            'canonical_panoptic_function_unchanged':True,
            'additional_pixel_confidence_threshold':self.min_confidence,
            'panoptic_gate':{'object_mask_threshold':float(self.model.object_mask_threshold),'overlap_threshold':float(self.model.overlap_threshold),'mask_probability_threshold':.5},
            'compatibility_adapters':['Pillow Image.LINEAR=BILINEAR',
                'deform CUDA AT_DISPATCH value.type() -> value.scalar_type() in model source copy'],
            'isolated_environment':str(PROJECT/'.openseed-venv'),
        }

    def metadata(self):
        return dict(self._metadata)

    def predict_views(self, views):
        from PIL import Image
        from .taxonomy import class_histogram
        torch = self.torch
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        outputs = []
        for view in views:
            rgb = np.asarray(view['image'],dtype=np.uint8)
            if rgb.ndim != 3 or rgb.shape[2] != 3:
                raise ValueError('OpenSeeD views require RGB HxWx3')
            h,w = rgb.shape[:2]
            scale = self.image_size/min(h,w)
            resized = np.asarray(Image.fromarray(rgb).resize((round(w*scale),round(h*scale)),Image.Resampling.BICUBIC))
            image = torch.from_numpy(resized.copy()).permute(2,0,1).to(self.device)
            valid = np.asarray(view.get('valid_mask',np.ones((h,w),bool)),bool)
            if valid.shape!=(h,w):
                raise ValueError('view validity mask must match its RGB pixel domain')
            with torch.inference_mode():
                result = self.model([{'image':image,'height':h,'width':w}],inference_task='sem_pan_seg')[0]
                mass = result['sem_seg'].float().clamp_min(0)
                evidence = mass / mass.sum(0).clamp_min(1e-8)
                native_prob, native_labels = evidence.topk(self.topk,dim=0)
                pano, segments = result['panoptic_seg']
                if self.output_mode == 'panoptic':
                    labels = torch.full((h,w),255,dtype=torch.uint8,device=self.device)
                    for segment in segments:
                        labels[pano==segment['id']] = segment['category_id']
                    conf = self.model._last_panoptic_evidence['confidence'].clone()
                    query_evidence = self.model._last_panoptic_evidence['evidence']
                    top_prob,top_labels = query_evidence.topk(self.topk,dim=0)
                else:
                    top_prob,top_labels = native_prob.clone(),native_labels.clone()
                    conf,labels = top_prob[0].clone(),top_labels[0].to(torch.uint8)
                trusted = torch.as_tensor(valid,device=self.device) & (conf>=self.min_confidence) & (labels!=10) & (labels!=255)
                labels[~trusted] = 255
                conf[~trusted] = 0
                top_prob[:,~trusted] = 0
                outputs.append({'labels':labels.cpu().numpy(), 'confidence':conf.cpu().numpy().astype(np.float32),
                    'topk_labels':top_labels.permute(1,2,0).cpu().numpy().astype(np.uint8),
                    'topk_probabilities':top_prob.permute(1,2,0).cpu().numpy().astype(np.float32),
                    'source_labels':native_labels[0].cpu().numpy().astype(np.uint8),
                    'source_confidence':evidence.max(0).values.cpu().numpy().astype(np.float32),
                    'instance_ids':pano.cpu().numpy().astype(np.int32),
                    'panoptic_query_ids':self.model._last_panoptic_evidence['query_ids'].cpu().numpy().astype(np.int32),
                    'ego_mask':np.asarray(view.get('ego_mask',np.zeros((h,w),bool))),
                    'rectification_valid_mask':np.asarray(view.get('rectification_valid_mask',np.ones((h,w),bool))),
                    'instance_segments_json':json.dumps(segments),
                    'valid_mask':valid,'pixel_domain':'rectified_virtual_pinhole',
                    'registry':self.registry['name'],'view_id':str(view.get('view_id',view.get('camera_id')))})
        self.last_info = {'runtime_ms':(time.perf_counter()-started)*1000,'device':self.device,
            'views':len(views),'class_histograms':[class_histogram(o['labels'],self.registry) for o in outputs],
            'peak_allocated_gpu_mib':torch.cuda.max_memory_allocated()/2**20}
        return outputs

    def close(self):
        self.model.to('cpu')
        self.torch.cuda.empty_cache()


class OpenSeeDSubprocessProvider:
    """Compatible with CameraSemanticProvider, without its dependency conflicts."""
    def __init__(self, model_dir, device='auto', image_size=512, min_confidence=0.,
                 topk=3, fallback_cpu=False, python=None, output_mode="panoptic", registry=None, **kwargs):
        from .taxonomy import load_registry
        self.registry = registry or load_registry()
        self.class_count = len(self.registry['classes'])
        self.device = 'cuda' if device=='auto' else device
        self.last_info = {}
        self._temporary = tempfile.TemporaryDirectory(prefix='ttocc-openseed-')
        self._step = 0
        python = python or str(PROJECT/'.openseed-venv/bin/python')
        command = [python,'-m','orbvi_occ.openseed_provider','--serve','--model-dir',str(Path(model_dir).resolve()),
                   '--image-size',str(image_size),'--min-confidence',str(min_confidence),'--topk',str(topk),'--device',self.device,'--output-mode',output_mode]
        self._process = subprocess.Popen(command,cwd=PROJECT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                                         stderr=sys.stderr,text=True,bufsize=1)
        hello = self._read()
        self._metadata = hello['metadata']

    def _read(self):
        line = self._process.stdout.readline()
        if not line:
            raise RuntimeError(f'OpenSeeD isolated worker exited ({self._process.poll()}); see stderr for actual dependency/model error')
        data = json.loads(line)
        if data.get('status')=='error':
            raise RuntimeError('OpenSeeD worker failed: '+data.get('detail','unknown error'))
        return data

    def metadata(self):
        return dict(self._metadata)

    def predict_views(self,views):
        directory = Path(self._temporary.name)/f'{self._step:06d}'
        directory.mkdir()
        sources=[]
        for i,view in enumerate(views):
            path=directory/f'in_{i}.npz'
            np.savez(path,image=view['image'],valid_mask=view.get('valid_mask',np.ones(view['image'].shape[:2],bool)),
                     camera_id=str(view.get('camera_id',i)),view_id=str(view.get('view_id',i)),
                     ego_mask=view.get('ego_mask',np.zeros(view['image'].shape[:2],bool)),
                     rectification_valid_mask=view.get('rectification_valid_mask',np.ones(view['image'].shape[:2],bool)))
            sources.append(str(path))
        self._process.stdin.write(json.dumps({'command':'predict','inputs':sources,'output':str(directory)})+'\n')
        self._process.stdin.flush()
        response=self._read()
        self.last_info=response['last_info']
        outputs=[]
        for file in response['outputs']:
            with np.load(file,allow_pickle=False) as stored:
                result={k:stored[k].copy() for k in stored.files}
            for key in ('pixel_domain','registry','view_id','instance_segments_json'):
                result[key]=str(result[key].item())
            outputs.append(result)
        import shutil
        shutil.rmtree(directory)
        self._step+=1
        return outputs

    def close(self):
        if self._process.poll() is None:
            try:
                self._process.stdin.write('{"command":"close"}\n');self._process.stdin.flush()
                self._process.wait(timeout=15)
            except (OSError,subprocess.TimeoutExpired):
                self._process.terminate();self._process.wait(timeout=5)
        self._temporary.cleanup()


def serve(args):
    try:
        with redirect_stdout(sys.stderr):
            worker=OpenSeeDWorker(args.model_dir,image_size=args.image_size,min_confidence=args.min_confidence,topk=args.topk,device=args.device,output_mode=args.output_mode)
        print(json.dumps({'status':'ready','metadata':worker.metadata()}),flush=True)
        for line in sys.stdin:
            request=json.loads(line)
            if request['command']=='close':
                worker.close();return
            if request['command']!='predict':
                raise ValueError('unsupported worker command')
            views=[]
            for file in request['inputs']:
                with np.load(file,allow_pickle=False) as stored:
                    views.append({'image':stored['image'],'valid_mask':stored['valid_mask'],
                                  'camera_id':str(stored['camera_id'].item()),'view_id':str(stored['view_id'].item()),
                                  'ego_mask':stored['ego_mask'],'rectification_valid_mask':stored['rectification_valid_mask']})
            with redirect_stdout(sys.stderr):
                outputs=worker.predict_views(views)
            paths=[]
            for i,output in enumerate(outputs):
                path=Path(request['output'])/f'out_{i}.npz';np.savez(path,**output);paths.append(str(path))
            print(json.dumps({'status':'ready','outputs':paths,'last_info':worker.last_info}),flush=True)
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({'status':'error','detail':f'{type(exc).__name__}: {exc}'}),flush=True)
        raise SystemExit(1)


# Backward-compatible class spelling for the dedicated full-recording runner.
OpenSeeDProvider = OpenSeeDSubprocessProvider


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serve',action='store_true',required=True)
    parser.add_argument('--model-dir',default=str(PROJECT/'models/openseed-swint'))
    parser.add_argument('--image-size',type=int,default=512)
    parser.add_argument('--min-confidence',type=float,default=0.)
    parser.add_argument('--topk',type=int,default=3)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--output-mode',choices=('panoptic','semantic'),default='panoptic')
    args=parser.parse_args();serve(args)


if __name__=='__main__':
    main()
