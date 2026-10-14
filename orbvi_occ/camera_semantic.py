"""Camera-only semantic lifting with immutable source-view/pixel provenance.

Images for depth and semantics are the same calibrated virtual pinhole views.
The label at a point's source UV is retained; cross-camera central-angle
reprojection is intentionally absent. Existing geometry-only routes are untouched.
"""
from __future__ import annotations
from pathlib import Path
import json
import hashlib
import time
import cv2
import numpy as np
from .projection import project_triple_sphere
from .taxonomy import load_registry, class_mapping, class_histogram, UNKNOWN

OPTICAL_TO_FLU = np.array([[0,0,1,0],[-1,0,0,0],[0,-1,0,0],[0,0,0,1.]], dtype=float)


class VirtualPinhole:
    """A calibrated yaw/pitch virtual view from one physical fish-eye camera."""
    def __init__(self, camera, size=(518,392), fov_deg=120., yaw_deg=0., pitch_deg=0.):
        from scipy.spatial.transform import Rotation
        self.size = tuple(map(int, size))
        self.source_size = (camera['width'], camera['height'])
        w, h = self.size
        if not (30 <= fov_deg < 160) or min(w,h) < 32:
            raise ValueError('Invalid virtual pinhole geometry')
        f = w / (2*np.tan(np.deg2rad(fov_deg)/2))
        self.K = np.array([[f,0,(w-1)/2],[0,f,(h-1)/2],[0,0,1.]])
        self.T_physical_virtual = np.eye(4)
        # Right-handed optical coordinates: yaw about optical y, pitch about x.
        self.T_physical_virtual[:3,:3] = Rotation.from_euler('yx',[yaw_deg,pitch_deg],degrees=True).as_matrix()
        v,u = np.meshgrid(np.arange(h),np.arange(w),indexing='ij')
        rays = np.stack([(u-self.K[0,2])/f,(v-self.K[1,2])/f,np.ones_like(u)],-1)
        physical_rays = rays.reshape(-1,3) @ self.T_physical_virtual[:3,:3].T
        uv, valid = project_triple_sphere(physical_rays,camera['params'])
        sw,sh = self.source_size
        valid &= (uv[:,0]>=0)&(uv[:,0]<sw-1)&(uv[:,1]>=0)&(uv[:,1]<sh-1)
        self.valid = valid.reshape(h,w)
        uv[~valid] = -1
        self.mapx,self.mapy = uv[:,0].reshape(h,w).astype(np.float32),uv[:,1].reshape(h,w).astype(np.float32)

    def rectify(self, image):
        if image.shape[:2] != self.source_size[::-1]:
            raise ValueError('RGB size disagrees with the physical camera calibration')
        return cv2.remap(image,self.mapx,self.mapy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)


def build_virtual_views(images, camera_cfg, view_config=None):
    """Create real calibrated virtual views; no fabricated additional sensors."""
    cfg = view_config or {}
    angles = cfg.get('angles', [[0.,0.]])
    ego_cfg=cfg.get('ego_mask_config')
    ego_path=cfg.get('ego_mask_path')
    if ego_path:
        path=Path(ego_path)
        if not path.is_absolute():
            path=Path(__file__).resolve().parents[1]/path
        ego_cfg=json.loads(path.read_text())
    if ego_cfg is not None:
        expected=ego_cfg.get('virtual_views',{})
        if ego_cfg.get('coordinate_domain')!='normalized_virtual_pinhole':
            raise ValueError('Ego mask coordinate domain is not calibrated virtual pinhole')
        actual={'size':list(cfg.get('size',[518,392])),
                'fov_deg':float(cfg.get('fov_deg',120.)),'angles':angles}
        if expected.get('size')!=actual['size'] or not np.isclose(expected.get('fov_deg',-1),actual['fov_deg']) or not np.allclose(expected.get('angles',[]),actual['angles']):
            raise ValueError('Ego mask was authored for a different virtual camera geometry')
    views = []
    for cid in camera_cfg['camera_order']:
        if cid not in images:
            raise ValueError(f'Missing physical camera {cid}')
        camera = camera_cfg['cameras'][cid]
        for angle_i, (yaw,pitch) in enumerate(angles):
            rect = VirtualPinhole(camera,cfg.get('size',[518,392]),cfg.get('fov_deg',120.),yaw,pitch)
            view_id=f'{cid}_v{angle_i}'
            ego_mask=np.zeros(rect.valid.shape,np.uint8)
            if ego_cfg is not None:
                item=ego_cfg.get('views',{}).get(view_id)
                if item is None:
                    raise ValueError(f'Manual ego-mask config is missing virtual view {view_id}')
                polygon=np.asarray(item.get('polygon',[]),float)
                if polygon.ndim!=2 or polygon.shape[1]!=2 or len(polygon)<3 or not np.isfinite(polygon).all() or np.any((polygon<0)|(polygon>1)):
                    raise ValueError('Ego-body polygon must contain finite normalized image coordinates')
                h,w=rect.valid.shape
                vertices=np.rint(polygon*np.array([w,h])).astype(np.int32)
                vertices=np.clip(vertices,[0,0],[w-1,h-1])
                cv2.fillPoly(ego_mask,[vertices],1)
                padding=int(ego_cfg.get('padding_px',0))
                if padding<0 or padding>min(w,h)//10:
                    raise ValueError('Invalid ego polygon padding')
                if padding:
                    ego_mask=cv2.dilate(ego_mask,np.ones((2*padding+1,2*padding+1),np.uint8))
            views.append({'camera_id':cid,'view_id':view_id,
                'image':rect.rectify(images[cid]),'K':rect.K,
                'T_rig_camera':OPTICAL_TO_FLU @ np.asarray(camera['T_camera0_camera']) @ rect.T_physical_virtual,
                'valid_mask':rect.valid & ~ego_mask.astype(bool),
                'rectification_valid_mask':rect.valid,'ego_mask':ego_mask.astype(bool),
                'validity_sampling':'pixel_center',
                'ego_mask_scope':'manual static acquisition-body image polygon; excluded pixels remain unknown, not free' if ego_cfg is not None else 'no ego-body exclusion configured',
                'physical_camera_id':cid,
                'virtual_yaw_deg':float(yaw),'virtual_pitch_deg':float(pitch),
                'rectification_mapx':rect.mapx,'rectification_mapy':rect.mapy})
    return views


class CameraSemanticProvider:
    """Frozen official SegFormer logits, remapped by names to garage classes.

    A larger ADE20K checkpoint is supported by passing its model directory.
    This is explicitly a semantic-model substitution for TT-Occ OpenSeeD.
    No model training or confidence calibration is performed here.
    """
    def __init__(self,model_dir,device='auto',image_size=512,provider='segformer',
                 min_confidence=.25,topk=3,registry=None,fallback_cpu=True):
        import torch
        from transformers import SegformerImageProcessor,SegformerForSemanticSegmentation
        if provider != 'segformer':
            raise ValueError('This provider supports SegFormer; use the dedicated OpenSeeD provider for OpenSeeD')
        self.registry = registry or load_registry()
        self.model_dir = Path(model_dir).resolve()
        provenance_path=self.model_dir/'provenance.json'
        self.provenance=json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
        weight_path=self.model_dir/'model.safetensors'
        expected=self.provenance.get('files',{}).get('model.safetensors',{}).get('sha256')
        if expected is None or not weight_path.exists():
            raise RuntimeError('Semantic checkpoint requires a local safetensors file and provenance SHA-256')
        digest=hashlib.sha256()
        with weight_path.open('rb') as handle:
            for chunk in iter(lambda:handle.read(8*1024**2),b''):
                digest.update(chunk)
        self.weight_sha256=digest.hexdigest()
        if self.weight_sha256 !=expected:
            raise RuntimeError('Semantic checkpoint SHA-256 differs from the recorded official snapshot')
        self.image_size,self.min_confidence,self.topk = int(image_size),float(min_confidence),int(topk)
        if self.image_size<128 or self.image_size%32 or not (0<=self.min_confidence<=1):
            raise ValueError('Invalid semantic image size or threshold')
        if device=='auto':
            free = torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else 0
            device = 'cuda' if free>1024**3 else 'cpu'
        self.device,self.fallback_cpu = device,bool(fallback_cpu)
        self.processor = SegformerImageProcessor.from_pretrained(self.model_dir,local_files_only=True)
        self.model,loading = SegformerForSemanticSegmentation.from_pretrained(self.model_dir,
            local_files_only=True,use_safetensors=True,output_loading_info=True)
        if any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
            raise RuntimeError(f'Pretrained semantic checkpoint was not loaded completely: {loading}')
        self.model=self.model.eval().to(device)
        self.model.requires_grad_(False)
        self.source_names = {int(k):v for k,v in self.model.config.id2label.items()}
        self.mapping = class_mapping(self.source_names,self.registry)
        self.class_count = len(self.registry['classes'])
        self.last_info = {}

    def metadata(self):
        return {'provider':'Frozen official SegFormer ADE20K checkpoint',
            'model_dir':str(self.model_dir),'architecture':self.model.config.architectures,
            'repository':self.provenance.get('repository'),'revision':self.provenance.get('revision'),
            'weights_loaded_strict':True,
            'weights_sha256_verified':True,'weights_sha256':self.weight_sha256,
            'hidden_sizes':self.model.config.hidden_sizes,'device':self.device,
            'image_size':self.image_size,'min_confidence':self.min_confidence,'topk':self.topk,
            'registry':self.registry['name'],'class_mapping':self.mapping,
            'input_domain':'same calibrated virtual pinhole RGB as depth',
            'ttocc_semantic_provider_deviation':'SegFormer replaces OpenSeeD; not an OpenSeeD reproduction',
            'trained_on_orbvi':False,'softmax_is_calibrated_probability':False}

    def _predict(self,view):
        import torch
        image = np.asarray(view['image'])
        h,w = image.shape[:2]
        if image.ndim !=3 or image.shape[-1]!=3 or image.dtype !=np.uint8:
            raise ValueError('Expected RGB uint8 image')
        inputs = self.processor(images=image,return_tensors='pt',size={'height':self.image_size,'width':self.image_size})
        inputs = {k:v.to(self.device) for k,v in inputs.items()}
        with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.float16,enabled=self.device.startswith('cuda')):
            logits = self.model(**inputs).logits.float()
        # Interpolate only to the actual semantic image, not 150 full raw maps.
        logits = torch.nn.functional.interpolate(logits,size=(h,w),mode='bilinear',align_corners=False)
        source_prob = logits.softmax(1)[0]
        raw_conf, raw_label = source_prob.max(0)
        evidence = torch.zeros((self.class_count,h,w),dtype=torch.float32,device=logits.device)
        for source,target in self.mapping.items():
            evidence[target] += source_prob[source]
        top_prob,top_label = evidence.topk(min(self.topk,self.class_count),dim=0)
        confidence = top_prob[0]
        labels = top_label[0].to(torch.uint8)
        valid = torch.as_tensor(np.asarray(view.get('valid_mask',np.ones((h,w),bool))),device=logits.device)
        # A grouped 'other' sum must not invent certainty when individual ADE
        # classes are all weak; both grouped and source confidence are checked.
        trusted = valid & (raw_conf>=self.min_confidence) & (confidence>=self.min_confidence)
        # Sky depth is not a finite observed occupied surface.
        sky_id = next(c['id'] for c in self.registry['classes'] if c['name']=='sky')
        trusted &= labels != sky_id
        labels[~trusted] = UNKNOWN
        confidence[~trusted] = 0.
        top_prob[:,~trusted] = 0.
        output = {'labels':labels.cpu().numpy(),'confidence':confidence.cpu().numpy().astype(np.float32),
            'topk_labels':top_label.permute(1,2,0).cpu().numpy().astype(np.uint8),
            'topk_probabilities':top_prob.permute(1,2,0).cpu().numpy().astype(np.float32),
            'source_labels':raw_label.cpu().numpy().astype(np.uint8),
            'source_confidence':raw_conf.cpu().numpy().astype(np.float32),
            'valid_mask':np.asarray(view.get('valid_mask',np.ones((h,w),bool))),
            'ego_mask':np.asarray(view.get('ego_mask',np.zeros((h,w),bool))),
            'rectification_valid_mask':np.asarray(view.get('rectification_valid_mask',np.ones((h,w),bool))),
            'pixel_domain':'rectified_virtual_pinhole','registry':self.registry['name'],
            'view_id':str(view.get('view_id',view.get('camera_id')))}
        return output

    def predict_views(self,views):
        import torch
        started = time.perf_counter()
        if self.device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats()
        outputs=[]
        for view in views:
            try:
                output = self._predict(view)
            except torch.cuda.OutOfMemoryError:
                if not self.fallback_cpu or not self.device.startswith('cuda'):
                    raise
                self.model.to('cpu');self.device='cpu';torch.cuda.empty_cache()
                output=self._predict(view)
            outputs.append(output)
        self.last_info={'runtime_ms':(time.perf_counter()-started)*1000,'device':self.device,
            'views':len(views),'class_histograms':[class_histogram(o['labels'],self.registry) for o in outputs],
            'peak_allocated_gpu_mib':torch.cuda.max_memory_allocated()/2**20 if self.device.startswith('cuda') else 0}
        return outputs

    def close(self):
        import gc,torch
        self.model.to('cpu')
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def lift_source_semantics(depth_result,semantic_maps,*,registry=None):
    """Lift only each point's source-view/source-UV evidence, preserving unknown."""
    registry = registry or load_registry()
    n = len(depth_result['points']); classes = len(registry['classes'])
    uv = np.asarray(depth_result['source_uv'],np.int32)
    view_ids = np.asarray(depth_result['source_view_ids'],np.int32)
    if uv.shape!=(n,2) or view_ids.shape!=(n,) or len(semantic_maps)!=len(depth_result['source_image_shapes']):
        raise ValueError('Point/source-view provenance shape mismatch')
    labels = np.full(n,UNKNOWN,np.uint8)
    conf = np.zeros(n,np.float32)
    evidence = np.zeros((n,classes),np.float32)
    semantic_uv = np.zeros((n,2),np.int32)
    for vid,sem in enumerate(semantic_maps):
        indices = np.flatnonzero(view_ids==vid)
        if not len(indices):
            continue
        label_map = np.asarray(sem['labels'])
        if label_map.ndim!=2:
            raise ValueError('Semantic labels must be HxW')
        dh,dw = depth_result['source_image_shapes'][vid]
        h,w = label_map.shape
        # OpenCV/torch resizing center convention, explicit domain mapping.
        su = np.floor((uv[indices,0]+.5)*w/dw).astype(int)
        sv = np.floor((uv[indices,1]+.5)*h/dh).astype(int)
        if np.any((su<0)|(su>=w)|(sv<0)|(sv>=h)):
            raise ValueError('Source pixel lies outside its semantic image')
        semantic_uv[indices] = np.stack([su,sv],-1)
        local_labels = label_map[sv,su]
        known = (local_labels !=UNKNOWN)&(local_labels <classes)
        labels[indices] = local_labels
        confidence_map = np.asarray(sem.get('confidence',np.ones((h,w),np.float32)))
        conf[indices] = np.where(known,confidence_map[sv,su],0)
        if 'topk_labels' in sem and 'topk_probabilities' in sem:
            ids = np.asarray(sem['topk_labels'])[sv,su]
            probs = np.asarray(sem['topk_probabilities'])[sv,su]
            for k in range(ids.shape[-1]):
                valid = known & (ids[:,k]<classes)&np.isfinite(probs[:,k])&(probs[:,k]>=0)
                np.add.at(evidence,(indices[valid],ids[valid,k]),probs[valid,k])
        else:
            evidence[indices[known],local_labels[known]] = conf[indices[known]]
        if np.any(known & (evidence[indices].sum(1)<=0)):
            raise ValueError('Known source semantics has no positive evidence')
    if np.any(view_ids<0) or np.any(view_ids>=len(semantic_maps)):
        raise ValueError('Invalid source-view ID')
    diagnostics={'association':'same_source_virtual_view_pixel','cross_camera_reprojection':False,
        'points':n,'labelled_points':int(np.sum(labels!=UNKNOWN)),
        'class_histogram':class_histogram(labels,registry),
        'source_uv_coordinate_system':'integer depth image pixel centers; center-scaled to semantic grid'}
    return {'labels':labels,'evidence':evidence,'semantic_confidence':conf,
            'semantic_uv':semantic_uv,'diagnostics':diagnostics}
