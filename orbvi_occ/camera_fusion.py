"""Causal camera Gaussian occupancy with separate geometry/semantic evidence.

Uses the pinned TT-Occ zero-iteration Gaussian stencil, plus an explicitly
documented camera-rig semantic adaptation. Unknown geometry never casts a
semantic unknown vote. Vehicle classes are retained like other classes;
ray-inconsistent historical surfaces can be removed using observed geometry.
This is not the original RAFT/instance/ICP dynamic tracking implementation.
"""
from __future__ import annotations
from collections import deque
from types import SimpleNamespace
import time
import numpy as np
from scipy.spatial import cKDTree
from .tt_backend import gaussian_offsets, transform_points, UPSTREAM_COMMIT
from .taxonomy import load_registry, UNKNOWN, class_histogram


def _grid_shape(bounds,voxel_size):
    """Use the same double-precision physical grid definition in every path."""
    bounds=np.asarray(bounds,np.float64)
    if bounds.shape!=(2,3) or np.any(~np.isfinite(bounds)) or np.any(bounds[1]<=bounds[0]):
        raise ValueError('Invalid occupancy bounds')
    if not np.isfinite(voxel_size) or voxel_size<=0:
        raise ValueError('Voxel size must be finite and positive')
    return np.ceil((bounds[1]-bounds[0])/np.float64(voxel_size)).astype(np.int64)


def semantic_gaussian_voxelize(points,evidence,*,bounds,voxel_size=.3,sigma=.3,
                               min_confidence=.35,min_support=.05,chunk_size=20000):
    """Sparse TT-Occ stencil with observed class evidence and support metadata."""
    points,evidence = np.asarray(points,np.float32),np.asarray(evidence,np.float32)
    bounds = np.asarray(bounds,np.float64)
    if points.ndim!=2 or points.shape[1]!=3 or evidence.ndim!=2 or len(evidence)!=len(points):
        raise ValueError('Expected points Nx3 and class evidence NxC')
    if np.any(~np.isfinite(evidence)) or np.any(evidence<0):
        raise ValueError('Semantic evidence must be finite nonnegative')
    classes = evidence.shape[1]
    keep = np.isfinite(points).all(1)&(points>=bounds[0]).all(1)&(points<bounds[1]).all(1)
    points,evidence = points[keep],evidence[keep]
    shape = _grid_shape(bounds,voxel_size)
    if not len(points):
        return {'indices':np.empty((0,3),np.int32),'centers':np.empty((0,3),np.float32),
            'labels':np.empty(0,np.uint8),'confidence':np.empty(0,np.float32),
            'known_support':np.empty(0,np.float32),'geometry_support':np.empty(0,np.float32),
            'support_count':np.empty(0,np.int32),'entropy':np.empty(0,np.float32),
            'evidence':np.empty((0,classes),np.float32)}
    offsets,weights = gaussian_offsets(sigma,voxel_size)
    flat_parts,source_parts,weight_parts=[],[],[]
    for start in range(0,len(points),chunk_size):
        xyz = points[start:start+chunk_size,None,:]+offsets[None,:,:]
        valid = (xyz>=bounds[0]).all(-1)&(xyz<bounds[1]).all(-1)
        # A float32 point strictly below upper=12 can yield (x+12)=24
        # after float32 rounding. Compute relative grid coordinates in double
        # so the legitimate point stays in cell 79 instead of becoming 80.
        idx = np.floor((xyz[valid].astype(np.float64)-bounds[0])/np.float64(voxel_size)).astype(np.int64)
        source=np.broadcast_to(np.arange(start,min(start+chunk_size,len(points)))[:,None],valid.shape)[valid]
        weight=np.broadcast_to(weights,valid.shape)[valid]
        # Axis-wise checks must precede flattening. Otherwise Y/Z overflow
        # wraps into another legal flat cell and creates unsupported geometry.
        integer_valid=(idx>=0).all(1)&(idx<shape).all(1)
        idx,source,weight=idx[integer_valid],source[integer_valid],weight[integer_valid]
        flat_parts.append((idx[:,0]*shape[1]+idx[:,1])*shape[2]+idx[:,2])
        source_parts.append(source)
        weight_parts.append(weight)
    flat,source,weight=np.concatenate(flat_parts),np.concatenate(source_parts),np.concatenate(weight_parts)
    unique,inverse=np.unique(flat,return_inverse=True)
    accum=np.zeros((len(unique),classes),np.float32)
    for cls in range(classes):
        accum[:,cls] = np.bincount(inverse,weights=weight*evidence[source,cls],minlength=len(unique))
    geometry_support = np.bincount(inverse,weights=weight,minlength=len(unique)).astype(np.float32)
    known_support=accum.sum(1)
    known_sources = evidence.sum(1)>0
    support_count=np.bincount(inverse,weights=known_sources[source],minlength=len(unique)).astype(np.int32)
    probability=np.divide(accum,known_support[:,None],out=np.zeros_like(accum),where=known_support[:,None]>0)
    confidence=probability.max(1)
    labels=probability.argmax(1).astype(np.uint8)
    labels[(known_support<min_support)|(confidence<min_confidence)] = UNKNOWN
    logp=np.log(np.maximum(probability,1e-12))
    entropy=-(probability*logp).sum(1)/np.log(max(classes,2))
    indices=np.stack([unique//(shape[1]*shape[2]),(unique//shape[2])%shape[1],unique%shape[2]],-1).astype(np.int32)
    centers=bounds[0]+(indices+.5)*voxel_size
    return {'indices':indices,'centers':centers.astype(np.float32),'labels':labels,
        'confidence':confidence.astype(np.float32),'known_support':known_support,
        'geometry_support':geometry_support,'support_count':support_count,
        'entropy':entropy.astype(np.float32),'evidence':accum}


def geometry_collision_voxelize(points,*,bounds,voxel_size=.3,sigma=.3):
    """Use the exact camera occupancy raster for the non-floor source subset.

    A dummy zero-evidence channel keeps this geometry-only callback on the same
    float32 stencil expansion, float64 grid floor and per-axis bounds path.
    Unknown dummy labels have no effect on collision geometry.
    """
    points=np.asarray(points,np.float32)
    raster=semantic_gaussian_voxelize(points,np.zeros((len(points),1),np.float32),
        bounds=bounds,voxel_size=voxel_size,sigma=sigma)
    return raster['indices'],raster['labels']


def _ray_index(current_points,ray_origins=None,min_neighbors=1):
    new=np.asarray(current_points,float)
    origins=np.zeros_like(new) if ray_origins is None else np.asarray(ray_origins,float)
    if origins.shape!=new.shape or not np.isfinite(origins).all():
        raise ValueError('Ray origins must be finite Nx3 and aligned with current points')
    unique,groups=np.unique(origins,axis=0,return_inverse=True)
    index=[]
    for group,origin in enumerate(unique):
        local_new=new[groups==group]-origin
        new_r=np.linalg.norm(local_new,axis=1)
        good=new_r>.25
        local_new,new_r=local_new[good],new_r[good]
        if len(new_r)>=min_neighbors:
            index.append((origin,new_r,cKDTree(local_new/new_r[:,None])))
    return index


def ray_inconsistent_history(past_points,current_points,*,angular_tolerance_deg=1.,range_margin=.6,
                             ray_origins=None,min_range=0.,min_neighbors=1,_prepared_index=None):
    """Conservative current ray contradiction from each physical optical center.

    A past occupied return closer than the current ray's first predicted surface
    can be stale. Uncovered directions and past surfaces behind the current
    return are retained. This does not identify an object's true velocity.
    """
    if not len(past_points) or not len(current_points):
        return np.zeros(len(past_points),bool)
    old=np.asarray(past_points,float)
    index=_ray_index(current_points,ray_origins,min_neighbors) if _prepared_index is None else _prepared_index
    angle_limit=2*np.sin(np.deg2rad(angular_tolerance_deg)/2)
    inconsistent=np.zeros(len(old),bool)
    for origin,new_r,tree in index:
        local_old=old-origin;old_r=np.linalg.norm(local_old,axis=1)
        eligible=np.flatnonzero(old_r>max(.25,min_range))
        if not len(eligible):
            continue
        # The original decision already rejects uncovered directions. Bounding
        # the search by this same angular gate avoids expensive nearest-neighbor
        # search over the opposite camera hemisphere without changing its vote.
        directions=local_old[eligible]/old_r[eligible,None]
        chord,idx=tree.query(directions,k=min_neighbors,
                            distance_upper_bound=np.nextafter(angle_limit,np.inf))
        if min_neighbors==1:
            chord,idx=chord[:,None],idx[:,None]
        # All nearest supporting rays must be in the angular window; use the
        # nearest first-surface range to avoid a distant ray winning at an edge.
        covered=np.all(chord<angle_limit,axis=1)
        if np.any(covered):
            selected=eligible[covered]
            # A bounded miss returns len(new_r) as an index; only fully covered
            # rows may index surface ranges.
            nearest_surface=new_r[idx[covered]].min(1)
            inconsistent[selected] |= old_r[selected]+range_margin<nearest_surface
    return inconsistent


class CameraSemanticFusion:
    def __init__(self,config=None,registry=None):
        cfg=dict(config or {})
        defaults={'voxel_size':.3,'gaussian_sigma':.3,'bounds':[[-12,-12,-2],[12,12,4]],
            'history_seconds':.6,'max_history_frames':6,'history_decay_seconds':.25,
            'input_voxel_size':.12,'min_semantic_confidence':.35,'min_semantic_support':.05,
            'current_priority_min_confidence':.45,'ray_consistency_enabled':True,
            'ray_angular_tolerance_deg':1.,'ray_range_margin':.6}
        defaults.update(ray_min_range=2.,ray_min_neighbors=3)
        defaults.update(cfg)
        self.config=SimpleNamespace(**defaults)
        self.registry=registry or load_registry()
        self.bounds=np.asarray(self.config.bounds,np.float32)
        if self.bounds.shape!=(2,3) or np.any(self.bounds[1]<=self.bounds[0]):
            raise ValueError('Invalid bounds')
        if self.config.history_seconds<0 or self.config.max_history_frames<0 or self.config.history_decay_seconds<=0:
            raise ValueError('Invalid causal history limits')
        gaussian_offsets(self.config.gaussian_sigma,self.config.voxel_size)
        self.history=deque();self.last_stamp_ns=None

    def reset(self):
        self.history.clear();self.last_stamp_ns=None

    def _rasterize(self,points,evidence):
        return semantic_gaussian_voxelize(points,evidence,bounds=self.bounds,
            voxel_size=self.config.voxel_size,sigma=self.config.gaussian_sigma,
            min_confidence=self.config.min_semantic_confidence,min_support=self.config.min_semantic_support)

    def update(self,points,pose,stamp_ns,evidence=None,semantic_confidence=None,
               origins=None,depth_confidence=None,**kwargs):
        started=time.perf_counter();stamp_ns=int(stamp_ns)
        if self.last_stamp_ns is not None and stamp_ns<=self.last_stamp_ns:
            raise ValueError('CameraSemanticFusion requires strictly increasing timestamps')
        points=np.asarray(points,np.float32);pose=np.asarray(pose,float)
        if points.ndim!=2 or points.shape[1]!=3 or pose.shape!=(4,4) or not np.isfinite(pose).all():
            raise ValueError('Expected Nx3 points and finite rigid pose')
        if not np.allclose(pose[3],[0,0,0,1]) or not np.allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3),atol=1e-3):
            raise ValueError('Pose is not rigid')
        classes=len(self.registry['classes'])
        evidence=np.zeros((len(points),classes),np.float32) if evidence is None else np.asarray(evidence,np.float32)
        if evidence.shape!=(len(points),classes):
            raise ValueError('Point/evidence registry shape mismatch')
        valid=np.isfinite(points).all(1)
        if semantic_confidence is not None:
            sem_conf=np.asarray(semantic_confidence,np.float32)
            if sem_conf.shape!=(len(points),) or np.any(~np.isfinite(sem_conf)) or np.any((sem_conf<0)|(sem_conf>1)):
                raise ValueError('Semantic confidence must be finite N scores in [0,1]')
            # Top-k evidence already contains soft confidence. Multiplying
            # again would double-weight uncertainty. Zero-confidence is unknown.
            evidence=evidence.copy();evidence[sem_conf<=0]=0
        if depth_confidence is not None:
            depth_conf=np.asarray(depth_confidence,np.float32)
            if depth_conf.shape!=(len(points),) or np.any(~np.isfinite(depth_conf)):
                raise ValueError('Depth confidence must be finite and aligned')
        ray_origins=None if origins is None else np.asarray(origins,np.float32)
        if ray_origins is not None and ray_origins.shape!=points.shape:
            raise ValueError('Ray origins must match point shape')
        points,evidence=points[valid],evidence[valid]
        if ray_origins is not None:
            ray_origins=ray_origins[valid]
        if np.any(~np.isfinite(evidence)) or np.any(evidence<0):
            raise ValueError('Invalid evidence')
        current=self._rasterize(points,evidence)
        cutoff=stamp_ns-int(self.config.history_seconds*1e9)
        while self.history and self.history[0]['stamp_ns']<cutoff:
            self.history.popleft()
        inverse=np.linalg.inv(pose)
        past_points,past_evidence=[],[]
        ray_removed=0
        current_ray_index=_ray_index(points,ray_origins,self.config.ray_min_neighbors) if self.config.ray_consistency_enabled and self.history else None
        for past in self.history:
            if past['stamp_ns']>=stamp_ns:
                raise RuntimeError('Future acquisition entered history')
            local=transform_points(past['points_world'],inverse).astype(np.float32)
            keep=(local>=self.bounds[0]).all(1)&(local<self.bounds[1]).all(1)
            if self.config.ray_consistency_enabled:
                stale=ray_inconsistent_history(local,points,
                    angular_tolerance_deg=self.config.ray_angular_tolerance_deg,
                    range_margin=self.config.ray_range_margin,ray_origins=ray_origins,
                    min_range=self.config.ray_min_range,min_neighbors=self.config.ray_min_neighbors,
                    _prepared_index=current_ray_index)
                ray_removed+=int(np.sum(keep&stale));keep &=~stale
            if np.any(keep):
                age=(stamp_ns-past['stamp_ns'])/1e9
                decay=np.exp(-age/self.config.history_decay_seconds)
                past_points.append(local[keep]);past_evidence.append(past['evidence'][keep]*decay)
        fused_points=np.concatenate([points]+past_points) if past_points else points
        fused_evidence=np.concatenate([evidence]+past_evidence) if past_points else evidence
        fused=self._rasterize(fused_points,fused_evidence) if past_points else {k:v.copy() for k,v in current.items()}
        # Prevent repeated old dense wall/floor votes from suppressing an
        # actually observed current vehicle. This is class independent, and its
        # conflict count and effective evidence are recorded explicitly.
        overrides=0
        if past_points and len(current['indices']):
            shape=_grid_shape(self.bounds,self.config.voxel_size)
            flatten=lambda x:(x[:,0]*shape[1]+x[:,1])*shape[2]+x[:,2]
            current_flat,fused_flat=flatten(current['indices']),flatten(fused['indices'])
            pos=np.searchsorted(fused_flat,current_flat)
            is_current=(current['labels']!=UNKNOWN)&(current['confidence']>=self.config.current_priority_min_confidence)
            mismatch=is_current&(fused['labels'][pos]!=current['labels'])
            overrides=int(mismatch.sum())
            # Retain current semantic evidence while keeping accumulated
            # geometry support. Confidence/entropy never contradict its label.
            if overrides:
                for field in ('labels','confidence','known_support','support_count','entropy','evidence'):
                    fused[field][pos[mismatch]]=current[field][mismatch]
        # Downsampling affects history storage only; current inference/lifting
        # remains exact. Pick the strongest known evidence within each cell,
        # avoiding 'first unknown wins' starvation.
        chosen=np.arange(len(points))
        if self.config.input_voxel_size>0 and len(points):
            keys=np.floor(points/self.config.input_voxel_size).astype(np.int64)
            order=np.lexsort((-evidence.max(1),keys[:,2],keys[:,1],keys[:,0]))
            sorted_keys=keys[order]
            chosen=order[np.r_[True,np.any(sorted_keys[1:]!=sorted_keys[:-1],axis=1)]]
        if self.config.history_seconds>0 and self.config.max_history_frames>0:
            self.history.append({'stamp_ns':stamp_ns,'points_world':transform_points(points[chosen],pose).astype(np.float32),
                'evidence':evidence[chosen].copy()})
            while len(self.history)>self.config.max_history_frames:
                self.history.popleft()
        self.last_stamp_ns=stamp_ns
        out={}
        for prefix,raster in [('current',current),('fused',fused)]:
            for key,value in raster.items():
                out[f'{prefix}_{key}']=value
        out.update(indices=fused['indices'],labels=fused['labels'],centers=fused['centers'],
            current_source_points=points,current_source_labels=np.where(evidence.sum(1)>0,evidence.argmax(1),UNKNOWN).astype(np.uint8),
            fused_source_points=fused_points,
            diagnostics={'algorithm':'TT-Occ zero-iteration Gaussian camera semantic adaptation',
                'upstream_commit':UPSTREAM_COMMIT,'iterations':0,'input_points':len(points),
                'past_frames':len(past_points),'past_points':sum(map(len,past_points)),
                'current_occupied_voxels':len(current['indices']),'fused_occupied_voxels':len(fused['indices']),
                'new_voxels_from_history':len(fused['indices'])-len(current['indices']),
                'current_classes':class_histogram(current['labels'],self.registry),
                'fused_classes':class_histogram(fused['labels'],self.registry),
                'ray_inconsistent_history_points_removed':ray_removed,'current_semantic_priority_overrides':overrides,
                'gaussian_sigma_m':self.config.gaussian_sigma,'effective_gaussian_sigma_m':int(np.clip(np.rint(np.float32(self.config.gaussian_sigma/self.config.voxel_size+1e-3)),0,3))*self.config.voxel_size,
                'voxel_size_m':self.config.voxel_size,'history_seconds':self.config.history_seconds,
                'grid_indexing':'float64 relative-coordinate floor with per-axis integer bounds before flattening',
                'history_decay_seconds':self.config.history_decay_seconds,
                'depth_confidence_weighting':'No numerical weighting: official depth model confidence mask already filters accepted source pixels; raw confidence is uncalibrated',
                'semantic_confidence_weighting':'Soft top-k evidence used once; zero-confidence semantic observations have zero evidence',
                'ray_origins':'calibrated current physical optical centers' if ray_origins is not None else 'rig-centered approximation',
                'ray_min_range_m':self.config.ray_min_range,'ray_min_neighbors':self.config.ray_min_neighbors,
                'ray_query_policy':'Exact search bounded by the same angular gate; uncovered/near points cannot cast a removal vote',
                'dynamic_tracking':'No RAFT/instance velocity; short decayed history plus predicted-ray consistency. Parked cars are retained.',
                'semantics_255':'Unknown semantic label never contributes class evidence',
                'runtime_ms':(time.perf_counter()-started)*1000})
        return out
