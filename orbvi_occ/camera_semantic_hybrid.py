"""Frozen OpenSeeD-primary semantic ensemble with explicit garage refinements.

The canonical panoptic expert remains intact. ADE20K can fill its unknown
pixels or subtype a predicted wall as a column under fixed independent score
gates. Expert scores are never ranked against each other or fabricated.
"""
from __future__ import annotations
import json
from pathlib import Path
import time
import numpy as np
from .taxonomy import load_registry,class_histogram,UNKNOWN

EXPERT_NAMES={0:'official_OpenSeeD_panoptic',1:'ADE20K_unknown_fill',
              2:'ADE20K_wall_to_column_subtype',255:'semantic_unknown'}


def combine_semantic_experts(official,b5,*,valid_mask=None,ego_mask=None,
                             unknown_fill_confidence=.5,column_refine_confidence=.65,
                             registry=None):
    """Select genuine expert arrays using fixed rules, without modifying inputs."""
    registry=registry or load_registry()
    names={c['name']:c['id'] for c in registry['classes']}
    labels=np.asarray(official['labels'],np.uint8)
    ade_labels=np.asarray(b5['labels'],np.uint8)
    if labels.ndim!=2 or ade_labels.shape!=labels.shape:
        raise ValueError('Both semantic experts must share the same HxW pixel domain')
    h,w=labels.shape
    valid=np.ones((h,w),bool) if valid_mask is None else np.asarray(valid_mask,bool).copy()
    ego=np.zeros((h,w),bool) if ego_mask is None else np.asarray(ego_mask,bool).copy()
    if valid.shape!=labels.shape or ego.shape!=labels.shape:
        raise ValueError('Validity and ego-body masks must match both experts')
    # Both experts must actually accept the shared calibrated RGB location.
    for prediction in (official,b5):
        mask=np.asarray(prediction.get('valid_mask',valid),bool)
        if mask.shape!=labels.shape:
            raise ValueError('Expert validity masks have incompatible pixel domains')
        valid &= mask
    valid &=~ego
    for threshold in (unknown_fill_confidence,column_refine_confidence):
        if not np.isfinite(threshold) or not 0<=threshold<=1:
            raise ValueError('Hybrid expert gate must lie in [0,1]')
    required=('confidence','topk_labels','topk_probabilities')
    for prediction in (official,b5):
        if any(key not in prediction for key in required):
            raise ValueError('Hybrid inference requires true confidence and top-k evidence from both experts')
    official_conf=np.asarray(official['confidence'],np.float32)
    ade_conf=np.asarray(b5['confidence'],np.float32)
    native_ade_conf=np.asarray(b5['source_confidence'],np.float32)
    if any(x.shape!=labels.shape or np.any(~np.isfinite(x)) for x in (official_conf,ade_conf,native_ade_conf)):
        raise ValueError('Expert confidence arrays must be finite HxW scores')
    official_top=np.asarray(official['topk_labels'],np.uint8)
    ade_top=np.asarray(b5['topk_labels'],np.uint8)
    official_prob=np.asarray(official['topk_probabilities'],np.float32)
    ade_prob=np.asarray(b5['topk_probabilities'],np.float32)
    if official_top.shape!=ade_top.shape or official_top.shape[:2]!=labels.shape or official_prob.shape!=official_top.shape or ade_prob.shape!=ade_top.shape:
        raise ValueError('Both experts must produce aligned HxWxK top-k evidence')
    if np.any(~np.isfinite(official_prob)) or np.any(~np.isfinite(ade_prob)) or np.any(official_prob<0) or np.any(ade_prob<0):
        raise ValueError('Expert top-k evidence must be finite nonnegative')
    ade_known=(ade_labels!=UNKNOWN)&(ade_labels<len(registry['classes']))&(ade_labels!=names['sky'])
    fill=valid&(labels==UNKNOWN)&ade_known&(ade_conf>=unknown_fill_confidence)&(native_ade_conf>=unknown_fill_confidence)
    subtype=valid&(labels==names['wall'])&(ade_labels==names['column'])&ade_known&(ade_conf>=column_refine_confidence)&(native_ade_conf>=column_refine_confidence)
    chosen_ade=fill|subtype
    chosen_labels=labels.copy();chosen_conf=official_conf.copy()
    chosen_top=official_top.copy();chosen_prob=official_prob.copy()
    chosen_labels[chosen_ade]=ade_labels[chosen_ade]
    chosen_conf[chosen_ade]=ade_conf[chosen_ade]
    chosen_top[chosen_ade]=ade_top[chosen_ade]
    chosen_prob[chosen_ade]=ade_prob[chosen_ade]
    accepted=valid&(chosen_labels!=UNKNOWN)&(chosen_labels<len(registry['classes']))&(chosen_labels!=names['sky'])
    chosen_labels[~accepted]=UNKNOWN;chosen_conf[~accepted]=0;chosen_prob[~accepted]=0
    experts=np.full((h,w),UNKNOWN,np.uint8)
    experts[accepted]=0;experts[fill&accepted]=1;experts[subtype&accepted]=2
    result={'labels':chosen_labels,'confidence':chosen_conf,'topk_labels':chosen_top,
        'topk_probabilities':chosen_prob,'expert_id':experts,'valid_mask':valid,
        'ego_mask':ego,'registry':registry['name'],'pixel_domain':'rectified_virtual_pinhole',
        'view_id':str(official.get('view_id',b5.get('view_id','unknown'))),
        'rectification_valid_mask':np.asarray(official.get('rectification_valid_mask',valid),bool),
        'expert_names_json':json.dumps(EXPERT_NAMES,sort_keys=True),
        'hybrid_rules_json':json.dumps({'unknown_fill_confidence':unknown_fill_confidence,
            'column_refine_confidence':column_refine_confidence,
            'native_ade_and_mapped_confidence_both_required':True,
            'known_official_labels_preserved_except_wall_to_column_subtyping':True},sort_keys=True)}
    # Source IDs have an explicit expert-dependent native registry. These do
    # not participate in class selection or the source-UV geometric lift.
    official_native=np.asarray(official.get('source_labels',labels),np.uint8)
    b5_native=np.asarray(b5.get('source_labels',ade_labels),np.uint8)
    source_labels=official_native.copy();source_labels[chosen_ade]=b5_native[chosen_ade]
    native_conf=np.asarray(official.get('source_confidence',official_conf),np.float32).copy()
    native_conf[chosen_ade]=native_ade_conf[chosen_ade]
    source_labels[~accepted]=UNKNOWN;native_conf[~accepted]=0
    result.update(source_labels=source_labels,source_confidence=native_conf,
        source_registry_per_expert_json=json.dumps({'0':'garage OpenSeeD vocabulary','1':'ADE20K original IDs','2':'ADE20K original IDs','255':'unknown'},sort_keys=True))
    # Flattened arrays keep archives loadable with allow_pickle=False, and let
    # independent QA recompute every rule without rerunning either network.
    for prefix,prediction in [('official',official),('b5',b5)]:
        for key,value in prediction.items():
            if isinstance(value,np.ndarray):
                result[f'{prefix}_{key}']=value.copy()
    if 'instance_ids' in official:
        result['instance_ids']=np.asarray(official['instance_ids']).copy()
        result['instance_scope']='Official OpenSeeD panoptic IDs only; supplemental ADE pixels do not invent instances'
    if 'instance_segments_json' in official:
        result['instance_segments_json']=str(official['instance_segments_json'])
    if 'panoptic_query_ids' in official:
        result['panoptic_query_ids']=np.asarray(official['panoptic_query_ids']).copy()
    return result


class HybridCameraSemanticProvider:
    """OpenSeeD subprocess plus a frozen local SegFormer ADE20K expert."""
    def __init__(self,model_dir,ade_model_dir,device='cuda',image_size=512,
                 min_confidence=0.,topk=3,provider='hybrid',output_mode='panoptic',
                 unknown_fill_confidence=.5,column_refine_confidence=.65,
                 fallback_cpu=False,ade_device=None,ade_image_size=512,
                 ade_min_confidence=.3,registry=None,python=None,**kwargs):
        from .openseed_provider import OpenSeeDSubprocessProvider
        from .camera_semantic import CameraSemanticProvider
        if provider!='hybrid' or output_mode!='panoptic':
            raise ValueError('Hybrid route requires canonical panoptic OpenSeeD as its primary expert')
        self.registry=registry or load_registry()
        self.unknown_fill_confidence=float(unknown_fill_confidence)
        self.column_refine_confidence=float(column_refine_confidence)
        self.official=OpenSeeDSubprocessProvider(model_dir,device=device,image_size=image_size,
            min_confidence=min_confidence,topk=topk,fallback_cpu=False,python=python,
            output_mode=output_mode,registry=self.registry)
        try:
            self.ade=CameraSemanticProvider(ade_model_dir,device=ade_device or device,
                image_size=ade_image_size,min_confidence=ade_min_confidence,topk=topk,
                fallback_cpu=fallback_cpu,registry=self.registry)
        except Exception:
            self.official.close()
            raise
        self.last_info={}

    def metadata(self):
        official=self.official.metadata();ade=self.ade.metadata()
        return {'provider':'Frozen OpenSeeD-primary garage semantic ensemble',
            'repository':official.get('repository'),'source_commit':official.get('source_commit'),
            'revision':official.get('revision',official.get('source_commit')),
            'checkpoint_sha256':official.get('checkpoint_sha256'),
            'weights_loaded_strict':bool(official.get('weights_loaded_strict') and ade.get('weights_loaded_strict')),
            'trained_on_orbvi':False,'registry':self.registry['name'],'class_count':len(self.registry['classes']),
            'experts':{'official':official,'b5':ade},'expert_ids':EXPERT_NAMES,
            'unknown_fill_confidence':self.unknown_fill_confidence,
            'column_refine_confidence':self.column_refine_confidence,
            'hybrid_selection':'Canonical OpenSeeD panoptic base; fill only unknown with independent native+mapped ADE gate, or subtype wall as ADE column',
            'cross_expert_probability_comparison':False,
            'softmax_is_calibrated_probability':False,
            'original_paper_reproduction':False,
            'method_deviation':'A frozen two-model semantic ensemble extends original TT-Occ OpenSeeD; this is not a complete original-paper reproduction',
            'selected_pixel_evidence':'Entire label/confidence/top-k row copied from its selected true expert',
            'open_vocabulary':True,'ade_device':self.ade.device}

    def predict_views(self,views):
        import torch
        started=time.perf_counter()
        official=self.official.predict_views(views)
        ade=self.ade.predict_views(views)
        outputs=[combine_semantic_experts(o,a,valid_mask=v.get('valid_mask'),ego_mask=v.get('ego_mask'),
            unknown_fill_confidence=self.unknown_fill_confidence,
            column_refine_confidence=self.column_refine_confidence,registry=self.registry)
            for v,o,a in zip(views,official,ade)]
        if len(outputs)!=len(views):
            raise RuntimeError('An expert returned fewer predictions than calibrated views')
        self.last_info={'runtime_ms':(time.perf_counter()-started)*1000,'views':len(views),
            'class_histograms':[class_histogram(x['labels'],self.registry) for x in outputs],
            'expert_histograms':[{EXPERT_NAMES.get(int(k),str(k)):int(n) for k,n in zip(*np.unique(x['expert_id'],return_counts=True))} for x in outputs],
            'official':self.official.last_info,'b5':self.ade.last_info,'ade_device':self.ade.device,
            'cuda_free_mib_after':torch.cuda.mem_get_info()[0]/2**20 if torch.cuda.is_available() else None}
        return outputs

    def close(self):
        try:
            self.ade.close()
        finally:
            self.official.close()
