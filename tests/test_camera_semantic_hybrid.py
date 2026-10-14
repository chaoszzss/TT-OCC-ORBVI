import numpy as np
from orbvi_occ.camera_semantic_hybrid import combine_semantic_experts


def expert(labels,confidence,native_conf=None):
    labels=np.asarray([labels],np.uint8);conf=np.asarray([confidence],np.float32)
    top=np.repeat(labels[:,:,None],3,axis=2)
    probs=np.zeros_like(top,dtype=np.float32);probs[:,:,0]=conf
    return {'labels':labels,'confidence':conf,'source_confidence':conf.copy() if native_conf is None else np.asarray([native_conf],np.float32),
        'topk_labels':top,'topk_probabilities':probs,'source_labels':labels.copy(),
        'valid_mask':np.ones(labels.shape,bool),'view_id':'3_v0'}


def test_fixed_rules_fill_unknown_subtype_wall_preserve_other_official_classes():
    official=expert([255,0,4,1,255],[0,.6,.7,.6,0])
    ade=expert([1,3,0,4,4],[.8,.8,.99,.99,.49])
    saved={k:v.copy() for k,v in official.items() if isinstance(v,np.ndarray)}
    result=combine_semantic_experts(official,ade)
    np.testing.assert_array_equal(result['labels'],[[1,3,4,1,255]])
    np.testing.assert_array_equal(result['expert_id'],[[1,2,0,0,255]])
    assert result['confidence'][0,0]==ade['confidence'][0,0]
    assert result['confidence'][0,2]==official['confidence'][0,2]
    for key,value in saved.items():
        np.testing.assert_array_equal(official[key],value)


def test_native_and_mapped_confidence_gates_both_required():
    official=expert([255,0,0],[0,.6,.6])
    ade=expert([1,3,3],[.9,.9,.64],native_conf=[.49,.64,.9])
    result=combine_semantic_experts(official,ade)
    np.testing.assert_array_equal(result['labels'],[[255,0,0]])


def test_ego_and_invalid_exclusion_cannot_become_geometry_or_semantic_known():
    official=expert([0,255,0],[.8,0,.7]);ade=expert([3,1,3],[.99,.99,.99])
    result=combine_semantic_experts(official,ade,valid_mask=np.array([[True,False,True]]),ego_mask=np.array([[True,False,False]]))
    np.testing.assert_array_equal(result['labels'],[[255,255,3]])
    assert np.all(result['topk_probabilities'][0,:2]==0)
    assert np.all(result['confidence'][0,:2]==0)
    assert 'official_labels' in result and 'b5_labels' in result


def test_selected_topk_is_copied_whole_without_mixing_expert_scores_or_classes():
    official=expert([0,255],[.7,0]);ade=expert([3,4],[.9,.8])
    official['topk_labels'][:]=np.array([[[0,1,2],[0,1,2]]])
    ade['topk_labels'][:]=np.array([[[3,0,2],[4,8,1]]])
    result=combine_semantic_experts(official,ade)
    np.testing.assert_array_equal(result['topk_labels'],ade['topk_labels'])
    np.testing.assert_array_equal(result['topk_probabilities'],ade['topk_probabilities'])
