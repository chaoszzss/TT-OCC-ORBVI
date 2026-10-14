import numpy as np
import pytest
from orbvi_occ.camera_semantic import lift_source_semantics,VirtualPinhole,build_virtual_views
from orbvi_occ.taxonomy import load_registry,class_mapping,palette
from orbvi_occ.camera_depth import predictions_to_points,prepare_camera_views


def test_same_source_uv_cannot_be_taken_over_by_another_view():
    depth={'points':np.zeros((2,3)),'source_uv':np.array([[0,0],[1,0]]),
        'source_view_ids':np.array([0,1]),'source_image_shapes':[[1,2],[1,2]]}
    sem=[{'labels':np.array([[4,1]],np.uint8),'confidence':np.ones((1,2))},
         {'labels':np.array([[0,3]],np.uint8),'confidence':np.ones((1,2))}]
    lifted=lift_source_semantics(depth,sem)
    np.testing.assert_array_equal(lifted['labels'],[4,3])
    assert lifted['evidence'][0,4]==1
    assert lifted['evidence'][0,0]==0


def test_resize_pixel_centers_and_unknown_preserve_zero_evidence():
    depth={'points':np.zeros((2,3)),'source_uv':np.array([[0,0],[1,1]]),
        'source_view_ids':np.zeros(2,int),'source_image_shapes':[[2,2]]}
    labels=np.zeros((4,4),np.uint8);labels[1,1]=4;labels[3,3]=255
    lifted=lift_source_semantics(depth,[{'labels':labels,'confidence':np.full((4,4),.7)}])
    np.testing.assert_array_equal(lifted['semantic_uv'],[[1,1],[3,3]])
    assert lifted['labels'][1]==255
    assert lifted['evidence'][1].sum()==0
    assert lifted['semantic_confidence'][1]==0


def test_named_registry_ade_wall_id_zero_is_not_nuscenes_noise():
    registry=load_registry();mapping=class_mapping({0:'wall',20:'car',42:'column',5:'ceiling'},registry)
    assert mapping=={0:0,20:4,42:3,5:2}
    np.testing.assert_array_equal(palette(registry)[4],[0,150,245])
    assert registry['unknown_id']==255


def test_depth_source_xyz_uv_and_fixed_pose_are_independently_reconstructible():
    xyz=np.array([[[[0,0,2],[1,0,2]],[[0,1,2],[1,1,2]]]],np.float32)
    pose=np.eye(4);pose[0,3]=.1
    pred={'pts3d_cam':xyz,'depth_z':xyz[...,2:3],'conf':np.ones((1,2,2)),'mask':np.ones((1,2,2,1),bool)}
    views=[{'image':np.zeros((2,2,3),np.uint8),'pose':pose,'valid_mask':np.ones((2,2),bool),'K':np.eye(3)}]
    result=predictions_to_points([pred],views,pixel_stride=1)
    np.testing.assert_array_equal(result['source_uv'],[[0,0],[1,0],[0,1],[1,1]])
    np.testing.assert_allclose(result['source_camera_points']+.1*np.array([1,0,0]),result['points'])
    assert result['source_image_shapes']==[[2,2]]


def test_virtual_yaw_has_correct_rigid_calibration_without_extra_physical_sensor():
    camera={'params':[620,624,645,530,-.2,-.06,.59,0,0],'width':1280,'height':1080,'T_camera0_camera':np.eye(4).tolist()}
    cfg={'cameras':{'3':camera},'camera_order':['3']}
    views=build_virtual_views({'3':np.zeros((1080,1280,3),np.uint8)},cfg,
        {'size':[112,112],'fov_deg':90,'angles':[[0,0],[30,0]]})
    assert len(views)==2
    assert views[0]['camera_id']==views[1]['camera_id']=='3'
    R=views[1]['T_rig_camera'][:3,:3]
    np.testing.assert_allclose(R.T@R,np.eye(3),atol=1e-9)
    assert not np.allclose(views[0]['T_rig_camera'],views[1]['T_rig_camera'])


def test_bad_provenance_is_rejected():
    depth={'points':np.zeros((1,3)),'source_uv':np.array([[10,0]]),
        'source_view_ids':np.array([0]),'source_image_shapes':[[2,2]]}
    with pytest.raises(ValueError,match='outside'):
        lift_source_semantics(depth,[{'labels':np.zeros((2,2),np.uint8)}])


def test_ego_body_polygon_masks_evidence_and_preserves_unmodified_rgb():
    camera={'params':[620,624,645,530,-.2,-.06,.59,0,0],'width':1280,'height':1080,'T_camera0_camera':np.eye(4).tolist()}
    cfg={'cameras':{'3':camera},'camera_order':['3']}
    image=np.full((1080,1280,3),128,np.uint8)
    geometry={'size':[112,112],'fov_deg':90,'angles':[[0,0]]}
    mask={'coordinate_domain':'normalized_virtual_pinhole','virtual_views':geometry,'padding_px':0,
          'views':{'3_v0':{'polygon':[[0,.8],[1,.8],[1,1],[0,1]]}}}
    original=build_virtual_views({'3':image},cfg,geometry)[0]
    masked=build_virtual_views({'3':image},cfg,{**geometry,'ego_mask_config':mask})[0]
    np.testing.assert_array_equal(original['image'],masked['image'])
    assert masked['ego_mask'][105,56]
    assert not masked['valid_mask'][105,56]
    assert masked['valid_mask'][50,56]
    with pytest.raises(ValueError,match='different'):
        build_virtual_views({'3':image},cfg,{**geometry,'fov_deg':100,'ego_mask_config':mask})


def test_new_mask_pixel_center_sampling_matches_semantic_lift_at_body_boundary():
    image=np.zeros((518,518,3),np.uint8)
    mask=np.ones((518,518),bool);mask[390:]=False;mask[:,201:203]=False
    _,aux=prepare_camera_views([{'image':image,'K':np.eye(3),'valid_mask':mask,
        'validity_sampling':'pixel_center'}],336)
    v,u=np.meshgrid(np.arange(336),np.arange(336),indexing='ij')
    sy=np.floor((v+.5)*518/336).astype(int);sx=np.floor((u+.5)*518/336).astype(int)
    np.testing.assert_array_equal(aux[0]['valid_mask'],mask[sy,sx])
    assert aux[0]['validity_sampling']=='pixel_center'
    # Old default remains explicitly legacy: no changed selection in old runs.
    _,legacy=prepare_camera_views([{'image':image,'K':np.eye(3),'valid_mask':mask}],336)
    assert legacy[0]['validity_sampling']=='legacy_opencv_nearest'
    assert not np.array_equal(legacy[0]['valid_mask'],aux[0]['valid_mask'])
