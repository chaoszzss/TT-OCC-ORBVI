import numpy as np
from orbvi_occ.projection import PinholeRectifier,project_triple_sphere,transform_points,label_lidar_points

PARAMS=[620.,624.,646.,496.,-.202,-.066,.592,0.,0.]

def test_optical_axis_and_invalid_origin():
    uv,valid=project_triple_sphere(np.array([[0.,0.,1.],[0.,0.,0.]]),PARAMS)
    np.testing.assert_allclose(uv[0],[646.,496.])
    assert valid.tolist()==[True,False]

def test_projection_has_metric_scale_invariance():
    p=np.array([[1.,.2,3.],[-2.,.5,2.]])
    np.testing.assert_allclose(project_triple_sphere(p,PARAMS)[0],project_triple_sphere(p*12,PARAMS)[0])

def test_known_pinhole_depth_and_transform():
    r=PinholeRectifier(PARAMS,size=(5,5))
    p,uv=r.lift(np.full((5,5),2.),stride=1)
    center=p[(uv==[2,2]).all(1)][0]
    np.testing.assert_allclose(center,[0,0,2])
    t=np.eye(4);t[:3,3]=[1,2,3]
    np.testing.assert_allclose(transform_points(center[None],t),[[1,2,5]])

def test_unvalidated_sensor_calibration_cannot_assign_labels():
    labels,info=label_lidar_points(np.ones((2,3)),{}, {},np.eye(4),validated=False)
    assert (labels==255).all() and info['labelled_points']==0
