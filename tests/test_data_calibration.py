from pathlib import Path
import unittest

import numpy as np

from scipy.spatial.transform import Rotation
from orbvi_occ.calibration import deskew_scan, icp_point_to_plane, inverse_transform, transform_points


class CalibrationTests(unittest.TestCase):
    def test_gravity_constrained_registration_recovers_yaw_translation(self):
        axis = np.linspace(-2,2,21)
        a,b = np.meshgrid(axis,axis)
        a,b = a.ravel(),b.ravel()
        target = np.concatenate([np.column_stack([a,b,np.full(len(a),3.)]),
                                 np.column_stack([a,np.full(len(a),4.),b]),
                                 np.column_stack([np.full(len(a),5.),a,b])])
        prior = Rotation.from_euler('xy',[.05,-.03]).as_matrix()
        expected = np.eye(4)
        expected[:3,:3] = Rotation.from_euler('z',.04).as_matrix()@prior
        expected[:3,3] = [.15,-.1,.08]
        source = transform_points(target,inverse_transform(expected))
        initial = np.eye(4); initial[:3,:3]=prior
        estimated,quality = icp_point_to_plane(source,target,initial,rotation_prior=prior)
        self.assertEqual(quality['rank'],4)
        np.testing.assert_allclose(estimated,expected,atol=1e-3)

    def test_runtime_vio_rotation_is_causal(self):
        root = Path(__file__).resolve().parents[1] / 'prepared'
        path = root / 'vio_rotation_prior.npz'
        if not path.exists():
            self.skipTest('Rotation calibration artifact has not been prepared')
        with np.load(path) as data:
            self.assertEqual(len(data['stamp_ns']),2348)
            self.assertTrue((data['vio_stamp_ns']<=data['stamp_ns']).all())

    def test_first_scan_has_no_future_pose_dependency(self):
        poses = np.repeat(np.eye(4)[None], 3, axis=0)
        points = np.array([[2.,0.,0.], [3.,0.,0.]])
        offsets = np.array([0, 50], dtype=np.int64)
        trajectory = {'stamp_ns': np.array([100,200,300]), 'T_world_lidar': poses}
        before = deskew_scan(points, offsets, 50, 100, trajectory)
        poses[1:, :3, 3] = [[50, 20, 80], [80, 50, 80]]
        after = deskew_scan(points, offsets, 50, 100, trajectory)
        np.testing.assert_array_equal(before, points)
        np.testing.assert_array_equal(after, before)

    def test_scan_end_deskew_uses_actual_ray_origin(self):
        poses = np.repeat(np.eye(4)[None], 3, axis=0)
        poses[:,0,3] = [0., 1., 50.]
        trajectory = {'stamp_ns': np.array([100,200,300]), 'T_world_lidar': poses}
        origins = deskew_scan(np.zeros((2,3)), [0,100], 100, 200, trajectory)
        np.testing.assert_allclose(origins, [[-1.,0.,0.], [0.,0.,0.]])
        poses[2,:3,3] = [100.,100.,100.]
        np.testing.assert_allclose(deskew_scan(np.zeros((2,3)), [0,100], 100, 200, trajectory), origins)

    def test_complete_lidar_pose_contract(self):
        path = Path(__file__).resolve().parents[1] / 'prepared' / 'trajectory.npz'
        if not path.exists():
            self.skipTest('Full registration artifact has not been prepared')
        with np.load(path) as data:
            poses = data['T_world_lidar']
            self.assertEqual(poses.shape, (2348,4,4))
            self.assertTrue(np.isfinite(poses).all())
            np.testing.assert_allclose(poses[:,3], np.repeat([[0.,0.,0.,1.]],2348,axis=0))
            rotations = poses[:,:3,:3]
            np.testing.assert_allclose(np.einsum('nji,njk->nik',rotations,rotations),
                                       np.repeat(np.eye(3)[None],2348,axis=0), atol=1e-10)
            self.assertTrue((np.diff(data['stamp_ns'])>0).all())


if __name__ == '__main__':
    unittest.main()
