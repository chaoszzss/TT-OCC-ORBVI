import numpy as np
import pytest

from orbvi_occ.camera_depth import prepare_camera_views, predictions_to_points


def test_camera_resizing_updates_intrinsics_and_preserves_fixed_rig_pose():
    image = np.full((400, 800, 3), 128, np.uint8)
    K = np.array([[400, 0, 400], [0, 300, 200], [0, 0, 1]], np.float32)
    pose = np.eye(4, dtype=np.float32); pose[0, 3] = 0.18
    views, auxiliary = prepare_camera_views([{"image": image, "K": K, "T_rig_camera": pose}], 224)
    np.testing.assert_allclose(views[0]["intrinsics"].numpy()[0],
                               [[112, 0, 112], [0, 168, 112], [0, 0, 1]])
    np.testing.assert_array_equal(views[0]["camera_poses"].numpy()[0], pose)
    assert tuple(views[0]["img"].shape) == (1, 3, 224, 224)
    assert "depth_z" not in views[0]
    assert auxiliary[0]["valid_mask"].all()


@pytest.mark.parametrize("key", ["depth_z", "lidar", "points_lidar", "T_world_lidar"])
def test_camera_interface_rejects_depth_and_lidar_information(key):
    descriptor = {"image": np.zeros((200, 200, 3), np.uint8), "K": np.eye(3), key: None}
    with pytest.raises(ValueError, match="forbids"):
        prepare_camera_views([descriptor])


def test_camera_geometry_uses_fixed_extrinsics_and_excludes_invalid_pixels():
    pose = np.eye(4, dtype=np.float32); pose[0, 3] = 0.25
    image = np.zeros((2, 2, 3), np.uint8)
    camera_points = np.array([[[[0, 0, 2], [0, 0, 40]], [[1, 0, 2], [0, 0, -2]]]], np.float32)
    pred = {"pts3d_cam": camera_points, "depth_z": camera_points[..., 2:3],
            "mask": np.ones((1, 2, 2, 1), bool), "conf": np.ones((1, 2, 2))}
    auxiliary = [{"image": image, "valid_mask": np.ones((2, 2), bool), "pose": pose}]
    actual = predictions_to_points([pred], auxiliary, pixel_stride=1, max_range=20)
    np.testing.assert_allclose(actual["points"], [[0.25, 0, 2], [1.25, 0, 2]])
    np.testing.assert_allclose(actual["origins"], [[0.25, 0, 0], [0.25, 0, 0]])
    assert actual["valid_masks"][0].sum() == 2


def test_original_view_mask_remains_effective_after_rectification_resize():
    image = np.full((224, 224, 3), 120, np.uint8)
    valid = np.ones((224, 224), bool); valid[:, 112:] = False
    _, auxiliary = prepare_camera_views([{"image": image, "K": np.eye(3), "valid_mask": valid}])
    np.testing.assert_array_equal(auxiliary[0]["valid_mask"], valid)
