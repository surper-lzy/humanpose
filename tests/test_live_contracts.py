import numpy as np
import pytest

from rgbd_avatar.camera import CameraIntrinsics
from rgbd_avatar.live import (
    ApplicationExtrinsics,
    LivePosePacket,
    LiveSceneMapper,
    RGBDFrame,
    rotation_zyx_from_degrees,
)
from rgbd_avatar.scene import build_manual_scene_alignment

from test_scene_alignment import _placement


def test_rgbd_frame_requires_depth_aligned_to_rgb() -> None:
    intrinsics = CameraIntrinsics(100.0, 100.0, 2.0, 1.0, 4, 3)
    frame = RGBDFrame(
        rgb_bgr=np.zeros((3, 4, 3), dtype=np.uint8),
        depth_m=np.ones((3, 4), dtype=np.float32),
        intrinsics=intrinsics,
        timestamp_ns=4,
        frame_number=2,
        source_id="replay:test",
    )
    assert frame.depth_m.shape == frame.rgb_bgr.shape[:2]


def test_application_extrinsics_use_documented_zyx_column_convention() -> None:
    rotation = rotation_zyx_from_degrees(0.0, 0.0, 90.0)
    np.testing.assert_allclose(rotation @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_new_camera_extrinsics_form_a_proper_rigid_transform() -> None:
    extrinsics = ApplicationExtrinsics(
        roll_deg=91.51,
        pitch_deg=-179.71,
        yaw_deg=89.95,
        translation_m=np.array([0.0, 0.0, 0.90905]),
    )
    rotation = extrinsics.rotation_application_from_camera
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    transformed = extrinsics.transform_points(np.array([[0.0, 0.0, 0.0]]))
    np.testing.assert_allclose(transformed[0], [0.0, 0.0, 0.90905])


def test_root_locked_mapper_places_feet_at_scene_spawn() -> None:
    alignment = build_manual_scene_alignment(_placement())
    mapper = LiveSceneMapper(
        alignment,
        mode="root_locked",
        rotation_l_from_c=np.eye(3),
    )
    joints = np.full((26, 3), np.nan, dtype=np.float64)
    usable = np.zeros(26, dtype=bool)
    joints[15] = [-0.1, 0.0, 0.0]
    joints[16] = [0.1, 0.0, 0.0]
    joints[19] = [0.0, 0.0, 1.0]
    usable[[15, 16, 19]] = True

    joints_g, anchor_c = mapper.map_joints(joints, usable)

    np.testing.assert_allclose(anchor_c, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(
        0.5 * (joints_g[15] + joints_g[16]),
        alignment.spawn_point_g,
    )
    np.testing.assert_allclose(
        joints_g[19] - alignment.spawn_point_g,
        alignment.scale_g_per_m * alignment.ground_normal_g,
    )


def test_fixed_origin_mapper_preserves_live_translation() -> None:
    alignment = build_manual_scene_alignment(_placement())
    mapper = LiveSceneMapper(
        alignment,
        mode="fixed_origin",
        rotation_l_from_c=np.eye(3),
        origin_camera_m=np.array([1.0, 2.0, 3.0]),
    )
    joints = np.full((26, 3), np.nan, dtype=np.float64)
    usable = np.zeros(26, dtype=bool)
    joints[19] = [1.0, 2.0, 3.0]
    joints[18] = [1.0, 3.0, 3.0]
    usable[[18, 19]] = True

    joints_g, _ = mapper.map_joints(joints, usable)

    np.testing.assert_allclose(joints_g[19], alignment.spawn_point_g)
    np.testing.assert_allclose(
        joints_g[18] - alignment.spawn_point_g,
        alignment.scale_g_per_m * alignment.forward_g,
    )


def test_live_pose_packet_json_round_trip() -> None:
    joints = np.full((26, 3), np.nan, dtype=np.float32)
    usable = np.zeros(26, dtype=bool)
    joints[19] = [1.0, 2.0, 3.0]
    usable[19] = True
    packet = LivePosePacket(
        frame_number=7,
        timestamp_ns=123,
        source_id="camera:serial",
        joints_g=joints,
        confidence=np.linspace(0.0, 1.0, 26, dtype=np.float32),
        usable=usable,
        mapping_mode="root_locked",
    )

    restored = LivePosePacket.from_json_bytes(packet.to_json_bytes())

    assert restored.frame_number == 7
    assert restored.source_id == "camera:serial"
    np.testing.assert_allclose(restored.joints_g[19], [1.0, 2.0, 3.0])
    assert np.isnan(restored.joints_g[0]).all()
