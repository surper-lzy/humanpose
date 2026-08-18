import numpy as np
import pytest

from rgbd_avatar.camera import CameraIntrinsics
from rgbd_avatar.depth import (
    PointCloudRecoveryConfig,
    depth_to_organized_point_cloud,
    recover_pose3d,
    recover_pose3d_from_point_cloud,
)
from rgbd_avatar.pose import HALPE26_NAMES, Pose2D


TORSO_INDICES = (5, 6, 11, 12, 18, 19)


def make_intrinsics(width: int = 40, height: int = 40) -> CameraIntrinsics:
    return CameraIntrinsics(
        fx=100.0,
        fy=100.0,
        cx=width / 2,
        cy=height / 2,
        width=width,
        height=height,
    )


def make_pose(
    *,
    bbox: tuple[float, float, float, float] = (4, 4, 36, 36),
    target_index: int = 10,
    target_uv: tuple[float, float] = (20, 20),
    torso_uv: tuple[float, float] | None = (14, 18),
) -> Pose2D:
    count = len(HALPE26_NAMES)
    keypoints = np.full((count, 2), target_uv, dtype=np.float32)
    scores = np.zeros(count, dtype=np.float32)
    scores[target_index] = 0.9
    if torso_uv is not None:
        for index in TORSO_INDICES:
            keypoints[index] = torso_uv
            scores[index] = 0.9
    return Pose2D(
        keypoints=keypoints,
        scores=scores,
        bbox_xyxy=np.asarray(bbox, dtype=np.float32),
        bbox_score=0.9,
    )


def make_face_boundary_pose(
    *,
    ear_uv: tuple[float, float] = (24, 19),
) -> Pose2D:
    count = len(HALPE26_NAMES)
    keypoints = np.full((count, 2), (18, 20), dtype=np.float32)
    scores = np.zeros(count, dtype=np.float32)
    for index, uv in {
        0: (18, 17),
        1: (19, 18),
        2: (17, 18),
        3: ear_uv,
    }.items():
        keypoints[index] = uv
        scores[index] = 0.9
    for index in TORSO_INDICES:
        keypoints[index] = (18, 28)
        scores[index] = 0.9
    return Pose2D(
        keypoints=keypoints,
        scores=scores,
        bbox_xyxy=np.asarray((4, 4, 36, 36), dtype=np.float32),
        bbox_score=0.9,
    )


def fixed_radius_config(
    radius: int = 3,
    **overrides: object,
) -> PointCloudRecoveryConfig:
    values: dict[str, object] = {
        "min_radius_px": radius,
        "max_radius_px": radius,
        "expanded_max_radius_px": radius,
        "bbox_padding_ratio": 0.0,
    }
    values.update(overrides)
    return PointCloudRecoveryConfig(**values)


def test_uniform_point_cloud_recovers_ray_consistent_joint() -> None:
    intrinsics = make_intrinsics()
    depth = np.full((40, 40), 2.0, dtype=np.float32)
    points = depth_to_organized_point_cloud(depth, intrinsics)
    pose = make_pose(target_uv=(23.5, 17.25), torso_uv=(14, 18))

    result = recover_pose3d_from_point_cloud(
        pose,
        points,
        intrinsics,
        config=fixed_radius_config(),
    )
    joint = result.pose3d.joints_m[10]
    projected_u = intrinsics.fx * joint[0] / joint[2] + intrinsics.cx
    projected_v = intrinsics.fy * joint[1] / joint[2] + intrinsics.cy

    assert result.pose3d.valid[10]
    assert np.isclose(result.pose3d.depth_m[10], 2.0)
    np.testing.assert_allclose(
        [projected_u, projected_v],
        pose.keypoints[10],
        atol=1e-5,
    )
    assert result.diagnostics["method"] == "pointcloud_cluster"
    assert result.diagnostics["person_filter"]["person_depth_prior_m"] == 2.0


def test_point_cloud_recovery_can_limit_expensive_joint_search() -> None:
    intrinsics = make_intrinsics()
    depth = np.full((40, 40), 2.0, dtype=np.float32)
    points = depth_to_organized_point_cloud(depth, intrinsics)
    pose = make_pose(target_uv=(23.5, 17.25), torso_uv=(14, 18))

    result = recover_pose3d_from_point_cloud(
        pose,
        points,
        intrinsics,
        config=fixed_radius_config(),
        joint_indices=(10,),
    )

    assert result.pose3d.valid[10]
    assert np.count_nonzero(result.pose3d.valid) == 1
    assert result.diagnostics["requested_joint_indices"] == [10]
    assert result.diagnostics["joints"][5]["status"] == "not_requested"


def test_point_cloud_recovery_reuses_supplied_person_depth_hint() -> None:
    intrinsics = make_intrinsics()
    depth = np.full((40, 40), 2.0, dtype=np.float32)
    points = depth_to_organized_point_cloud(depth, intrinsics)

    result = recover_pose3d_from_point_cloud(
        make_pose(target_uv=(23.5, 17.25), torso_uv=(14, 18)),
        points,
        intrinsics,
        config=fixed_radius_config(),
        joint_indices=(10,),
        person_depth_hint_m=2.0,
    )

    person_filter = result.diagnostics["person_filter"]
    assert result.pose3d.valid[10]
    assert person_filter["person_depth_source"] == "provided_hint"
    assert person_filter["torso_seed_count"] == 0


def test_point_cloud_recovery_rejects_invalid_joint_subset() -> None:
    intrinsics = make_intrinsics()
    depth = np.full((40, 40), 2.0, dtype=np.float32)
    points = depth_to_organized_point_cloud(depth, intrinsics)

    with pytest.raises(ValueError, match="within"):
        recover_pose3d_from_point_cloud(
            make_pose(),
            points,
            intrinsics,
            joint_indices=(26,),
        )


def test_person_depth_proxy_rejects_background_only_joint() -> None:
    intrinsics = make_intrinsics()
    depth = np.full((40, 40), 4.0, dtype=np.float32)
    depth[8:31, 8:19] = 2.0
    points = depth_to_organized_point_cloud(depth, intrinsics)
    pose = make_pose(
        target_uv=(30, 20),
        torso_uv=(14, 18),
        bbox=(5, 5, 35, 35),
    )

    baseline = recover_pose3d(
        pose,
        depth,
        intrinsics,
        radius=3,
    )
    clustered = recover_pose3d_from_point_cloud(
        pose,
        points,
        intrinsics,
        config=fixed_radius_config(),
    )

    assert baseline.valid[10]
    assert np.isclose(baseline.depth_m[10], 4.0)
    assert not clustered.pose3d.valid[10]
    diagnostic = clustered.diagnostics["joints"][10]
    assert diagnostic["status"] == "no_valid_depth"


def test_cluster_recovers_nearby_person_surface_when_center_is_background() -> None:
    intrinsics = make_intrinsics()
    depth = np.full((40, 40), 4.0, dtype=np.float32)
    depth[8:31, 8:20] = 2.0
    points = depth_to_organized_point_cloud(depth, intrinsics)
    pose = make_pose(
        target_uv=(20.5, 20),
        torso_uv=(14, 18),
        bbox=(5, 5, 35, 35),
    )

    result = recover_pose3d_from_point_cloud(
        pose,
        points,
        intrinsics,
        config=fixed_radius_config(radius=4),
    )

    assert result.pose3d.valid[10]
    assert np.isclose(result.pose3d.depth_m[10], 2.0)
    diagnostic = result.diagnostics["joints"][10]
    assert diagnostic["surface_medoid_uv"][0] <= 19
    assert diagnostic["center_distance_px"] > 0


def test_patch_expands_only_when_base_radius_has_no_supported_surface() -> None:
    intrinsics = make_intrinsics()
    depth = np.full((40, 40), 4.0, dtype=np.float32)
    depth[8:31, 8:21] = 2.0
    points = depth_to_organized_point_cloud(depth, intrinsics)
    pose = make_pose(
        target_uv=(25, 20),
        torso_uv=(14, 18),
        bbox=(5, 5, 35, 35),
    )
    config = PointCloudRecoveryConfig(
        min_radius_px=3,
        max_radius_px=3,
        expansion_factor=2.0,
        expanded_max_radius_px=6,
        bbox_padding_ratio=0.0,
    )

    result = recover_pose3d_from_point_cloud(
        pose,
        points,
        intrinsics,
        config=config,
    )

    assert result.pose3d.valid[10]
    assert np.isclose(result.pose3d.depth_m[10], 2.0)
    diagnostic = result.diagnostics["joints"][10]
    assert diagnostic["expanded_radius"]
    assert diagnostic["radius_px"] == 6


def test_supplied_person_mask_filters_same_depth_connected_background() -> None:
    intrinsics = make_intrinsics()
    depth = np.full((40, 40), 2.0, dtype=np.float32)
    points = depth_to_organized_point_cloud(depth, intrinsics)
    pose = make_pose(target_uv=(20, 20), torso_uv=(14, 18))
    person_mask = np.zeros((40, 40), dtype=bool)
    person_mask[8:31, 8:22] = True

    result = recover_pose3d_from_point_cloud(
        pose,
        points,
        intrinsics,
        config=fixed_radius_config(),
        person_mask=person_mask,
    )

    assert result.pose3d.valid[10]
    person_filter = result.diagnostics["person_filter"]
    assert person_filter["type"] == "provided_mask_bbox_depth_band"
    assert person_filter["external_mask_supplied"]

    empty_result = recover_pose3d_from_point_cloud(
        pose,
        points,
        intrinsics,
        config=fixed_radius_config(),
        person_mask=np.zeros((40, 40), dtype=bool),
    )
    assert not empty_result.pose3d.valid[10]
    assert (
        empty_result.diagnostics["joints"][10]["status"]
        == "no_valid_depth"
    )


def test_keypoint_outside_bbox_uses_recorded_depth_band_fallback() -> None:
    intrinsics = make_intrinsics()
    depth = np.full((40, 40), 4.0, dtype=np.float32)
    depth[8:31, 8:19] = 2.0
    depth[19:22, 24:27] = 2.0
    points = depth_to_organized_point_cloud(depth, intrinsics)
    pose = make_pose(
        target_uv=(25, 20),
        torso_uv=(14, 18),
        bbox=(8, 8, 19, 31),
    )

    result = recover_pose3d_from_point_cloud(
        pose,
        points,
        intrinsics,
        config=fixed_radius_config(),
    )

    assert result.pose3d.valid[10]
    assert np.isclose(result.pose3d.depth_m[10], 2.0)
    assert result.diagnostics["joints"][10]["bbox_mismatch"]


def test_no_torso_prior_still_uses_local_point_cloud_cluster() -> None:
    intrinsics = make_intrinsics()
    depth = np.full((40, 40), 2.5, dtype=np.float32)
    points = depth_to_organized_point_cloud(depth, intrinsics)
    pose = make_pose(target_uv=(20, 20), torso_uv=None)

    result = recover_pose3d_from_point_cloud(
        pose,
        points,
        intrinsics,
        config=fixed_radius_config(),
    )

    assert result.pose3d.valid[10]
    assert np.isclose(result.pose3d.depth_m[10], 2.5)
    assert (
        result.diagnostics["person_filter"]["person_depth_prior_m"] is None
    )


def test_small_point_cloud_component_is_rejected() -> None:
    intrinsics = make_intrinsics()
    points = np.full((40, 40, 3), np.nan, dtype=np.float32)
    points[20, 20] = [0.0, 0.0, 2.0]
    points[20, 21] = [0.02, 0.0, 2.0]
    pose = make_pose(target_uv=(20, 20), torso_uv=None)

    result = recover_pose3d_from_point_cloud(
        pose,
        points,
        intrinsics,
        config=fixed_radius_config(min_cluster_points=3),
    )

    assert not result.pose3d.valid[10]
    assert (
        result.diagnostics["joints"][10]["status"]
        == "no_supported_cluster"
    )


def test_equal_disconnected_surfaces_are_left_for_temporal_recovery() -> None:
    intrinsics = make_intrinsics()
    depth = np.full((40, 40), np.nan, dtype=np.float32)
    depth[19:22, 18] = 2.0
    depth[19:22, 22] = 3.0
    points = depth_to_organized_point_cloud(depth, intrinsics)
    pose = make_pose(target_uv=(20, 20), torso_uv=None)

    result = recover_pose3d_from_point_cloud(
        pose,
        points,
        intrinsics,
        config=fixed_radius_config(
            radius=4,
            ambiguity_relative_margin=0.1,
            reject_ambiguous_clusters=True,
        ),
    )

    assert not result.pose3d.valid[10]
    diagnostic = result.diagnostics["joints"][10]
    assert diagnostic["status"] == "ambiguous_clusters"
    assert diagnostic["cluster_count"] == 2
    assert diagnostic["selected_depth_m"] in (2.0, 3.0)


def test_ear_topology_gate_reranks_wall_surface_to_face_surface() -> None:
    intrinsics = make_intrinsics()
    depth = np.full((40, 40), 2.55, dtype=np.float32)
    depth[:, :22] = 2.0
    points = depth_to_organized_point_cloud(depth, intrinsics)
    pose = make_face_boundary_pose()

    ungated = recover_pose3d_from_point_cloud(
        pose,
        points,
        intrinsics,
        config=fixed_radius_config(
            radius=5,
            ear_topology_gate_enabled=False,
        ),
    )
    gated = recover_pose3d_from_point_cloud(
        pose,
        points,
        intrinsics,
        config=fixed_radius_config(radius=5),
    )

    assert ungated.pose3d.valid[3]
    assert np.isclose(ungated.pose3d.depth_m[3], 2.55)
    assert gated.pose3d.valid[3]
    assert np.isclose(gated.pose3d.depth_m[3], 2.0)

    diagnostic = gated.diagnostics["joints"][3]
    topology_gate = diagnostic["topology_gate"]
    assert topology_gate["applied"]
    assert topology_gate["rejected_cluster_count"] == 1
    assert topology_gate["feasible_cluster_count"] == 1
    assert topology_gate["selected_eye_ear_length_m"] < 0.25

    joint = gated.pose3d.joints_m[3]
    projected_u = intrinsics.fx * joint[0] / joint[2] + intrinsics.cx
    projected_v = intrinsics.fy * joint[1] / joint[2] + intrinsics.cy
    np.testing.assert_allclose(
        [projected_u, projected_v],
        pose.keypoints[3],
        atol=1e-5,
    )


def test_face_group_gate_reranks_eye_to_nose_eye_surface() -> None:
    intrinsics = make_intrinsics()
    depth = np.full((40, 40), 2.55, dtype=np.float32)
    depth[:, :22] = 2.0
    points = depth_to_organized_point_cloud(depth, intrinsics)
    pose = make_face_boundary_pose()
    pose.keypoints[1] = (24, 18)

    ungated = recover_pose3d_from_point_cloud(
        pose,
        points,
        intrinsics,
        config=fixed_radius_config(
            radius=5,
            face_group_gate_enabled=False,
        ),
    )
    gated = recover_pose3d_from_point_cloud(
        pose,
        points,
        intrinsics,
        config=fixed_radius_config(radius=5),
    )

    assert ungated.pose3d.valid[1]
    assert np.isclose(ungated.pose3d.depth_m[1], 2.55)
    assert gated.pose3d.valid[1]
    assert np.isclose(gated.pose3d.depth_m[1], 2.0)
    diagnostic = gated.diagnostics["joints"][1]
    assert diagnostic["status"] == "selected_face_group"
    assert diagnostic["selected_candidate_rank"] == 1
    assert diagnostic["face_group_gate"]["applied"]


def test_ear_topology_gate_rejects_frame_without_feasible_surface() -> None:
    intrinsics = make_intrinsics()
    depth = np.full((40, 40), 2.55, dtype=np.float32)
    depth[:, :22] = 2.0
    points = depth_to_organized_point_cloud(depth, intrinsics)
    pose = make_face_boundary_pose(ear_uv=(30, 19))

    result = recover_pose3d_from_point_cloud(
        pose,
        points,
        intrinsics,
        config=fixed_radius_config(radius=5),
    )

    assert not result.pose3d.valid[3]
    diagnostic = result.diagnostics["joints"][3]
    assert diagnostic["status"] == "joint_topology_rejected"
    assert diagnostic["topology_gate"]["applied"]
    assert diagnostic["topology_gate"]["feasible_cluster_count"] == 0
    assert diagnostic["topology_gate"]["rejected_cluster_count"] == 1


def test_bbox_is_clipped_and_mask_shape_is_validated() -> None:
    intrinsics = make_intrinsics()
    depth = np.full((40, 40), 2.0, dtype=np.float32)
    points = depth_to_organized_point_cloud(depth, intrinsics)
    pose = make_pose(bbox=(-10, -20, 50, 60))

    result = recover_pose3d_from_point_cloud(
        pose,
        points,
        intrinsics,
        config=fixed_radius_config(),
    )
    assert result.diagnostics["person_filter"]["padded_bbox_xyxy"] == [
        0,
        0,
        40,
        40,
    ]

    with pytest.raises(ValueError, match="person_mask must match"):
        recover_pose3d_from_point_cloud(
            pose,
            points,
            intrinsics,
            config=fixed_radius_config(),
            person_mask=np.ones((10, 10), dtype=bool),
        )


def test_config_mapping_rejects_unknown_parameter() -> None:
    with pytest.raises(ValueError, match="Unknown pointcloud_cluster"):
        PointCloudRecoveryConfig.from_mapping({"radius_typo": 4})
