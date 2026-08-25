import numpy as np
import pytest

from rgbd_avatar.camera import CameraIntrinsics
from rgbd_avatar.live import (
    AdaptiveHybridConfig,
    ApplicationExtrinsics,
    KinematicFallbackConfig,
    LocalMultiPersonConfig,
    LocalMultiPersonPoseProcessor,
    Pose3DQualityConfig,
    RGBDFrame,
)
from rgbd_avatar.pose import Pose2D, Pose3D
from rgbd_avatar.tracking import (
    BoneLengthCalibrator,
    BoneLengthConstraint,
    FramePresenceConfig,
    PersonFramePresenceGate,
    Pose3DTemporalFilter,
)
from rgbd_avatar.visualization.live_multi_person import (
    DETECTION_2D_WINDOW_NAME,
    LocalMultiPerson2DRenderer,
    build_local_multi_avatar,
    build_local_multi_rgb_views,
    build_local_multi_skeleton_arrays,
)


class _SequenceBackend:
    def __init__(self, frames: list[list[Pose2D]]) -> None:
        self.frames = frames
        self.index = 0

    def infer(self, image: np.ndarray) -> list[Pose2D]:
        assert image.shape == (60, 80, 3)
        poses = self.frames[self.index]
        self.index += 1
        return poses


def _pose(center_x: float, *, score: float = 0.9) -> Pose2D:
    keypoints = np.tile(
        np.array([[center_x, 30.0]], dtype=np.float32),
        (26, 1),
    )
    return Pose2D(
        keypoints=keypoints,
        scores=np.full(26, score, dtype=np.float32),
        bbox_xyxy=np.array(
            [center_x - 10.0, 8.0, center_x + 10.0, 55.0],
            dtype=np.float32,
        ),
        bbox_score=score,
    )


def _articulated_pose(*, score: float = 0.9) -> Pose2D:
    keypoints = np.asarray(
        (
            (40, 12), (38, 10), (42, 10), (35, 11), (45, 11),
            (30, 20), (50, 20), (25, 28), (55, 28), (20, 36),
            (60, 36), (34, 35), (46, 35), (34, 45), (46, 45),
            (34, 55), (46, 55), (40, 8), (40, 18), (40, 35),
            (32, 58), (48, 58), (35, 58), (45, 58), (33, 57),
            (47, 57),
        ),
        dtype=np.float32,
    )
    return Pose2D(
        keypoints=keypoints,
        scores=np.full(26, score, dtype=np.float32),
        bbox_xyxy=np.asarray((18, 6, 62, 59), dtype=np.float32),
        bbox_score=score,
    )


def _metric_pose3d(
    *,
    depth_offset_m: float = 0.0,
    joint_overrides: dict[int, tuple[float, float, float]] | None = None,
) -> Pose3D:
    joints = np.zeros((26, 3), dtype=np.float32)
    joints[:, 2] = 1.0 + depth_offset_m
    for index, xyz in (joint_overrides or {}).items():
        joints[index] = xyz
    return Pose3D(
        joints_m=joints,
        confidence=np.ones(26, dtype=np.float32),
        valid=np.ones(26, dtype=bool),
        depth_m=joints[:, 2].copy(),
        depth_confidence=np.ones(26, dtype=np.float32),
    )


def _pose3d_at_constant_depth(pose2d: Pose2D, depth_m: float = 2.0) -> Pose3D:
    joints = np.empty((26, 3), dtype=np.float32)
    joints[:, 0] = (pose2d.keypoints[:, 0] - 40.0) * depth_m / 100.0
    joints[:, 1] = (pose2d.keypoints[:, 1] - 30.0) * depth_m / 100.0
    joints[:, 2] = depth_m
    return Pose3D(
        joints_m=joints,
        confidence=np.ones(26, dtype=np.float32),
        valid=np.ones(26, dtype=bool),
        depth_m=joints[:, 2].copy(),
        depth_confidence=np.ones(26, dtype=np.float32),
    )


def _frame(
    number: int,
    timestamp_s: float,
    *,
    depth_m: np.ndarray | None = None,
) -> RGBDFrame:
    return RGBDFrame(
        rgb_bgr=np.zeros((60, 80, 3), dtype=np.uint8),
        depth_m=(
            np.ones((60, 80), dtype=np.float32)
            if depth_m is None
            else depth_m
        ),
        intrinsics=CameraIntrinsics(
            fx=100.0,
            fy=100.0,
            cx=40.0,
            cy=30.0,
            width=80,
            height=60,
        ),
        timestamp_ns=int(timestamp_s * 1e9),
        frame_number=number,
        source_id="test:local-multi",
    )


def _processor(
    frames: list[list[Pose2D]],
    *,
    recovery_method: str = "window_median",
    identity_tracker: str = "geometry",
    pose3d_quality_config: Pose3DQualityConfig | None = None,
    temporal_max_prediction_s: float = 0.4,
    bone_components_factory=None,
    kinematic_fallback_config: KinematicFallbackConfig | None = None,
    depth_connected_refresh_interval: int = 1,
) -> LocalMultiPersonPoseProcessor:
    return LocalMultiPersonPoseProcessor(
        backend=_SequenceBackend(frames),
        extrinsics=ApplicationExtrinsics(
            roll_deg=0.0,
            pitch_deg=0.0,
            yaw_deg=0.0,
            translation_m=np.zeros(3),
        ),
        temporal_filter_factory=lambda: Pose3DTemporalFilter(
            reset_gap_s=2.0,
            max_prediction_s=temporal_max_prediction_s,
        ),
        presence_gate_factory=lambda: PersonFramePresenceGate(
            FramePresenceConfig(enabled=False)
        ),
        bone_components_factory=(
            bone_components_factory
            if bone_components_factory is not None
            else lambda: (None, None)
        ),
        keypoint_threshold=0.3,
        min_depth_m=0.3,
        max_depth_m=6.0,
        depth_window_radius=1,
        recovery_method=recovery_method,
        pose3d_quality_config=pose3d_quality_config,
        kinematic_fallback_config=kinematic_fallback_config,
        multi_person_config=LocalMultiPersonConfig(
            max_persons=4,
            max_missing_s=0.25,
        ),
        identity_tracker=identity_tracker,
        depth_connected_refresh_interval=depth_connected_refresh_interval,
    )


def _id_by_center(result) -> dict[int, int]:
    return {
        int(round(float(person.pose2d.bbox_xyxy[[0, 2]].mean()))): (
            person.track_id
        )
        for person in result.persons
        if person.pose2d is not None and person.observed_in_frame
    }


def test_local_multi_person_keeps_ids_when_detection_order_changes() -> None:
    left = _pose(22.0)
    right = _pose(58.0)
    processor = _processor([[left, right], [right, left]])

    first = processor.process(_frame(0, 1.0))
    second = processor.process(_frame(1, 1.1))

    assert first.detected_person_count == 2
    assert _id_by_center(first) == {22: 1, 58: 2}
    assert _id_by_center(second) == {22: 1, 58: 2}
    assert processor.active_track_ids == (1, 2)
    assert all(person.status == "ok" for person in second.persons)


def test_shadow_identity_is_integrated_and_keeps_ids() -> None:
    left = _pose(22.0)
    right = _pose(58.0)
    processor = _processor(
        [[left, right], [right, left]],
        identity_tracker="shadow",
    )

    first = processor.process(_frame(0, 1.0))
    second = processor.process(_frame(1, 1.1))

    assert first.identity_method == "shadow"
    assert not first.identity_fallback
    assert _id_by_center(first) == {22: 1, 58: 2}
    assert _id_by_center(second) == {22: 1, 58: 2}


def test_shadow_failure_automatically_falls_back_to_geometry(monkeypatch) -> None:
    left = _pose(22.0)
    processor = _processor([[left]], identity_tracker="shadow")
    assert processor._shadow_tracker is not None

    def fail(*_args, **_kwargs):
        raise RuntimeError("synthetic shadow failure")

    monkeypatch.setattr(processor._shadow_tracker, "update", fail)
    result = processor.process(_frame(0, 1.0))

    assert result.identity_method == "geometry"
    assert result.identity_fallback
    assert _id_by_center(result) == {22: 1}


def test_local_multi_person_hides_short_gap_then_assigns_new_id() -> None:
    left = _pose(22.0)
    right = _pose(58.0)
    processor = _processor(
        [
            [left, right],
            [right],
            [right],
            [left, right],
        ]
    )

    processor.process(_frame(0, 1.0))
    short_gap = processor.process(_frame(1, 1.1))
    expired = processor.process(_frame(2, 1.4))
    reentered = processor.process(_frame(3, 1.5))

    missing = next(person for person in short_gap.persons if person.track_id == 1)
    assert not missing.observed_in_frame
    assert missing.status == "temporarily_missing"
    assert not np.any(missing.pose3d_output.usable)
    assert not np.any(missing.pose3d_output.predicted)
    assert [person.track_id for person in expired.persons] == [2]
    assert _id_by_center(reentered) == {22: 3, 58: 2}


def test_detection_without_valid_depth_does_not_create_or_renew_track() -> None:
    pose = _pose(22.0)
    processor = _processor([[pose], [pose], [pose]])
    no_depth = np.zeros((60, 80), dtype=np.float32)

    first = processor.process(_frame(0, 1.0))
    missing = processor.process(_frame(1, 1.1, depth_m=no_depth))
    expired = processor.process(_frame(2, 1.4, depth_m=no_depth))

    assert first.persons[0].track_id == 1
    assert missing.detected_person_count == 1
    assert [person.track_id for person in missing.persons] == [1]
    assert not missing.persons[0].observed_in_frame
    assert not np.any(missing.persons[0].pose3d_output.usable)
    assert expired.detected_person_count == 1
    assert not expired.persons
    assert processor.active_track_ids == ()


def test_detection_without_valid_depth_cannot_start_shadow_track() -> None:
    pose = _pose(22.0)
    processor = _processor([[pose]], identity_tracker="shadow")
    no_depth = np.zeros((60, 80), dtype=np.float32)

    result = processor.process(_frame(0, 1.0, depth_m=no_depth))

    assert result.identity_method == "shadow"
    assert result.detected_person_count == 1
    assert not result.persons
    assert processor.active_track_ids == ()


def test_missing_track_is_retained_internally_but_not_rendered() -> None:
    pose = _pose(22.0)
    processor = _processor([[pose], []])
    processor.process(_frame(0, 1.0))

    missing = processor.process(_frame(1, 1.1))
    avatar = build_local_multi_avatar(missing)
    points, lines, colors = build_local_multi_skeleton_arrays(missing)

    assert processor.active_track_ids == (1,)
    assert [person.track_id for person in missing.persons] == [1]
    assert avatar.primitive_count == 0
    assert points.shape == (0, 3)
    assert lines.shape == (0, 2)
    assert colors.shape == (0, 3)


def test_stale_track_cannot_be_revived_by_a_late_detection() -> None:
    left = _pose(22.0)
    processor = _processor([[left], [], [left]])

    first = processor.process(_frame(0, 1.0))
    processor.process(_frame(1, 1.1))
    late = processor.process(_frame(2, 1.4))

    assert _id_by_center(first) == {22: 1}
    assert _id_by_center(late) == {22: 2}


def test_pose3d_quality_invalidates_one_contaminated_limb(monkeypatch) -> None:
    import rgbd_avatar.live.multi_person_processor as module

    bad_wrist = _metric_pose3d(
        joint_overrides={9: (2.0, 0.0, 1.0)},
    )
    monkeypatch.setattr(module, "recover_pose3d", lambda **_kwargs: bad_wrist)
    processor = _processor([[_pose(22.0)]])

    result = processor.process(_frame(0, 1.0))

    assert len(result.persons) == 1
    assert result.persons[0].observed_in_frame
    assert not result.persons[0].pose3d_raw.valid[9]
    assert not result.persons[0].pose3d_output.usable[9]
    assert np.count_nonzero(result.persons[0].pose3d_output.usable) == 25
    assert result.recovery_stats["quality_invalidated_joint_count"] == 1
    assert result.persons[0].joint_sources[9] == "bone_length_violation"


def test_kinematic_fallback_preserves_bone_then_expires(monkeypatch) -> None:
    import rgbd_avatar.live.multi_person_processor as module

    arm_pose = _metric_pose3d(
        joint_overrides={
            5: (-0.20, 0.0, 2.0),
            7: (-0.45, 0.0, 2.0),
            9: (-0.70, 0.0, 2.0),
        },
    )
    missing_wrist = Pose3D(
        joints_m=arm_pose.joints_m.copy(),
        confidence=arm_pose.confidence.copy(),
        valid=arm_pose.valid.copy(),
        depth_m=arm_pose.depth_m.copy(),
        depth_confidence=arm_pose.depth_confidence.copy(),
    )
    missing_wrist.joints_m[9] = np.nan
    missing_wrist.confidence[9] = 0.0
    missing_wrist.valid[9] = False
    missing_wrist.depth_m[9] = np.nan
    missing_wrist.depth_confidence[9] = 0.0
    recovered = iter((arm_pose, missing_wrist, missing_wrist))
    monkeypatch.setattr(
        module,
        "recover_pose3d",
        lambda **_kwargs: next(recovered),
    )

    def bone_components():
        return (
            BoneLengthCalibrator(
                min_samples_per_bone=1,
                target_samples_per_bone=1,
                max_samples_per_bone=3,
                min_keypoint_confidence=0.1,
                min_depth_confidence=0.1,
            ),
            BoneLengthConstraint(),
        )

    pose = _pose(22.0)
    processor = _processor(
        [[pose], [pose], [pose]],
        pose3d_quality_config=Pose3DQualityConfig(enabled=False),
        temporal_max_prediction_s=0.05,
        bone_components_factory=bone_components,
        kinematic_fallback_config=KinematicFallbackConfig(
            max_age_s=0.25,
            reconstruct_from_current_2d=False,
        ),
    )

    first = processor.process(_frame(0, 1.0))
    filled = processor.process(_frame(1, 1.1))
    expired = processor.process(_frame(2, 1.31))

    first_person = first.persons[0]
    filled_person = filled.persons[0]
    expired_person = expired.persons[0]
    expected_length = float(
        np.linalg.norm(
            first_person.pose3d_output.joints_m[9]
            - first_person.pose3d_output.joints_m[7]
        )
    )
    filled_length = float(
        np.linalg.norm(
            filled_person.pose3d_output.joints_m[9]
            - filled_person.pose3d_output.joints_m[7]
        )
    )

    assert not filled_person.pose3d_raw.valid[9]
    assert filled_person.pose3d_output.usable[9]
    assert filled_person.pose3d_output.predicted[9]
    assert filled_person.kinematic_fallback[9]
    assert filled_person.joint_sources[9] == "kinematic_fallback"
    assert filled_length == pytest.approx(expected_length, abs=1e-5)
    assert filled.recovery_stats["kinematic_fallback_joint_count"] == 1
    assert not expired_person.pose3d_output.usable[9]
    assert expired_person.joint_sources[9] == "no_depth_candidate"

    track = processor._tracks[first_person.track_id]
    forearm_index = track.bone_calibrator.prior().links.index((7, 9))
    assert track.bone_calibrator.prior().sample_count[forearm_index] == 1


def test_kinematic_fallback_config_rejects_invalid_age() -> None:
    with np.testing.assert_raises(ValueError):
        KinematicFallbackConfig(max_age_s=0.0)


def test_established_sparse_track_completes_last_skeleton_then_expires(
    monkeypatch,
) -> None:
    import rgbd_avatar.live.multi_person_processor as module

    complete = _metric_pose3d()
    sparse = _metric_pose3d(joint_overrides={19: (0.10, 0.0, 1.0)})
    sparse.valid[:] = False
    sparse.valid[19] = True
    sparse.confidence[:] = 0.0
    sparse.confidence[19] = 1.0
    sparse.depth_confidence[:] = 0.0
    sparse.depth_confidence[19] = 1.0
    sparse.joints_m[~sparse.valid] = np.nan
    sparse.depth_m[~sparse.valid] = np.nan
    recovered = iter((complete, sparse, sparse))
    monkeypatch.setattr(
        module,
        "recover_pose3d_from_depth_connected",
        lambda **_kwargs: next(recovered),
    )

    complete_2d = _pose(22.0)
    sparse_scores = np.full(26, 0.05, dtype=np.float32)
    sparse_scores[[18, 19]] = 0.9
    sparse_2d = Pose2D(
        keypoints=complete_2d.keypoints.copy(),
        scores=sparse_scores,
        bbox_xyxy=complete_2d.bbox_xyxy.copy(),
        bbox_score=complete_2d.bbox_score,
    )
    processor = _processor(
        [[complete_2d], [sparse_2d], [sparse_2d]],
        recovery_method="depth_connected",
        temporal_max_prediction_s=0.05,
        bone_components_factory=lambda: (
            BoneLengthCalibrator(),
            BoneLengthConstraint(),
        ),
        kinematic_fallback_config=KinematicFallbackConfig(
            max_age_s=0.25,
            complete_skeleton=True,
            reconstruct_from_current_2d=False,
            min_core_2d_joint_count=2,
            min_core_3d_joint_count=1,
            min_history_joint_count=8,
        ),
    )

    first = processor.process(_frame(0, 1.0))
    completed = processor.process(_frame(1, 1.1))
    expired = processor.process(_frame(2, 1.31))

    assert first.persons[0].pose3d_output.usable.all()
    completed_person = completed.persons[0]
    assert completed_person.observed_in_frame
    assert completed_person.pose3d_output.usable.all()
    assert np.count_nonzero(completed_person.kinematic_fallback) == 25
    assert completed_person.joint_sources[19] == "observed"
    assert all(
        source == "skeleton_completion"
        for index, source in enumerate(completed_person.joint_sources)
        if index != 19
    )
    assert completed.recovery_stats["skeleton_completion_joint_count"] == 25
    assert completed.recovery_stats["missing_output_joint_count"] == 0

    expired_person = expired.persons[0]
    assert expired_person.observed_in_frame
    assert np.count_nonzero(expired_person.pose3d_output.usable) == 1
    assert expired_person.pose3d_output.usable[19]


def test_current_2d_reconstructs_joint_without_any_joint_depth_history(
    monkeypatch,
) -> None:
    import rgbd_avatar.live.multi_person_processor as module

    missing_wrist = _metric_pose3d(
        joint_overrides={
            5: (-0.20, 0.0, 2.0),
            7: (-0.45, 0.0, 2.0),
        },
    )
    missing_wrist.joints_m[9] = np.nan
    missing_wrist.confidence[9] = 0.0
    missing_wrist.valid[9] = False
    missing_wrist.depth_m[9] = np.nan
    missing_wrist.depth_confidence[9] = 0.0
    monkeypatch.setattr(
        module,
        "recover_pose3d_from_depth_connected",
        lambda **_kwargs: missing_wrist,
    )

    pose = _pose(22.0)
    keypoints = pose.keypoints.copy()
    keypoints[5] = (30.0, 25.0)
    keypoints[7] = (22.0, 20.0)
    keypoints[9] = (12.0, 15.0)
    current_2d = Pose2D(
        keypoints=keypoints,
        scores=pose.scores.copy(),
        bbox_xyxy=pose.bbox_xyxy.copy(),
        bbox_score=pose.bbox_score,
    )
    processor = _processor(
        [[current_2d], [current_2d], [current_2d]],
        recovery_method="depth_connected",
        pose3d_quality_config=Pose3DQualityConfig(enabled=False),
        temporal_max_prediction_s=0.05,
        bone_components_factory=lambda: (
            BoneLengthCalibrator(),
            BoneLengthConstraint(),
        ),
        kinematic_fallback_config=KinematicFallbackConfig(
            max_age_s=0.25,
            complete_skeleton=True,
            reconstruct_from_current_2d=True,
        ),
    )

    first = processor.process(_frame(0, 1.0))
    completed = processor.process(_frame(1, 1.1))
    still_completed = processor.process(_frame(2, 1.5))

    assert not first.persons[0].pose3d_output.usable[9]
    for result in (completed, still_completed):
        person = result.persons[0]
        assert person.pose3d_output.usable[9]
        assert person.pose3d_output.predicted[9]
        assert person.pose3d_output.age_s[9] == pytest.approx(0.0)
        assert person.joint_sources[9] == "skeleton_completion"
        forearm_length_m = np.linalg.norm(
            person.pose3d_output.joints_m[9]
            - person.pose3d_output.joints_m[7]
        )
        assert 0.0 < forearm_length_m <= 0.50


def test_current_2d_reconstructs_complete_halpe26_from_core_depth(
    monkeypatch,
) -> None:
    import rgbd_avatar.live.multi_person_processor as module

    keypoints = np.asarray(
        (
            (40, 12), (38, 10), (42, 10), (35, 11), (45, 11),
            (30, 20), (50, 20), (25, 28), (55, 28), (20, 36),
            (60, 36), (34, 35), (46, 35), (34, 45), (46, 45),
            (34, 55), (46, 55), (40, 8), (40, 18), (40, 35),
            (32, 58), (48, 58), (35, 58), (45, 58), (33, 57),
            (47, 57),
        ),
        dtype=np.float32,
    )
    pose2d = Pose2D(
        keypoints=keypoints,
        scores=np.full(26, 0.9, dtype=np.float32),
        bbox_xyxy=np.asarray((18, 6, 62, 59), dtype=np.float32),
        bbox_score=0.9,
    )
    core_indices = np.asarray((18, 19, 5, 6, 11, 12, 13, 14))
    joints = np.full((26, 3), np.nan, dtype=np.float32)
    joints[core_indices, 0] = (
        keypoints[core_indices, 0] - 40.0
    ) * 2.0 / 100.0
    joints[core_indices, 1] = (
        keypoints[core_indices, 1] - 30.0
    ) * 2.0 / 100.0
    joints[core_indices, 2] = 2.0
    valid = np.zeros(26, dtype=bool)
    valid[core_indices] = True
    sparse = Pose3D(
        joints_m=joints,
        confidence=valid.astype(np.float32),
        valid=valid,
        depth_m=joints[:, 2].copy(),
        depth_confidence=valid.astype(np.float32),
    )
    monkeypatch.setattr(
        module,
        "recover_pose3d_from_depth_connected",
        lambda **_kwargs: sparse,
    )
    processor = _processor(
        [[pose2d], [pose2d]],
        recovery_method="depth_connected",
        temporal_max_prediction_s=0.05,
        bone_components_factory=lambda: (
            BoneLengthCalibrator(),
            BoneLengthConstraint(),
        ),
        kinematic_fallback_config=KinematicFallbackConfig(
            complete_skeleton=True,
            reconstruct_from_current_2d=True,
            min_history_joint_count=8,
        ),
    )

    initial = processor.process(_frame(0, 1.0))
    completed = processor.process(_frame(1, 1.1))

    assert np.count_nonzero(initial.persons[0].pose3d_output.usable) == 8
    person = completed.persons[0]
    assert person.pose3d_output.usable.all()
    assert np.count_nonzero(person.kinematic_fallback) == 18
    assert completed.recovery_stats["missing_output_joint_count"] == 0
    assert completed.recovery_stats["skeleton_completion_joint_count"] == 18


def test_stretched_spine_is_rebuilt_from_current_2d_projection(
    monkeypatch,
) -> None:
    import rgbd_avatar.live.multi_person_processor as module

    pose2d = _articulated_pose()
    normal = _pose3d_at_constant_depth(pose2d)
    stretched_joints = normal.joints_m.copy()
    stretched_joints[18, 2] += 0.50
    stretched = Pose3D(
        joints_m=stretched_joints,
        confidence=normal.confidence.copy(),
        valid=normal.valid.copy(),
        depth_m=stretched_joints[:, 2].copy(),
        depth_confidence=normal.depth_confidence.copy(),
    )
    recovered = iter((normal, stretched))
    monkeypatch.setattr(
        module,
        "recover_pose3d",
        lambda **_kwargs: next(recovered),
    )
    processor = _processor(
        [[pose2d], [pose2d]],
        recovery_method="window_median",
        temporal_max_prediction_s=0.05,
        bone_components_factory=lambda: (
            BoneLengthCalibrator(),
            BoneLengthConstraint(),
        ),
    )

    processor.process(_frame(0, 1.0))
    result = processor.process(_frame(1, 1.1))

    person = result.persons[0]
    assert not person.pose3d_raw.valid[18]
    assert person.pose3d_output.usable.all()
    assert person.joint_sources[18] in {
        "kinematic_fallback",
        "skeleton_completion",
    }
    spine_length_m = np.linalg.norm(
        person.pose3d_output.joints_m[18]
        - person.pose3d_output.joints_m[19]
    )
    assert spine_length_m < 0.50


def test_face_outlier_is_invalidated_before_frontend_publish(monkeypatch) -> None:
    import rgbd_avatar.live.multi_person_processor as module

    pose2d = _articulated_pose()
    bad_face = _pose3d_at_constant_depth(pose2d)
    bad_face.joints_m[0] = (2.0, 0.0, 2.0)
    monkeypatch.setattr(module, "recover_pose3d", lambda **_kwargs: bad_face)
    processor = _processor([[pose2d]])

    result = processor.process(_frame(0, 1.0))

    person = result.persons[0]
    assert not person.pose3d_raw.valid[0]
    assert not person.pose3d_output.usable[0]
    assert person.joint_sources[0] == "face_geometry_violation"


def test_back_view_gets_proxy_head_without_face_landmarks(monkeypatch) -> None:
    import rgbd_avatar.live.multi_person_processor as module

    pose2d = _articulated_pose()
    back_scores = pose2d.scores.copy()
    back_scores[[0, 1, 2, 3, 4, 17]] = 0.05
    back_pose2d = Pose2D(
        keypoints=pose2d.keypoints.copy(),
        scores=back_scores,
        bbox_xyxy=pose2d.bbox_xyxy.copy(),
        bbox_score=pose2d.bbox_score,
    )
    sparse = _pose3d_at_constant_depth(pose2d)
    core_indices = np.asarray((18, 19, 5, 6, 11, 12, 13, 14))
    sparse.valid[:] = False
    sparse.valid[core_indices] = True
    sparse.confidence[~sparse.valid] = 0.0
    sparse.depth_confidence[~sparse.valid] = 0.0
    sparse.joints_m[~sparse.valid] = np.nan
    sparse.depth_m[~sparse.valid] = np.nan
    monkeypatch.setattr(module, "recover_pose3d", lambda **_kwargs: sparse)
    processor = _processor(
        [[back_pose2d], [back_pose2d]],
        recovery_method="window_median",
        temporal_max_prediction_s=0.05,
        bone_components_factory=lambda: (
            BoneLengthCalibrator(),
            BoneLengthConstraint(),
        ),
    )

    initial = processor.process(_frame(0, 1.0))
    completed = processor.process(_frame(1, 1.1))

    assert not initial.persons[0].pose3d_output.usable[17]
    person = completed.persons[0]
    assert person.pose3d_output.usable[17]
    assert person.pose3d_output.predicted[17]
    assert person.joint_sources[17] == "skeleton_completion"
    assert not np.any(person.pose3d_output.usable[:5])


def test_sparse_detection_cannot_create_a_new_completed_person(
    monkeypatch,
) -> None:
    import rgbd_avatar.live.multi_person_processor as module

    sparse = _metric_pose3d()
    sparse.valid[:] = False
    sparse.valid[19] = True
    sparse.confidence[:] = 0.0
    sparse.confidence[19] = 1.0
    sparse.depth_confidence[:] = 0.0
    sparse.depth_confidence[19] = 1.0
    sparse.joints_m[~sparse.valid] = np.nan
    sparse.depth_m[~sparse.valid] = np.nan
    monkeypatch.setattr(
        module,
        "recover_pose3d_from_depth_connected",
        lambda **_kwargs: sparse,
    )
    pose = _pose(22.0)
    scores = np.full(26, 0.05, dtype=np.float32)
    scores[[18, 19]] = 0.9
    sparse_2d = Pose2D(
        keypoints=pose.keypoints.copy(),
        scores=scores,
        bbox_xyxy=pose.bbox_xyxy.copy(),
        bbox_score=pose.bbox_score,
    )
    processor = _processor(
        [[sparse_2d]],
        recovery_method="depth_connected",
        bone_components_factory=lambda: (
            BoneLengthCalibrator(),
            BoneLengthConstraint(),
        ),
    )

    result = processor.process(_frame(0, 1.0))

    assert not result.persons
    assert processor.active_track_ids == ()


def test_kinematic_fallback_config_rejects_invalid_completion_counts() -> None:
    with np.testing.assert_raises(ValueError):
        KinematicFallbackConfig(min_core_2d_joint_count=0)
    with np.testing.assert_raises(ValueError):
        KinematicFallbackConfig(min_history_joint_count=27)


def test_pose3d_quality_rejects_multiple_bad_limb_depths(monkeypatch) -> None:
    import rgbd_avatar.live.multi_person_processor as module

    malformed = _metric_pose3d(
        joint_overrides={
            9: (2.0, 0.0, 1.0),
            10: (-2.0, 0.0, 1.0),
        },
    )
    monkeypatch.setattr(module, "recover_pose3d", lambda **_kwargs: malformed)
    processor = _processor([[_pose(22.0)]])

    result = processor.process(_frame(0, 1.0))

    assert result.detected_person_count == 1
    assert not result.persons
    assert processor.active_track_ids == ()
    assert result.recovery_stats["quality_rejected_person_count"] == 1


def test_pose3d_quality_rejects_coherent_background_depth_jump(
    monkeypatch,
) -> None:
    import rgbd_avatar.live.multi_person_processor as module

    recovered = iter((_metric_pose3d(), _metric_pose3d(depth_offset_m=1.0)))
    monkeypatch.setattr(
        module,
        "recover_pose3d",
        lambda **_kwargs: next(recovered),
    )
    pose = _pose(22.0)
    processor = _processor([[pose], [pose]])

    first = processor.process(_frame(0, 1.0))
    jumped = processor.process(_frame(1, 1.1))

    assert first.persons[0].observed_in_frame
    assert [person.track_id for person in jumped.persons] == [1]
    assert not jumped.persons[0].observed_in_frame
    assert not np.any(jumped.persons[0].pose3d_output.usable)
    assert jumped.recovery_stats["quality_rejected_person_count"] == 1


def test_pose3d_quality_accepts_normal_depth_motion(monkeypatch) -> None:
    import rgbd_avatar.live.multi_person_processor as module

    recovered = iter((_metric_pose3d(), _metric_pose3d(depth_offset_m=0.15)))
    monkeypatch.setattr(
        module,
        "recover_pose3d",
        lambda **_kwargs: next(recovered),
    )
    pose = _pose(22.0)
    processor = _processor([[pose], [pose]])

    processor.process(_frame(0, 1.0))
    moved = processor.process(_frame(1, 1.1))

    assert moved.persons[0].observed_in_frame
    assert np.all(moved.persons[0].pose3d_output.usable)
    assert moved.recovery_stats["quality_rejected_person_count"] == 0


def test_point_cloud_is_built_once_for_all_people(monkeypatch) -> None:
    import rgbd_avatar.live.multi_person_processor as module

    calls = 0
    original = module.depth_to_organized_point_cloud

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "depth_to_organized_point_cloud", counted)
    processor = _processor(
        [[_pose(22.0), _pose(58.0)]],
        recovery_method="pointcloud_cluster",
    )

    result = processor.process(_frame(0, 1.0))

    assert calls == 1
    assert result.detected_person_count == 2
    assert len(result.persons) == 2


def test_depth_connected_uses_history_without_building_point_cloud(
    monkeypatch,
) -> None:
    import rgbd_avatar.live.multi_person_processor as module

    pointcloud_calls = 0
    fast_recovery_calls = 0
    expected_depth_calls: list[np.ndarray | None] = []
    original_cloud = module.depth_to_organized_point_cloud
    original_fast = module.recover_pose3d
    original_recovery = module.recover_pose3d_from_depth_connected

    def counted_cloud(*args, **kwargs):
        nonlocal pointcloud_calls
        pointcloud_calls += 1
        return original_cloud(*args, **kwargs)

    def counted_fast(*args, **kwargs):
        nonlocal fast_recovery_calls
        fast_recovery_calls += 1
        return original_fast(*args, **kwargs)

    def recorded_recovery(*args, **kwargs):
        expected = kwargs["expected_depths_m"]
        expected_depth_calls.append(
            None if expected is None else expected.copy()
        )
        return original_recovery(*args, **kwargs)

    monkeypatch.setattr(module, "depth_to_organized_point_cloud", counted_cloud)
    monkeypatch.setattr(module, "recover_pose3d", counted_fast)
    monkeypatch.setattr(
        module,
        "recover_pose3d_from_depth_connected",
        recorded_recovery,
    )
    pose = _pose(22.0)
    processor = _processor(
        [[pose], [pose]],
        recovery_method="depth_connected",
    )

    first = processor.process(_frame(0, 1.0))
    second = processor.process(_frame(1, 1.1))

    assert pointcloud_calls == 0
    assert fast_recovery_calls == 0
    assert len(expected_depth_calls) == 2
    assert expected_depth_calls[0] is None
    assert expected_depth_calls[1] is not None
    assert np.isfinite(expected_depth_calls[1]).all()
    assert np.all(first.persons[0].pose3d_raw.valid)
    assert np.all(second.persons[0].pose3d_raw.valid)
    assert first.timing_ms["recovery_fast"] == 0.0
    assert second.timing_ms["recovery_fast"] == 0.0


def test_depth_connected_refresh_interval_uses_guided_frames(
    monkeypatch,
) -> None:
    import rgbd_avatar.live.multi_person_processor as module

    full_calls = 0
    guided_calls = 0

    def full_recovery(**kwargs):
        nonlocal full_calls
        full_calls += 1
        return _pose3d_at_constant_depth(kwargs["pose2d"])

    def guided_recovery(**kwargs):
        nonlocal guided_calls
        guided_calls += 1
        assert kwargs["expected_depths_m"] is not None
        return _pose3d_at_constant_depth(kwargs["pose2d"])

    monkeypatch.setattr(
        module,
        "recover_pose3d_from_depth_connected",
        full_recovery,
    )
    monkeypatch.setattr(module, "recover_pose3d", guided_recovery)
    pose = _pose(22.0)
    processor = _processor(
        [[pose], [pose], [pose], [pose]],
        recovery_method="depth_connected",
        depth_connected_refresh_interval=3,
    )

    results = [
        processor.process(_frame(index, 1.0 + index * 0.1))
        for index in range(4)
    ]

    assert full_calls == 2
    assert guided_calls == 2
    assert [
        result.recovery_stats["depth_connected_full_person_count"]
        for result in results
    ] == [1, 0, 1, 0]
    assert [
        result.recovery_stats["depth_connected_guided_person_count"]
        for result in results
    ] == [0, 1, 0, 1]


def test_depth_guidance_is_disabled_after_repeated_invalid_observations() -> None:
    pose = _pose(22.0)
    processor = _processor(
        [[pose], [pose], [pose], [pose]],
        recovery_method="depth_connected",
    )
    processor.process(_frame(0, 1.0))
    empty_depth = np.zeros((60, 80), dtype=np.float32)
    processor.process(_frame(1, 1.1, depth_m=empty_depth))
    processor.process(_frame(2, 1.2, depth_m=empty_depth))
    processor.process(_frame(3, 1.3, depth_m=empty_depth))

    track = processor._tracks[1]
    assert np.all(track.depth_guidance_failure_counts >= 3)
    expected = processor._expected_depths(track)
    assert expected is not None
    assert not np.isfinite(expected).any()


def test_hybrid_recovers_only_high_risk_joints_with_point_cloud(
    monkeypatch,
) -> None:
    import rgbd_avatar.live.multi_person_processor as module

    requested: list[frozenset[int]] = []
    original = module.recover_pose3d_from_point_cloud

    def recorded(*args, **kwargs):
        requested.append(frozenset(kwargs["joint_indices"]))
        assert kwargs["person_depth_hint_m"] == 1.0
        return original(*args, **kwargs)

    monkeypatch.setattr(
        module,
        "recover_pose3d_from_point_cloud",
        recorded,
    )
    processor = _processor(
        [[_pose(22.0), _pose(58.0)]],
        recovery_method="hybrid",
    )

    result = processor.process(_frame(0, 1.0))

    expected = frozenset((*range(0, 11), 17, 18))
    assert requested == [expected, expected]
    assert all(
        np.count_nonzero(person.pose3d_raw.valid) == 26
        for person in result.persons
    )


def test_adaptive_hybrid_skips_point_cloud_for_confident_depth(
    monkeypatch,
) -> None:
    import rgbd_avatar.live.multi_person_processor as module

    calls = 0
    original = module.depth_to_organized_point_cloud

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "depth_to_organized_point_cloud", counted)
    processor = _processor(
        [[_pose(22.0), _pose(58.0)]],
        recovery_method="adaptive_hybrid",
    )

    result = processor.process(_frame(0, 1.0))

    assert calls == 0
    assert result.recovery_stats["robust_joint_count"] == 0
    assert result.timing_ms["recovery_cloud_build"] == 0.0
    assert result.timing_ms["recovery_robust"] == 0.0
    assert all(np.all(person.pose3d_raw.valid) for person in result.persons)


def test_adaptive_hybrid_recovers_complete_suspicious_arm_group(
    monkeypatch,
) -> None:
    import rgbd_avatar.live.multi_person_processor as module

    requested: list[frozenset[int]] = []
    original_robust = module.recover_pose3d_from_point_cloud

    def fast_with_missing_left_wrist(*_args, **_kwargs):
        joints = np.zeros((26, 3), dtype=np.float32)
        joints[:, 2] = 1.0
        valid = np.ones(26, dtype=bool)
        valid[9] = False
        joints[9] = np.nan
        depth = np.ones(26, dtype=np.float32)
        depth[9] = np.nan
        depth_confidence = np.ones(26, dtype=np.float32)
        depth_confidence[9] = 0.0
        return Pose3D(
            joints_m=joints,
            confidence=depth_confidence.copy(),
            valid=valid,
            depth_m=depth,
            depth_confidence=depth_confidence,
        )

    def recorded_robust(*args, **kwargs):
        requested.append(frozenset(kwargs["joint_indices"]))
        return original_robust(*args, **kwargs)

    monkeypatch.setattr(module, "recover_pose3d", fast_with_missing_left_wrist)
    monkeypatch.setattr(
        module,
        "recover_pose3d_from_point_cloud",
        recorded_robust,
    )
    processor = _processor(
        [[_pose(22.0)]],
        recovery_method="adaptive_hybrid",
    )

    result = processor.process(_frame(0, 1.0))

    assert requested == [frozenset((5, 7, 9, 18))]
    assert result.recovery_stats["robust_joint_count"] == 4
    assert result.timing_ms["recovery_cloud_build"] > 0.0
    assert result.timing_ms["recovery_robust"] > 0.0
    assert result.persons[0].pose3d_raw.valid[9]


def test_adaptive_hybrid_config_rejects_invalid_confidence() -> None:
    with np.testing.assert_raises(ValueError):
        AdaptiveHybridConfig(min_depth_confidence=1.1)


def test_pose3d_quality_config_rejects_invalid_prior_ratio() -> None:
    with np.testing.assert_raises(ValueError):
        Pose3DQualityConfig(prior_length_ratio=1.0)
    with np.testing.assert_raises(ValueError):
        Pose3DQualityConfig(max_spine_projection_ratio=0.9)
def test_guided_window_keeps_track_depth_and_rejects_full_occluder() -> None:
    pose = _pose(22.0)
    processor = _processor(
        [[pose], [pose], [pose], [pose]],
        recovery_method="guided_window",
    )
    initial_depth = np.zeros((60, 80), dtype=np.float32)
    initial_depth[29:32, 21:24] = 2.0
    split_depth = np.zeros((60, 80), dtype=np.float32)
    split_depth[29:32, 21:24] = np.array(
        [
            [1.0, 1.0, 2.0],
            [1.0, 1.0, 2.0],
            [2.0, 2.0, 2.0],
        ],
        dtype=np.float32,
    )
    occluded_depth = np.zeros((60, 80), dtype=np.float32)
    occluded_depth[29:32, 21:24] = 1.0

    first = processor.process(_frame(0, 1.0, depth_m=initial_depth))
    split = processor.process(_frame(1, 1.1, depth_m=split_depth))
    occluded = processor.process(_frame(2, 1.2, depth_m=occluded_depth))
    reacquired = processor.process(_frame(3, 1.3, depth_m=initial_depth))

    assert np.allclose(first.persons[0].pose3d_raw.depth_m, 2.0)
    assert np.allclose(split.persons[0].pose3d_raw.depth_m, 2.0)
    assert occluded.persons[0].pose3d_raw is None
    assert not occluded.persons[0].observed_in_frame
    assert not np.any(occluded.persons[0].pose3d_output.usable)
    assert not np.any(occluded.persons[0].pose3d_output.predicted)
    assert reacquired.persons[0].track_id == first.persons[0].track_id
    assert reacquired.persons[0].observed_in_frame


def test_local_multi_person_builds_colored_views_without_web_payload() -> None:
    processor = _processor([[_pose(22.0), _pose(58.0)]])
    result = processor.process(_frame(0, 1.0))

    raw, overlay = build_local_multi_rgb_views(
        result,
        keypoint_threshold=0.3,
        scale=0.5,
    )
    avatar = build_local_multi_avatar(result)
    points, lines, colors = build_local_multi_skeleton_arrays(result)

    assert raw.shape == overlay.shape == (30, 40, 3)
    assert np.count_nonzero(raw) == 0
    assert np.count_nonzero(overlay) > 0
    assert avatar.primitive_count > 0
    assert len({primitive.color for primitive in avatar.ellipsoids}) >= 2
    assert points.shape == (52, 3)
    assert lines.shape == colors.shape[:1] + (2,)
    assert colors.shape[1] == 3


def test_standalone_2d_renderer_opens_and_updates(monkeypatch) -> None:
    import rgbd_avatar.visualization.live_multi_person as module

    shown: list[tuple[str, np.ndarray]] = []
    destroyed: list[str] = []
    monkeypatch.setattr(module.cv2, "namedWindow", lambda *_args: None)
    monkeypatch.setattr(module.cv2, "moveWindow", lambda *_args: None)
    monkeypatch.setattr(
        module.cv2,
        "imshow",
        lambda name, image: shown.append((name, image)),
    )
    monkeypatch.setattr(module.cv2, "waitKey", lambda _delay: ord("q"))
    monkeypatch.setattr(
        module.cv2,
        "destroyWindow",
        lambda name: destroyed.append(name),
    )
    result = _processor([[_pose(22.0), _pose(58.0)]]).process(
        _frame(0, 1.0)
    )
    renderer = LocalMultiPerson2DRenderer(
        rgb_view_scale=0.5,
        keypoint_threshold=0.3,
    )

    renderer.open()
    renderer.update(result)
    assert shown[0][0] == DETECTION_2D_WINDOW_NAME
    assert shown[0][1].shape == (30, 40, 3)
    assert not renderer.poll()
    renderer.close()
    assert destroyed == [DETECTION_2D_WINDOW_NAME]
