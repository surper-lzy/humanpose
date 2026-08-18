import numpy as np

from rgbd_avatar.pose import Pose2D
from rgbd_avatar.tracking.shadow_identity import (
    ShadowIdentityConfig,
    ShadowIdentityObservation,
    ShadowRGBDIdentityTracker,
    extract_upper_body_hsv_descriptor,
)


def _pose(center_x: float, center_y: float = 30.0) -> Pose2D:
    keypoints = np.tile(
        np.array([[center_x, center_y]], dtype=np.float32),
        (26, 1),
    )
    keypoints[5] = (center_x - 5.0, center_y - 8.0)
    keypoints[6] = (center_x + 5.0, center_y - 8.0)
    keypoints[11] = (center_x - 4.0, center_y + 8.0)
    keypoints[12] = (center_x + 4.0, center_y + 8.0)
    return Pose2D(
        keypoints=keypoints,
        scores=np.full(26, 0.9, dtype=np.float32),
        bbox_xyxy=np.array(
            [center_x - 10.0, 5.0, center_x + 10.0, 55.0],
            dtype=np.float32,
        ),
        bbox_score=0.9,
    )


def _observation(
    identity: int,
    center_x: float,
    appearance: tuple[float, float] | None,
) -> ShadowIdentityObservation:
    return ShadowIdentityObservation(
        observation_id=identity,
        pose2d=_pose(center_x),
        root_camera_m=np.array(
            [(center_x - 50.0) / 100.0, 0.0, 2.0],
            dtype=np.float64,
        ),
        appearance=(
            np.asarray(appearance, dtype=np.float32)
            if appearance is not None
            else None
        ),
    )


def _identity_map(frame) -> dict[int, int]:
    return {
        assignment.observation_id: assignment.shadow_id
        for assignment in frame.assignments
    }


def test_shadow_tracker_keeps_appearance_ids_through_crossing() -> None:
    tracker = ShadowRGBDIdentityTracker()
    sequence = [
        (0.0, [(101, 20.0, (1.0, 0.0)), (202, 80.0, (0.0, 1.0))]),
        (0.1, [(202, 65.0, (0.0, 1.0)), (101, 35.0, (1.0, 0.0))]),
        (0.2, [(202, 52.0, (0.0, 1.0)), (101, 48.0, (1.0, 0.0))]),
        (0.3, [(101, 65.0, (1.0, 0.0)), (202, 35.0, (0.0, 1.0))]),
        (0.4, [(202, 20.0, (0.0, 1.0)), (101, 80.0, (1.0, 0.0))]),
    ]

    identity_maps = []
    overlap_frozen = False
    for timestamp_s, values in sequence:
        frame = tracker.update(
            [
                _observation(identity, center_x, appearance)
                for identity, center_x, appearance in values
            ],
            timestamp_s,
        )
        identity_maps.append(_identity_map(frame))
        overlap_frozen |= any(
            assignment.appearance_frozen
            for assignment in frame.assignments
        )

    assert identity_maps[0] == {101: 1, 202: 2}
    assert all(mapping == {101: 1, 202: 2} for mapping in identity_maps)
    assert overlap_frozen


def test_shadow_tracker_rejects_forced_identity_during_ambiguous_overlap() -> None:
    tracker = ShadowRGBDIdentityTracker(
        ShadowIdentityConfig(ambiguity_margin=0.10)
    )
    tracker.update(
        [
            _observation(1, 30.0, None),
            _observation(2, 70.0, None),
        ],
        0.0,
    )

    ambiguous = tracker.update(
        [
            _observation(11, 50.0, None),
            _observation(22, 50.0, None),
        ],
        0.1,
    )

    assert ambiguous.assignments == ()
    assert ambiguous.predicted_shadow_ids == (1, 2)
    assert ambiguous.ambiguous_observation_ids == (11, 22)


def test_occlusion_grace_keeps_then_expires_shadow_tracks() -> None:
    tracker = ShadowRGBDIdentityTracker(
        ShadowIdentityConfig(
            normal_missing_s=0.2,
            occluded_missing_s=1.0,
        )
    )
    tracker.update(
        [
            _observation(1, 35.0, (1.0, 0.0)),
            _observation(2, 65.0, (0.0, 1.0)),
        ],
        0.0,
    )
    overlap = tracker.update(
        [
            _observation(1, 48.0, (1.0, 0.0)),
            _observation(2, 52.0, (0.0, 1.0)),
        ],
        0.1,
    )
    assert all(item.appearance_frozen for item in overlap.assignments)

    retained = tracker.update([], 0.8)
    expired = tracker.update([], 1.2)

    assert retained.predicted_shadow_ids == (1, 2)
    assert expired.removed_shadow_ids == (1, 2)
    assert tracker.active_shadow_ids == ()


def test_hsv_descriptor_separates_torso_colours() -> None:
    red = np.zeros((60, 100, 3), dtype=np.uint8)
    blue = np.zeros((60, 100, 3), dtype=np.uint8)
    red[:, :] = (0, 0, 255)
    blue[:, :] = (255, 0, 0)
    pose = _pose(50.0)

    red_descriptor = extract_upper_body_hsv_descriptor(red, pose)
    blue_descriptor = extract_upper_body_hsv_descriptor(blue, pose)

    assert red_descriptor is not None
    assert blue_descriptor is not None
    assert np.isclose(np.linalg.norm(red_descriptor), 1.0)
    assert float(np.dot(red_descriptor, blue_descriptor)) < 0.2
