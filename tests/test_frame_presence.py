import numpy as np
import pytest

from rgbd_avatar.pose import HALPE26_NAMES, Pose2D
from rgbd_avatar.tracking import (
    FramePresenceConfig,
    PersonFramePresenceGate,
)


def make_pose(
    *,
    bbox: tuple[float, float, float, float],
    valid_ids: set[int] | None = None,
) -> Pose2D:
    count = len(HALPE26_NAMES)
    keypoints = np.column_stack(
        (
            np.linspace(250.0, 400.0, count),
            np.linspace(150.0, 580.0, count),
        )
    ).astype(np.float32)
    scores = np.full(count, 0.8, dtype=np.float32)
    if valid_ids is not None:
        scores.fill(0.1)
        scores[list(valid_ids)] = 0.8
    return Pose2D(
        keypoints=keypoints,
        scores=scores,
        bbox_xyxy=np.asarray(bbox, dtype=np.float32),
        bbox_score=0.9,
    )


def test_complete_pose_may_touch_bottom_border_without_termination() -> None:
    gate = PersonFramePresenceGate()
    decision = gate.evaluate(
        make_pose(bbox=(250.0, 158.0, 427.0, 611.5)),
        image_width=816,
        image_height=612,
        keypoint_threshold=0.3,
    )

    assert decision.accepted
    assert decision.reason == "border_contact_but_pose_complete"
    assert decision.border_contacts == ("bottom",)
    assert decision.visible_foot_keypoint_count == 8
    assert not decision.track_reset_required


def test_complete_hallucinated_pose_at_top_border_is_rejected() -> None:
    gate = PersonFramePresenceGate()
    decision = gate.evaluate(
        make_pose(bbox=(250.0, 1.0, 427.0, 500.0)),
        image_width=816,
        image_height=612,
        keypoint_threshold=0.3,
    )

    assert not decision.accepted
    assert decision.reason == "partial_person_out_of_frame"
    assert "unsafe_side_or_top_border_contact" in decision.quality_failures
    assert decision.track_reset_required


def test_missing_feet_at_bottom_terminates_and_latches_track() -> None:
    gate = PersonFramePresenceGate()
    truncated = make_pose(
        bbox=(75.0, 129.0, 368.0, 611.9),
        valid_ids=set(range(17)),
    )
    first = gate.evaluate(
        truncated,
        image_width=816,
        image_height=612,
        keypoint_threshold=0.3,
    )

    assert not first.accepted
    assert first.reason == "partial_person_out_of_frame"
    assert "feet_missing_at_bottom_border" in first.quality_failures
    assert first.track_reset_required
    assert first.awaiting_full_reentry

    still_clipped = gate.evaluate(
        make_pose(bbox=(200.0, 120.0, 500.0, 611.0)),
        image_width=816,
        image_height=612,
        keypoint_threshold=0.3,
    )
    assert not still_clipped.accepted
    assert still_clipped.reason == "awaiting_full_reentry"
    assert not still_clipped.track_reset_required

    reentered = gate.evaluate(
        make_pose(bbox=(200.0, 120.0, 500.0, 590.0)),
        image_width=816,
        image_height=612,
        keypoint_threshold=0.3,
    )
    assert reentered.accepted
    assert reentered.reason == "fully_reentered"
    assert reentered.reacquired_after_exit
    assert not gate.awaiting_full_reentry


def test_no_detection_preserves_reentry_latch() -> None:
    gate = PersonFramePresenceGate()
    gate.evaluate(
        make_pose(
            bbox=(75.0, 129.0, 368.0, 611.9),
            valid_ids=set(range(17)),
        ),
        image_width=816,
        image_height=612,
        keypoint_threshold=0.3,
    )

    decision = gate.evaluate(
        None,
        image_width=816,
        image_height=612,
        keypoint_threshold=0.3,
    )

    assert not decision.accepted
    assert decision.reason == "no_detection"
    assert decision.awaiting_full_reentry


def test_frame_presence_config_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        FramePresenceConfig.from_mapping({"not_a_threshold": 1})
