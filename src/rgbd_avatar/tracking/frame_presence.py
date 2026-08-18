"""Frame-level person-presence decisions for truncated detections."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np

from rgbd_avatar.pose import Pose2D


FACE_KEYPOINT_IDS = (0, 1, 2, 3, 4)
FOOT_KEYPOINT_IDS = (15, 16, 20, 21, 22, 23, 24, 25)


@dataclass(frozen=True)
class FramePresenceConfig:
    """Thresholds for distinguishing truncation from a local occlusion."""

    enabled: bool = True
    border_margin_px: float = 2.0
    min_valid_keypoints_on_border: int = 20
    min_mean_keypoint_score_on_border: float = 0.55
    min_visible_face_keypoints_at_top: int = 2
    min_visible_foot_keypoints_at_bottom: int = 4
    reject_side_or_top_border_contact: bool = True
    latch_until_fully_inside: bool = True

    def __post_init__(self) -> None:
        if not np.isfinite(self.border_margin_px):
            raise ValueError("border_margin_px must be finite.")
        if self.border_margin_px < 0:
            raise ValueError("border_margin_px must be non-negative.")
        for name, value in (
            (
                "min_valid_keypoints_on_border",
                self.min_valid_keypoints_on_border,
            ),
            (
                "min_visible_face_keypoints_at_top",
                self.min_visible_face_keypoints_at_top,
            ),
            (
                "min_visible_foot_keypoints_at_bottom",
                self.min_visible_foot_keypoints_at_bottom,
            ),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative.")
        if not 0 <= self.min_mean_keypoint_score_on_border <= 1:
            raise ValueError(
                "min_mean_keypoint_score_on_border must be in [0, 1]."
            )

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any] | None,
    ) -> "FramePresenceConfig":
        if values is None:
            return cls()
        if not isinstance(values, Mapping):
            raise ValueError("tracking.frame_presence must be a mapping.")
        known = {field.name for field in cls.__dataclass_fields__.values()}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(
                "Unknown tracking.frame_presence keys: "
                + ", ".join(unknown)
            )
        return cls(**dict(values))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FramePresenceDecision:
    """One frame's acceptance decision and reproducible evidence."""

    accepted: bool
    reason: str
    border_contacts: tuple[str, ...] = ()
    quality_failures: tuple[str, ...] = ()
    valid_keypoint_count: int = 0
    visible_face_keypoint_count: int = 0
    visible_foot_keypoint_count: int = 0
    mean_keypoint_score: float | None = None
    bbox_xyxy: tuple[float, float, float, float] | None = None
    track_reset_required: bool = False
    awaiting_full_reentry: bool = False
    reacquired_after_exit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "border_contacts": list(self.border_contacts),
            "quality_failures": list(self.quality_failures),
            "valid_keypoint_count": self.valid_keypoint_count,
            "visible_face_keypoint_count": (
                self.visible_face_keypoint_count
            ),
            "visible_foot_keypoint_count": (
                self.visible_foot_keypoint_count
            ),
            "mean_keypoint_score": self.mean_keypoint_score,
            "bbox_xyxy": (
                list(self.bbox_xyxy)
                if self.bbox_xyxy is not None
                else None
            ),
            "track_reset_required": self.track_reset_required,
            "awaiting_full_reentry": self.awaiting_full_reentry,
            "reacquired_after_exit": self.reacquired_after_exit,
        }


class PersonFramePresenceGate:
    """Reject a truncated track and latch until it fully re-enters."""

    def __init__(
        self,
        config: FramePresenceConfig | None = None,
    ) -> None:
        self.config = config or FramePresenceConfig()
        self._awaiting_full_reentry = False

    @property
    def awaiting_full_reentry(self) -> bool:
        return self._awaiting_full_reentry

    def reset(self) -> None:
        self._awaiting_full_reentry = False

    def evaluate(
        self,
        pose: Pose2D | None,
        *,
        image_width: int,
        image_height: int,
        keypoint_threshold: float,
    ) -> FramePresenceDecision:
        if image_width <= 0 or image_height <= 0:
            raise ValueError("Image dimensions must be positive.")
        if not np.isfinite(keypoint_threshold):
            raise ValueError("keypoint_threshold must be finite.")
        if pose is None:
            return FramePresenceDecision(
                accepted=False,
                reason="no_detection",
                awaiting_full_reentry=self._awaiting_full_reentry,
            )
        if not self.config.enabled:
            return self._accepted_decision(
                pose,
                reason="gate_disabled",
                image_width=image_width,
                image_height=image_height,
                keypoint_threshold=keypoint_threshold,
            )

        valid = self._visible_keypoints(
            pose,
            image_width=image_width,
            image_height=image_height,
            keypoint_threshold=keypoint_threshold,
        )
        valid_count = int(np.count_nonzero(valid))
        face_count = int(np.count_nonzero(valid[list(FACE_KEYPOINT_IDS)]))
        foot_count = int(np.count_nonzero(valid[list(FOOT_KEYPOINT_IDS)]))
        contacts = self._border_contacts(
            pose.bbox_xyxy,
            image_width=image_width,
            image_height=image_height,
        )
        failures: list[str] = []
        if (
            self.config.reject_side_or_top_border_contact
            and any(side in contacts for side in ("left", "top", "right"))
        ):
            failures.append("unsafe_side_or_top_border_contact")
        if valid_count < self.config.min_valid_keypoints_on_border:
            failures.append("insufficient_valid_keypoints")
        if (
            pose.mean_score
            < self.config.min_mean_keypoint_score_on_border
        ):
            failures.append("low_mean_keypoint_score")
        if (
            "top" in contacts
            and face_count
            < self.config.min_visible_face_keypoints_at_top
        ):
            failures.append("face_missing_at_top_border")
        if (
            "bottom" in contacts
            and foot_count
            < self.config.min_visible_foot_keypoints_at_bottom
        ):
            failures.append("feet_missing_at_bottom_border")

        bbox = tuple(float(value) for value in pose.bbox_xyxy)
        common = {
            "border_contacts": contacts,
            "quality_failures": tuple(failures),
            "valid_keypoint_count": valid_count,
            "visible_face_keypoint_count": face_count,
            "visible_foot_keypoint_count": foot_count,
            "mean_keypoint_score": pose.mean_score,
            "bbox_xyxy": bbox,
        }
        truncation_detected = bool(contacts and failures)
        if truncation_detected:
            newly_terminated = not self._awaiting_full_reentry
            if self.config.latch_until_fully_inside:
                self._awaiting_full_reentry = True
            return FramePresenceDecision(
                accepted=False,
                reason="partial_person_out_of_frame",
                track_reset_required=newly_terminated,
                awaiting_full_reentry=self._awaiting_full_reentry,
                **common,
            )

        if (
            self._awaiting_full_reentry
            and self.config.latch_until_fully_inside
        ):
            if contacts:
                return FramePresenceDecision(
                    accepted=False,
                    reason="awaiting_full_reentry",
                    awaiting_full_reentry=True,
                    **common,
                )
            self._awaiting_full_reentry = False
            return FramePresenceDecision(
                accepted=True,
                reason="fully_reentered",
                reacquired_after_exit=True,
                **common,
            )

        return FramePresenceDecision(
            accepted=True,
            reason=(
                "border_contact_but_pose_complete"
                if contacts
                else "fully_inside"
            ),
            **common,
        )

    def _accepted_decision(
        self,
        pose: Pose2D,
        *,
        reason: str,
        image_width: int,
        image_height: int,
        keypoint_threshold: float,
    ) -> FramePresenceDecision:
        visible = self._visible_keypoints(
            pose,
            image_width=image_width,
            image_height=image_height,
            keypoint_threshold=keypoint_threshold,
        )
        return FramePresenceDecision(
            accepted=True,
            reason=reason,
            border_contacts=self._border_contacts(
                pose.bbox_xyxy,
                image_width=image_width,
                image_height=image_height,
            ),
            valid_keypoint_count=int(np.count_nonzero(visible)),
            visible_face_keypoint_count=int(
                np.count_nonzero(visible[list(FACE_KEYPOINT_IDS)])
            ),
            visible_foot_keypoint_count=int(
                np.count_nonzero(visible[list(FOOT_KEYPOINT_IDS)])
            ),
            mean_keypoint_score=pose.mean_score,
            bbox_xyxy=tuple(float(value) for value in pose.bbox_xyxy),
        )

    @staticmethod
    def _visible_keypoints(
        pose: Pose2D,
        *,
        image_width: int,
        image_height: int,
        keypoint_threshold: float,
    ) -> np.ndarray:
        points = pose.keypoints
        return (
            (pose.scores >= keypoint_threshold)
            & np.isfinite(points).all(axis=1)
            & (points[:, 0] >= 0)
            & (points[:, 0] < image_width)
            & (points[:, 1] >= 0)
            & (points[:, 1] < image_height)
        )

    def _border_contacts(
        self,
        bbox_xyxy: np.ndarray,
        *,
        image_width: int,
        image_height: int,
    ) -> tuple[str, ...]:
        x1, y1, x2, y2 = np.asarray(
            bbox_xyxy,
            dtype=np.float64,
        )
        margin = self.config.border_margin_px
        contacts = []
        if x1 <= margin:
            contacts.append("left")
        if y1 <= margin:
            contacts.append("top")
        if x2 >= image_width - margin:
            contacts.append("right")
        if y2 >= image_height - margin:
            contacts.append("bottom")
        return tuple(contacts)
