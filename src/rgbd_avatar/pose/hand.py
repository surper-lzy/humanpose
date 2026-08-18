"""Top-down RTMPose hand landmarks anchored by a Halpe26 body pose."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from rgbd_avatar.camera import CameraIntrinsics
from rgbd_avatar.depth.deprojection import deproject_pixel
from rgbd_avatar.depth.sampling import sample_joint_depth

from .models import Pose2D
from .rtmpose_backend import Device, resolve_device


HandSide = Literal["left", "right"]

HAND21_NAMES: tuple[str, ...] = (
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
)

HAND_TIP_INDICES: tuple[int, ...] = (4, 8, 12, 16, 20)
HAND21_LINKS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)
# Maximum plausible camera-Z change for each Hand21 link.  These are slightly
# tighter than anatomical 3D bone-length upper bounds because image-plane X/Y
# already accounts for part of every bone.  The root-to-MCP links are allowed
# more depth than the short distal phalanges.
HAND21_MAX_LINK_DEPTH_DELTA_M: tuple[float, ...] = (
    0.070, 0.055, 0.045, 0.040,  # thumb
    0.100, 0.060, 0.045, 0.040,  # index
    0.105, 0.065, 0.045, 0.040,  # middle
    0.100, 0.060, 0.045, 0.040,  # ring
    0.090, 0.055, 0.040, 0.035,  # pinky
)
# Conservative 3D upper bounds for the same links.  They are deliberately
# looser than adult anatomy so perspective/depth noise is tolerated, while
# grossly stretched fingers are still rejected.
HAND21_MAX_LINK_LENGTH_M: tuple[float, ...] = (
    0.110, 0.075, 0.060, 0.055,  # thumb
    0.140, 0.085, 0.060, 0.055,  # index
    0.140, 0.090, 0.065, 0.055,  # middle
    0.140, 0.085, 0.060, 0.055,  # ring
    0.130, 0.075, 0.055, 0.050,  # pinky
)
HALPE_ARM_INDICES: dict[HandSide, tuple[int, int]] = {
    "left": (7, 9),
    "right": (8, 10),
}


@dataclass(frozen=True)
class HandPose2D:
    side: HandSide
    keypoints: np.ndarray
    scores: np.ndarray
    bbox_xyxy: np.ndarray
    wrist_alignment_px: float = 0.0

    def __post_init__(self) -> None:
        keypoints = np.asarray(self.keypoints, dtype=np.float32)
        scores = np.asarray(self.scores, dtype=np.float32)
        bbox = np.asarray(self.bbox_xyxy, dtype=np.float32)
        if self.side not in ("left", "right"):
            raise ValueError(f"Unsupported hand side: {self.side!r}.")
        if keypoints.shape != (21, 2) or not np.isfinite(keypoints).all():
            raise ValueError("Hand keypoints must be a finite 21x2 array.")
        if scores.shape != (21,) or not np.isfinite(scores).all():
            raise ValueError("Hand scores must be a finite length-21 array.")
        if bbox.shape != (4,) or not np.isfinite(bbox).all():
            raise ValueError("Hand bounding box must contain four values.")
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            raise ValueError("Hand bounding box must have positive area.")
        if not np.isfinite(self.wrist_alignment_px) or self.wrist_alignment_px < 0:
            raise ValueError("Hand wrist alignment must be finite and non-negative.")
        object.__setattr__(self, "keypoints", keypoints)
        object.__setattr__(self, "scores", np.clip(scores, 0.0, 1.0))
        object.__setattr__(self, "bbox_xyxy", bbox)

    @property
    def mean_score(self) -> float:
        return float(np.mean(self.scores))

    def to_dict(self, *, score_threshold: float) -> dict:
        return {
            "side": self.side,
            "keypoint_format": "hand21",
            "bbox_xyxy": self.bbox_xyxy.astype(float).tolist(),
            "mean_keypoint_score": self.mean_score,
            "wrist_alignment_px": float(self.wrist_alignment_px),
            "keypoints": [
                {
                    "id": index,
                    "name": name,
                    "x": float(self.keypoints[index, 0]),
                    "y": float(self.keypoints[index, 1]),
                    "confidence": float(self.scores[index]),
                    "valid": bool(self.scores[index] >= score_threshold),
                }
                for index, name in enumerate(HAND21_NAMES)
            ],
        }


@dataclass(frozen=True)
class HandPose3D:
    side: HandSide
    joints_m: np.ndarray
    confidence: np.ndarray
    valid: np.ndarray
    depth_m: np.ndarray
    depth_confidence: np.ndarray

    def __post_init__(self) -> None:
        joints = np.asarray(self.joints_m, dtype=np.float32)
        confidence = np.asarray(self.confidence, dtype=np.float32)
        valid = np.asarray(self.valid, dtype=bool)
        depth = np.asarray(self.depth_m, dtype=np.float32)
        depth_confidence = np.asarray(
            self.depth_confidence,
            dtype=np.float32,
        )
        if self.side not in ("left", "right"):
            raise ValueError(f"Unsupported hand side: {self.side!r}.")
        if joints.shape != (21, 3):
            raise ValueError("Hand 3D joints must have shape 21x3.")
        for name, array in (
            ("confidence", confidence),
            ("valid", valid),
            ("depth_m", depth),
            ("depth_confidence", depth_confidence),
        ):
            if array.shape != (21,):
                raise ValueError(f"Hand {name} must have length 21.")
        if np.any(valid) and not np.isfinite(joints[valid]).all():
            raise ValueError("Valid hand joints must be finite.")
        if np.any(~valid) and not np.isnan(joints[~valid]).all():
            raise ValueError("Invalid hand joints must contain NaN XYZ.")
        object.__setattr__(self, "joints_m", joints)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "depth_m", depth)
        object.__setattr__(self, "depth_confidence", depth_confidence)

    def to_dict(self) -> dict:
        return {
            "side": self.side,
            "coordinate_system": "right_handed_camera_xyz_m",
            "joints": [
                {
                    "id": index,
                    "name": name,
                    "xyz_m": (
                        self.joints_m[index].astype(float).tolist()
                        if self.valid[index]
                        else None
                    ),
                    "confidence": float(self.confidence[index]),
                    "valid": bool(self.valid[index]),
                    "depth_m": (
                        float(self.depth_m[index])
                        if np.isfinite(self.depth_m[index])
                        else None
                    ),
                    "depth_confidence": float(
                        self.depth_confidence[index]
                    ),
                }
                for index, name in enumerate(HAND21_NAMES)
            ],
        }


def hand_observation_quality(
    joints_m: np.ndarray,
    valid: np.ndarray,
    *,
    minimum_extent_m: float = 0.035,
    maximum_extent_m: float = 0.25,
    minimum_palm_length_m: float = 0.035,
    maximum_palm_length_m: float = 0.16,
    minimum_palm_width_m: float = 0.025,
    maximum_palm_width_m: float = 0.13,
    minimum_palm_axis_sine: float = 0.40,
    minimum_link_length_m: float = 0.004,
    maximum_short_links: int = 1,
    maximum_link_length_scale: float = 1.35,
) -> tuple[bool, str, float | None]:
    """Reject missing, collapsed, or geometrically degenerate Hand21 data.

    The returned scalar remains the wrist-to-tip extent for compatibility.
    Palm checks are important for SMPL retargeting: a plausible fingertip
    extent alone does not guarantee that the MCP landmarks define a usable
    hand coordinate frame.
    """

    joints = np.asarray(joints_m, dtype=np.float64)
    usable = np.asarray(valid, dtype=bool)
    if joints.shape != (21, 3) or usable.shape != (21,):
        raise ValueError("Hand quality arrays must have shapes 21x3 and 21.")
    selected_tips = np.asarray((4, 12, 20), dtype=np.int64)
    available = selected_tips[usable[selected_tips]]
    if not usable[0]:
        return False, "missing_wrist", None
    if len(available) < 2:
        return False, "insufficient_fingertips", None
    extent = float(
        np.median(np.linalg.norm(joints[available] - joints[0], axis=1))
    )
    if extent < minimum_extent_m:
        return False, "collapsed_hand", extent
    if extent > maximum_extent_m:
        return False, "implausible_hand_extent", extent

    palm_indices = np.asarray((0, 5, 9, 17), dtype=np.int64)
    if not np.all(usable[palm_indices]):
        return False, "insufficient_palm_landmarks", extent
    palm_forward = joints[9] - joints[0]
    palm_lateral = joints[5] - joints[17]
    palm_length = float(np.linalg.norm(palm_forward))
    palm_width = float(np.linalg.norm(palm_lateral))
    if not minimum_palm_length_m <= palm_length <= maximum_palm_length_m:
        return False, "implausible_palm_length", extent
    if not minimum_palm_width_m <= palm_width <= maximum_palm_width_m:
        return False, "implausible_palm_width", extent
    palm_axis_sine = float(
        np.linalg.norm(np.cross(palm_forward, palm_lateral))
        / max(palm_length * palm_width, np.finfo(np.float64).eps)
    )
    if palm_axis_sine < minimum_palm_axis_sine:
        return False, "degenerate_palm_axes", extent

    link_lengths = np.asarray(
        [
            np.linalg.norm(joints[end] - joints[start])
            if usable[start] and usable[end]
            else np.nan
            for start, end in HAND21_LINKS
        ],
        dtype=np.float64,
    )
    if (
        np.count_nonzero(
            np.isfinite(link_lengths)
            & (link_lengths < minimum_link_length_m)
        )
        > maximum_short_links
    ):
        return False, "collapsed_finger_links", extent
    maximum_lengths = (
        np.asarray(HAND21_MAX_LINK_LENGTH_M, dtype=np.float64)
        * maximum_link_length_scale
    )
    if np.any(np.isfinite(link_lengths) & (link_lengths > maximum_lengths)):
        return False, "implausible_finger_link", extent
    return True, "ok", extent


@dataclass(frozen=True)
class RTMPoseHandBackendConfig:
    model_config: Path
    model_checkpoint: Path
    device: Device = "auto"
    body_keypoint_threshold: float = 0.3
    hand_keypoint_threshold: float = 0.10
    minimum_valid_keypoints: int = 8
    minimum_mean_score: float = 0.12
    crop_scale: float = 1.15
    crop_forward_offset: float = 0.20
    minimum_crop_px: float = 48.0
    maximum_wrist_offset_fraction: float = 0.32


def build_hand_bbox(
    body_pose: Pose2D,
    side: HandSide,
    *,
    image_width: int,
    image_height: int,
    keypoint_threshold: float,
    crop_scale: float = 1.15,
    forward_offset: float = 0.20,
    minimum_crop_px: float = 48.0,
) -> np.ndarray | None:
    """Estimate a square hand crop by extending elbow-to-wrist direction."""

    elbow_index, wrist_index = HALPE_ARM_INDICES[side]
    if (
        body_pose.scores[elbow_index] < keypoint_threshold
        or body_pose.scores[wrist_index] < keypoint_threshold
    ):
        return None
    elbow = body_pose.keypoints[elbow_index].astype(np.float64)
    wrist = body_pose.keypoints[wrist_index].astype(np.float64)
    forearm = wrist - elbow
    forearm_length = float(np.linalg.norm(forearm))
    if not np.isfinite(forearm_length) or forearm_length < 6.0:
        return None
    size = max(minimum_crop_px, crop_scale * forearm_length)
    center = wrist + forward_offset * forearm
    half = 0.5 * size
    bbox = np.array(
        [center[0] - half, center[1] - half,
         center[0] + half, center[1] + half],
        dtype=np.float32,
    )
    bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0.0, image_width - 1.0)
    bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0.0, image_height - 1.0)
    if bbox[2] - bbox[0] < 16.0 or bbox[3] - bbox[1] < 16.0:
        return None
    return bbox


class RTMPoseHandBackend:
    """Run one top-down Hand5 model on body-anchored left/right crops."""

    def __init__(self, config: RTMPoseHandBackendConfig) -> None:
        config_path = Path(config.model_config).expanduser().resolve()
        checkpoint_path = Path(config.model_checkpoint).expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"Hand config not found: {config_path}")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Hand checkpoint not found: {checkpoint_path}"
            )
        from mmpose.apis import init_model
        from mmpose.utils import register_all_modules

        register_all_modules()
        self.device = resolve_device(config.device)
        self.body_keypoint_threshold = float(
            config.body_keypoint_threshold
        )
        self.hand_keypoint_threshold = float(
            config.hand_keypoint_threshold
        )
        self.minimum_valid_keypoints = int(config.minimum_valid_keypoints)
        self.minimum_mean_score = float(config.minimum_mean_score)
        self.crop_scale = float(config.crop_scale)
        self.crop_forward_offset = float(config.crop_forward_offset)
        self.minimum_crop_px = float(config.minimum_crop_px)
        self.maximum_wrist_offset_fraction = float(
            config.maximum_wrist_offset_fraction
        )
        self._model = init_model(
            str(config_path),
            str(checkpoint_path),
            device=self.device,
        )

    def infer(
        self,
        image_bgr: np.ndarray,
        body_pose: Pose2D,
    ) -> dict[HandSide, HandPose2D]:
        from mmpose.apis import inference_topdown

        height, width = image_bgr.shape[:2]
        sides: list[HandSide] = []
        bboxes: list[np.ndarray] = []
        for side in ("left", "right"):
            bbox = build_hand_bbox(
                body_pose,
                side,
                image_width=width,
                image_height=height,
                keypoint_threshold=self.body_keypoint_threshold,
                crop_scale=self.crop_scale,
                forward_offset=self.crop_forward_offset,
                minimum_crop_px=self.minimum_crop_px,
            )
            if bbox is not None:
                sides.append(side)
                bboxes.append(bbox)
        if not bboxes:
            return {}

        samples = inference_topdown(
            self._model,
            image_bgr,
            bboxes=np.stack(bboxes),
            bbox_format="xyxy",
        )
        results: dict[HandSide, HandPose2D] = {}
        for side, bbox, sample in zip(sides, bboxes, samples, strict=True):
            instances = sample.pred_instances
            keypoints = np.asarray(instances.keypoints)[0, :, :2]
            scores = np.asarray(instances.keypoint_scores)[0]
            valid_count = int(
                np.count_nonzero(scores >= self.hand_keypoint_threshold)
            )
            _, body_wrist_index = HALPE_ARM_INDICES[side]
            body_wrist = body_pose.keypoints[body_wrist_index]
            wrist_alignment_px = float(
                np.linalg.norm(keypoints[0] - body_wrist)
            )
            crop_size = float(max(bbox[2] - bbox[0], bbox[3] - bbox[1]))
            if (
                keypoints.shape != (21, 2)
                or scores.shape != (21,)
                or scores[0] < self.hand_keypoint_threshold
                or valid_count < self.minimum_valid_keypoints
                or float(np.mean(scores)) < self.minimum_mean_score
                or wrist_alignment_px
                > self.maximum_wrist_offset_fraction * crop_size
            ):
                continue
            keypoints = keypoints + (body_wrist - keypoints[0])
            results[side] = HandPose2D(
                side=side,
                keypoints=keypoints,
                scores=scores,
                bbox_xyxy=bbox,
                wrist_alignment_px=wrist_alignment_px,
            )
        return results


def recover_hand_pose3d(
    pose2d: HandPose2D,
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    keypoint_threshold: float = 0.10,
    radius: int = 2,
    min_depth_m: float = 0.2,
    max_depth_m: float = 8.0,
    anchor_depth_m: float | None = None,
    max_anchor_delta_m: float = 0.12,
    minimum_sample_confidence: float = 0.20,
    fallback_depth_confidence: float = 0.35,
    topology_depth_gate: bool = True,
    anchor_point_m: np.ndarray | None = None,
) -> HandPose3D:
    """Lift Hand21 while keeping all depths relative to one wrist surface.

    The body-constrained wrist may differ slightly from the depth observed at
    the wrist pixel.  Samples are therefore checked against the *observed*
    wrist surface first, then the whole hand is translated once to the body
    wrist.  Using the constrained depth as a per-joint fallback before that
    translation would apply the wrist correction twice.
    """

    if not 0.0 <= minimum_sample_confidence <= 1.0:
        raise ValueError("Minimum hand depth confidence must be in [0, 1].")
    joints = np.full((21, 3), np.nan, dtype=np.float32)
    confidence = np.zeros(21, dtype=np.float32)
    valid = np.zeros(21, dtype=bool)
    sampled_depth = np.full(21, np.nan, dtype=np.float32)
    depth_confidence = np.zeros(21, dtype=np.float32)
    samples = [None] * 21
    for index, ((u, v), score) in enumerate(
        zip(pose2d.keypoints, pose2d.scores, strict=True)
    ):
        if score < keypoint_threshold:
            continue
        samples[index] = sample_joint_depth(
            depth_m,
            float(u),
            float(v),
            radius=radius,
            min_depth_m=min_depth_m,
            max_depth_m=max_depth_m,
        )

    has_anchor_depth = bool(
        anchor_depth_m is not None
        and np.isfinite(anchor_depth_m)
        and anchor_depth_m > 0
    )
    wrist_sample = samples[0]
    wrist_sample_usable = bool(
        wrist_sample is not None
        and wrist_sample.confidence >= minimum_sample_confidence
        and (
            not has_anchor_depth
            or abs(wrist_sample.depth_m - float(anchor_depth_m))
            <= max_anchor_delta_m
        )
    )
    reference_depth_m = (
        float(wrist_sample.depth_m)
        if wrist_sample_usable
        else (float(anchor_depth_m) if has_anchor_depth else None)
    )

    for index, ((u, v), score, sample) in enumerate(
        zip(pose2d.keypoints, pose2d.scores, samples, strict=True)
    ):
        if score < keypoint_threshold:
            continue
        sample_usable = bool(
            sample is not None
            and sample.confidence >= minimum_sample_confidence
            and (
                reference_depth_m is None
                or abs(sample.depth_m - reference_depth_m)
                <= max_anchor_delta_m
            )
        )
        if sample_usable:
            resolved_depth = float(sample.depth_m)
            resolved_depth_confidence = float(sample.confidence)
        elif reference_depth_m is not None:
            resolved_depth = reference_depth_m
            resolved_depth_confidence = fallback_depth_confidence
        else:
            continue
        sampled_depth[index] = resolved_depth
        depth_confidence[index] = resolved_depth_confidence
        confidence[index] = float(score) * resolved_depth_confidence
        valid[index] = True

    if topology_depth_gate:
        for (parent, child), maximum_delta in zip(
            HAND21_LINKS,
            HAND21_MAX_LINK_DEPTH_DELTA_M,
            strict=True,
        ):
            if not (valid[parent] and valid[child]):
                continue
            if (
                abs(sampled_depth[child] - sampled_depth[parent])
                <= maximum_delta
            ):
                continue
            sampled_depth[child] = sampled_depth[parent]
            depth_confidence[child] = fallback_depth_confidence
            confidence[child] = (
                float(pose2d.scores[child]) * fallback_depth_confidence
            )

    for index in np.flatnonzero(valid):
        u, v = pose2d.keypoints[index]
        joints[index] = deproject_pixel(
            float(u),
            float(v),
            float(sampled_depth[index]),
            intrinsics,
        )
    if anchor_point_m is not None:
        anchor = np.asarray(anchor_point_m, dtype=np.float32)
        if anchor.shape != (3,) or not np.isfinite(anchor).all() or anchor[2] <= 0:
            raise ValueError("Hand anchor point must be a finite positive-Z XYZ.")
        if valid[0]:
            joints[valid] += anchor - joints[0]
            sampled_depth[valid] = joints[valid, 2]
    return HandPose3D(
        side=pose2d.side,
        joints_m=joints,
        confidence=confidence,
        valid=valid,
        depth_m=sampled_depth,
        depth_confidence=depth_confidence,
    )
