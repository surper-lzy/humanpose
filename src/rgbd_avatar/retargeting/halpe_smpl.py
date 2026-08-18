"""Semantic Halpe26-to-SMPL retargeting with fixed avatar proportions.

Halpe points are observations on an image-space human skeleton; SMPL joints
are internal kinematic centres.  Matching equally named joints one-for-one
therefore transfers detector-dependent hip/shoulder widths and noisy limb
lengths into the avatar.  This module instead preserves a scaled SMPL rest
skeleton and transfers only root position, torso orientation, and bone
directions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence

import numpy as np


@dataclass(frozen=True)
class _SegmentSpec:
    name: str
    halpe_start: int
    halpe_end: int
    minimum_m: float
    maximum_m: float
    minimum_confidence: float = 0.35


_SEGMENTS: tuple[_SegmentSpec, ...] = (
    _SegmentSpec("torso", 19, 18, 0.20, 0.90),
    _SegmentSpec("hip_axis", 12, 11, 0.06, 0.50),
    _SegmentSpec("shoulder_axis", 6, 5, 0.12, 0.70),
    _SegmentSpec("neck_head", 18, 17, 0.08, 0.50),
    _SegmentSpec("left_thigh", 11, 13, 0.18, 0.75),
    _SegmentSpec("left_shin", 13, 15, 0.18, 0.75),
    _SegmentSpec("right_thigh", 12, 14, 0.18, 0.75),
    _SegmentSpec("right_shin", 14, 16, 0.18, 0.75),
    _SegmentSpec("left_upper_arm", 5, 7, 0.10, 0.55),
    _SegmentSpec("left_forearm", 7, 9, 0.10, 0.55),
    _SegmentSpec("right_upper_arm", 6, 8, 0.10, 0.55),
    _SegmentSpec("right_forearm", 8, 10, 0.10, 0.55),
    _SegmentSpec("left_foot_big", 24, 20, 0.04, 0.35, 0.45),
    _SegmentSpec("left_foot_small", 24, 22, 0.04, 0.35, 0.45),
    _SegmentSpec("right_foot_big", 25, 21, 0.04, 0.35, 0.45),
    _SegmentSpec("right_foot_small", 25, 23, 0.04, 0.35, 0.45),
)
_SEGMENT_BY_NAME = {spec.name: spec for spec in _SEGMENTS}


@dataclass(frozen=True)
class RobustLengthPrior:
    median_m: float
    lower_m: float
    upper_m: float
    sample_count: int

    def __post_init__(self) -> None:
        values = (self.median_m, self.lower_m, self.upper_m)
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError("Retarget length priors must be finite and positive.")
        if self.lower_m > self.median_m or self.median_m > self.upper_m:
            raise ValueError("Retarget length prior bounds are inconsistent.")
        if self.sample_count < 0:
            raise ValueError("Retarget prior sample count must be non-negative.")

    def accepts(self, length_m: float) -> bool:
        return bool(self.lower_m <= length_m <= self.upper_m)

    def to_mapping(self) -> dict[str, float | int]:
        return {
            "median_m": self.median_m,
            "lower_m": self.lower_m,
            "upper_m": self.upper_m,
            "sample_count": self.sample_count,
        }


@dataclass(frozen=True)
class HalpeSMPLRetargetProfile:
    length_priors: dict[str, RobustLengthPrior]

    def __post_init__(self) -> None:
        if set(self.length_priors) != set(_SEGMENT_BY_NAME):
            raise ValueError("Retarget profile does not cover every segment.")

    def classify(
        self,
        name: str,
        joints_m: np.ndarray,
        confidence: np.ndarray,
        usable: np.ndarray,
        predicted: np.ndarray,
    ) -> Literal["inlier", "soft_outlier", "invalid"]:
        spec = _SEGMENT_BY_NAME[name]
        start, end = spec.halpe_start, spec.halpe_end
        if not (usable[start] and usable[end]):
            return "invalid"
        if min(confidence[start], confidence[end]) < spec.minimum_confidence:
            return "invalid"
        vector = joints_m[end] - joints_m[start]
        length = float(np.linalg.norm(vector))
        if (
            not np.isfinite(vector).all()
            or not spec.minimum_m <= length <= spec.maximum_m
        ):
            return "invalid"
        if not self.length_priors[name].accepts(length):
            return "soft_outlier"
        return "inlier"

    def accepts(
        self,
        name: str,
        joints_m: np.ndarray,
        confidence: np.ndarray,
        usable: np.ndarray,
        predicted: np.ndarray,
    ) -> bool:
        return self.classify(
            name,
            joints_m,
            confidence,
            usable,
            predicted,
        ) == "inlier"

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "length_priors": {
                name: prior.to_mapping()
                for name, prior in sorted(self.length_priors.items())
            },
        }


@dataclass(frozen=True)
class RetargetedSMPLTargets:
    smpl_joint_indices: np.ndarray
    points_display_m: np.ndarray
    weights: np.ndarray
    smpl_direction_pairs: np.ndarray
    directions_display: np.ndarray
    direction_weights: np.ndarray
    rejected_segments: tuple[str, ...]
    soft_segments: tuple[str, ...]

    def __post_init__(self) -> None:
        indices = np.asarray(self.smpl_joint_indices, dtype=np.int64)
        points = np.asarray(self.points_display_m, dtype=np.float64)
        weights = np.asarray(self.weights, dtype=np.float64)
        pairs = np.asarray(self.smpl_direction_pairs, dtype=np.int64)
        directions = np.asarray(self.directions_display, dtype=np.float64)
        direction_weights = np.asarray(self.direction_weights, dtype=np.float64)
        if indices.ndim != 1 or points.shape != (len(indices), 3):
            raise ValueError("Retargeted positional targets have invalid shapes.")
        if weights.shape != (len(indices),):
            raise ValueError("Retargeted positional weights have invalid shape.")
        if pairs.shape != (len(pairs), 2):
            raise ValueError("Retargeted direction pairs must have shape Nx2.")
        if directions.shape != (len(pairs), 3):
            raise ValueError("Retargeted directions must have shape Nx3.")
        if direction_weights.shape != (len(pairs),):
            raise ValueError("Retargeted direction weights have invalid shape.")
        if (
            not np.isfinite(points).all()
            or not np.isfinite(weights).all()
            or not np.isfinite(directions).all()
            or not np.isfinite(direction_weights).all()
        ):
            raise ValueError("Retargeted targets must be finite.")
        if np.any(weights <= 0) or np.any(direction_weights <= 0):
            raise ValueError("Retargeted weights must be positive.")
        if len(directions) and not np.allclose(
            np.linalg.norm(directions, axis=1), 1.0, atol=1e-6
        ):
            raise ValueError("Retargeted directions must be normalized.")
        object.__setattr__(self, "smpl_joint_indices", indices)
        object.__setattr__(self, "points_display_m", points)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "smpl_direction_pairs", pairs)
        object.__setattr__(self, "directions_display", directions)
        object.__setattr__(self, "direction_weights", direction_weights)


def calibrate_halpe_smpl_profile(
    joints_sequence: Sequence[np.ndarray],
    confidence_sequence: Sequence[np.ndarray],
    usable_sequence: Sequence[np.ndarray],
    predicted_sequence: Sequence[np.ndarray],
    *,
    minimum_samples: int = 8,
) -> HalpeSMPLRetargetProfile:
    """Estimate conservative per-segment length gates from one sequence."""

    if minimum_samples <= 0:
        raise ValueError("minimum_samples must be positive.")
    if not (
        len(joints_sequence)
        == len(confidence_sequence)
        == len(usable_sequence)
        == len(predicted_sequence)
    ):
        raise ValueError("Retarget calibration sequences must have equal length.")

    priors: dict[str, RobustLengthPrior] = {}
    for spec in _SEGMENTS:
        samples: list[float] = []
        for joints_value, confidence_value, usable_value, predicted_value in zip(
            joints_sequence,
            confidence_sequence,
            usable_sequence,
            predicted_sequence,
        ):
            joints = np.asarray(joints_value, dtype=np.float64)
            confidence = np.asarray(confidence_value, dtype=np.float64)
            usable = np.asarray(usable_value, dtype=bool)
            predicted = np.asarray(predicted_value, dtype=bool)
            start, end = spec.halpe_start, spec.halpe_end
            if not (usable[start] and usable[end]):
                continue
            if predicted[start] or predicted[end]:
                continue
            if min(confidence[start], confidence[end]) < spec.minimum_confidence:
                continue
            length = float(np.linalg.norm(joints[end] - joints[start]))
            if spec.minimum_m <= length <= spec.maximum_m:
                samples.append(length)

        if len(samples) >= minimum_samples:
            values = np.asarray(samples, dtype=np.float64)
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            relative_tolerance = float(
                np.clip(3.0 * 1.4826 * mad / max(median, 1e-8), 0.12, 0.25)
            )
            lower = max(spec.minimum_m, median * (1.0 - relative_tolerance))
            upper = min(spec.maximum_m, median * (1.0 + relative_tolerance))
        else:
            median = 0.5 * (spec.minimum_m + spec.maximum_m)
            lower, upper = spec.minimum_m, spec.maximum_m
        priors[spec.name] = RobustLengthPrior(
            median_m=median,
            lower_m=lower,
            upper_m=upper,
            sample_count=len(samples),
        )
    return HalpeSMPLRetargetProfile(priors)


def _unit(vector: np.ndarray) -> np.ndarray | None:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(value).all() or norm <= 1e-8:
        return None
    return value / norm


def _body_basis(up_value: np.ndarray, lateral_value: np.ndarray) -> np.ndarray | None:
    up = _unit(up_value)
    lateral = _unit(lateral_value)
    if up is None or lateral is None:
        return None
    lateral = _unit(lateral - np.dot(lateral, up) * up)
    if lateral is None:
        return None
    forward = _unit(np.cross(up, lateral))
    if forward is None:
        return None
    lateral = _unit(np.cross(forward, up))
    assert lateral is not None
    basis = np.column_stack((lateral, forward, up))
    if np.linalg.det(basis) < 0.0:
        basis[:, 1] *= -1.0
    return basis


def retarget_halpe26_to_smpl(
    joints_display_m: np.ndarray,
    confidence: np.ndarray,
    usable: np.ndarray,
    predicted: np.ndarray,
    *,
    rest_joints_display_m: np.ndarray,
    profile: HalpeSMPLRetargetProfile,
    minimum_weight: float = 0.05,
    predicted_weight_scale: float = 0.25,
    soft_outlier_weight_scale: float = 0.25,
) -> RetargetedSMPLTargets:
    """Build fixed-proportion SMPL joint targets from Halpe directions."""

    joints = np.asarray(joints_display_m, dtype=np.float64)
    scores = np.asarray(confidence, dtype=np.float64)
    valid = np.asarray(usable, dtype=bool) & np.isfinite(joints).all(axis=1)
    predicted_mask = np.asarray(predicted, dtype=bool)
    rest = np.asarray(rest_joints_display_m, dtype=np.float64)
    if joints.shape != (26, 3) or scores.shape != (26,):
        raise ValueError("Halpe retarget arrays have invalid shapes.")
    if valid.shape != (26,) or predicted_mask.shape != (26,):
        raise ValueError("Halpe retarget masks have invalid shapes.")
    if rest.ndim != 2 or rest.shape[0] < 35 or rest.shape[1] != 3:
        raise ValueError("SMPL retarget rest joints must contain at least 35 XYZ rows.")
    if not np.isfinite(rest).all():
        raise ValueError("SMPL retarget rest joints must be finite.")
    if not 0.0 < minimum_weight <= 1.0:
        raise ValueError("minimum_weight must be in (0, 1].")
    if not 0.0 < predicted_weight_scale <= 1.0:
        raise ValueError("predicted_weight_scale must be in (0, 1].")
    if not 0.0 < soft_outlier_weight_scale <= 1.0:
        raise ValueError("soft_outlier_weight_scale must be in (0, 1].")

    rejected: list[str] = []
    softened: list[str] = []
    quality_cache: dict[tuple[str, bool], float | None] = {}

    def quality(name: str, *, strict: bool = False) -> float | None:
        cache_key = (name, strict)
        if cache_key in quality_cache:
            return quality_cache[cache_key]
        classification = profile.classify(
            name,
            joints,
            scores,
            valid,
            predicted_mask,
        )
        if classification == "invalid" or (
            strict and classification == "soft_outlier"
        ):
            rejected.append(name)
            result = None
        elif classification == "soft_outlier":
            softened.append(name)
            result = soft_outlier_weight_scale
        else:
            result = 1.0
        quality_cache[cache_key] = result
        return result

    def weight(indices: Sequence[int], scale: float = 1.0) -> float:
        value = float(np.mean(scores[np.asarray(indices, dtype=np.int64)]))
        value = float(np.clip(value, minimum_weight, 1.0)) * scale
        if any(predicted_mask[index] for index in indices):
            value *= predicted_weight_scale
        return max(minimum_weight, value)

    torso_quality = quality("torso")
    hip_axis_quality = quality("hip_axis")
    shoulder_axis_quality = quality("shoulder_axis")
    root: np.ndarray | None = None
    if valid[19] and scores[19] >= 0.35:
        root = joints[19]
    elif valid[11] and valid[12]:
        root = 0.5 * (joints[11] + joints[12])

    lateral_candidates: list[tuple[np.ndarray, float]] = []
    if hip_axis_quality is not None:
        lateral_candidates.append(
            (
                joints[11] - joints[12],
                weight((11, 12), hip_axis_quality),
            )
        )
    if shoulder_axis_quality is not None:
        lateral_candidates.append(
            (
                joints[5] - joints[6],
                weight((5, 6), shoulder_axis_quality),
            )
        )
    lateral = np.zeros(3, dtype=np.float64)
    for candidate, candidate_weight in lateral_candidates:
        direction = _unit(candidate)
        if direction is not None:
            lateral += candidate_weight * direction

    observed_basis = (
        _body_basis(joints[18] - joints[19], lateral)
        if torso_quality is not None and root is not None and lateral_candidates
        else None
    )
    rest_basis = _body_basis(
        rest[12] - rest[0],
        rest[1] - rest[2],
    )

    target_indices: list[int] = []
    target_points: list[np.ndarray] = []
    target_weights: list[float] = []
    direction_pairs: list[tuple[int, int]] = []
    target_directions: list[np.ndarray] = []
    direction_weights: list[float] = []
    target_by_joint: dict[int, np.ndarray] = {}

    def add_position(index: int, point: np.ndarray, point_weight: float) -> None:
        target_by_joint[index] = np.asarray(point, dtype=np.float64)
        target_indices.append(index)
        target_points.append(target_by_joint[index])
        target_weights.append(max(minimum_weight, point_weight))

    if root is not None and observed_basis is not None and rest_basis is not None:
        rotation = observed_basis @ rest_basis.T
        torso_indices = (0, 1, 2, 12, 16, 17)
        torso_weight = weight((19, 18), torso_quality)
        for index in torso_indices:
            point = root + rotation @ (rest[index] - rest[0])
            if index == 0:
                point_weight = weight((19,))
            elif index in (1, 2):
                point_weight = weight((11, 12), 0.65)
            elif index in (16, 17):
                point_weight = weight((5, 6), 0.65)
            else:
                point_weight = torso_weight
            add_position(index, point, point_weight)

    limb_specs = (
        ("left_thigh", 1, 4, 11, 13),
        ("left_shin", 4, 7, 13, 15),
        ("right_thigh", 2, 5, 12, 14),
        ("right_shin", 5, 8, 14, 16),
        ("left_upper_arm", 16, 18, 5, 7),
        ("left_forearm", 18, 20, 7, 9),
        ("right_upper_arm", 17, 19, 6, 8),
        ("right_forearm", 19, 21, 8, 10),
    )
    for name, smpl_start, smpl_end, halpe_start, halpe_end in limb_specs:
        segment_quality = quality(name)
        if smpl_start not in target_by_joint or segment_quality is None:
            continue
        direction = _unit(joints[halpe_end] - joints[halpe_start])
        if direction is None:
            continue
        rest_length = float(np.linalg.norm(rest[smpl_end] - rest[smpl_start]))
        add_position(
            smpl_end,
            target_by_joint[smpl_start] + rest_length * direction,
            weight((halpe_start, halpe_end), segment_quality),
        )

    neck_head_quality = quality("neck_head")
    if 12 in target_by_joint and neck_head_quality is not None:
        direction = _unit(joints[17] - joints[18])
        if direction is not None:
            add_position(
                15,
                target_by_joint[12]
                + float(np.linalg.norm(rest[15] - rest[12])) * direction,
                weight((18, 17), 0.45 * neck_head_quality),
            )

    foot_specs = (
        ("left_foot_big", 7, 31, 29, 24, 20),
        ("left_foot_small", 7, 31, 30, 24, 22),
        ("right_foot_big", 8, 34, 32, 25, 21),
        ("right_foot_small", 8, 34, 33, 25, 23),
    )
    for (
        name,
        required_ankle,
        smpl_start,
        smpl_end,
        halpe_start,
        halpe_end,
    ) in foot_specs:
        if required_ankle not in target_by_joint or quality(name, strict=True) is None:
            continue
        direction = _unit(joints[halpe_end] - joints[halpe_start])
        if direction is None:
            continue
        direction_pairs.append((smpl_start, smpl_end))
        target_directions.append(direction)
        direction_weights.append(weight((halpe_start, halpe_end), 0.65))

    return RetargetedSMPLTargets(
        smpl_joint_indices=np.asarray(target_indices, dtype=np.int64),
        points_display_m=np.asarray(target_points, dtype=np.float64).reshape(-1, 3),
        weights=np.asarray(target_weights, dtype=np.float64),
        smpl_direction_pairs=np.asarray(direction_pairs, dtype=np.int64).reshape(-1, 2),
        directions_display=np.asarray(target_directions, dtype=np.float64).reshape(-1, 3),
        direction_weights=np.asarray(direction_weights, dtype=np.float64),
        rejected_segments=tuple(dict.fromkeys(rejected)),
        soft_segments=tuple(dict.fromkeys(softened)),
    )
