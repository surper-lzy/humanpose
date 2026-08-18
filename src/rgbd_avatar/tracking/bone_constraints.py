"""Robust bone calibration and confidence-aware metric pose constraints."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from rgbd_avatar.pose import (
    HALPE26_CONSTRAINT_LINKS,
    HALPE26_CONSTRAINT_TOLERANCE_RATIOS,
    HALPE26_NAMES,
    Pose3D,
)

from .one_euro import TemporalPose3D


@dataclass(frozen=True)
class BoneLengthPrior:
    """Per-link targets estimated only from unfiltered real observations."""

    links: tuple[tuple[int, int], ...]
    target_lengths_m: np.ndarray
    sample_count: np.ndarray
    relative_mad: np.ndarray
    tolerance_ratio: np.ndarray
    ready: np.ndarray
    frozen: np.ndarray

    def __post_init__(self) -> None:
        bone_count = len(self.links)
        for name, values, dtype in (
            ("target_lengths_m", self.target_lengths_m, np.float64),
            ("sample_count", self.sample_count, np.int64),
            ("relative_mad", self.relative_mad, np.float64),
            ("tolerance_ratio", self.tolerance_ratio, np.float64),
            ("ready", self.ready, bool),
            ("frozen", self.frozen, bool),
        ):
            array = np.asarray(values, dtype=dtype)
            if array.shape != (bone_count,):
                raise ValueError(
                    f"Expected {name} shape {(bone_count,)}, got {array.shape}."
                )
            object.__setattr__(self, name, array.copy())


class BoneLengthCalibrator:
    """Build a temporary single-person prior from raw RGB-D observations."""

    def __init__(
        self,
        links: Sequence[tuple[int, int]] = HALPE26_CONSTRAINT_LINKS,
        tolerance_ratios: Sequence[float] | None = None,
        min_samples_per_bone: int = 12,
        target_samples_per_bone: int = 20,
        max_samples_per_bone: int = 30,
        min_keypoint_confidence: float = 0.6,
        min_depth_confidence: float = 0.7,
        max_relative_mad: float = 0.10,
        outlier_relative_tolerance: float = 0.12,
        outlier_absolute_tolerance_m: float = 0.02,
        min_length_m: float = 0.03,
        max_length_m: float = 1.2,
    ) -> None:
        if min_samples_per_bone <= 0:
            raise ValueError("min_samples_per_bone must be positive.")
        if target_samples_per_bone < min_samples_per_bone:
            raise ValueError(
                "target_samples_per_bone must be at least "
                "min_samples_per_bone."
            )
        if max_samples_per_bone < target_samples_per_bone:
            raise ValueError(
                "max_samples_per_bone must be at least "
                "target_samples_per_bone."
            )
        if min_keypoint_confidence < 0 or min_depth_confidence < 0:
            raise ValueError("Calibration confidence gates must be non-negative.")
        if max_relative_mad < 0:
            raise ValueError("max_relative_mad must be non-negative.")
        if outlier_relative_tolerance <= 0:
            raise ValueError(
                "outlier_relative_tolerance must be positive."
            )
        if outlier_absolute_tolerance_m <= 0:
            raise ValueError(
                "outlier_absolute_tolerance_m must be positive."
            )
        if min_length_m <= 0 or max_length_m <= min_length_m:
            raise ValueError("Bone length limits are invalid.")

        self.links = tuple(links)
        if tolerance_ratios is None:
            tolerance_ratios = (
                HALPE26_CONSTRAINT_TOLERANCE_RATIOS
                if self.links == HALPE26_CONSTRAINT_LINKS
                else (0.05,) * len(self.links)
            )
        self.tolerance_ratios = np.asarray(
            tolerance_ratios, dtype=np.float64
        )
        if self.tolerance_ratios.shape != (len(self.links),):
            raise ValueError(
                "tolerance_ratios must contain one value per link."
            )
        if np.any(
            (self.tolerance_ratios < 0)
            | (self.tolerance_ratios >= 1)
        ):
            raise ValueError("All tolerance ratios must be in [0, 1).")

        self.min_samples_per_bone = int(min_samples_per_bone)
        self.target_samples_per_bone = int(target_samples_per_bone)
        self.max_samples_per_bone = int(max_samples_per_bone)
        self.min_keypoint_confidence = float(min_keypoint_confidence)
        self.min_depth_confidence = float(min_depth_confidence)
        self.max_relative_mad = float(max_relative_mad)
        self.outlier_relative_tolerance = float(
            outlier_relative_tolerance
        )
        self.outlier_absolute_tolerance_m = float(
            outlier_absolute_tolerance_m
        )
        self.min_length_m = float(min_length_m)
        self.max_length_m = float(max_length_m)
        self.reset()

    def reset(self) -> None:
        """Clear the profile when the person/track identity changes."""

        bone_count = len(self.links)
        self._samples: list[list[float]] = [[] for _ in self.links]
        self._invalid_rejections = np.zeros(bone_count, dtype=np.int64)
        self._frozen_targets = np.full(bone_count, np.nan, dtype=np.float64)
        self._frozen_counts = np.zeros(bone_count, dtype=np.int64)
        self._frozen_relative_mad = np.full(
            bone_count, np.nan, dtype=np.float64
        )

    def update(
        self,
        pose3d_raw: Pose3D,
        keypoint_confidence: np.ndarray,
    ) -> None:
        """Consume raw observed coordinates, never predictions or constraints."""

        keypoint_scores = np.asarray(
            keypoint_confidence, dtype=np.float64
        )
        if keypoint_scores.shape != (len(HALPE26_NAMES),):
            raise ValueError(
                "keypoint_confidence must contain one score per Halpe26 joint."
            )

        for bone_index, (start, end) in enumerate(self.links):
            if np.isfinite(self._frozen_targets[bone_index]):
                continue
            samples = self._samples[bone_index]
            if len(samples) >= self.max_samples_per_bone:
                continue
            if (
                not pose3d_raw.valid[start]
                or not pose3d_raw.valid[end]
                or keypoint_scores[start] < self.min_keypoint_confidence
                or keypoint_scores[end] < self.min_keypoint_confidence
                or pose3d_raw.depth_confidence[start]
                < self.min_depth_confidence
                or pose3d_raw.depth_confidence[end]
                < self.min_depth_confidence
                or not np.isfinite(
                    pose3d_raw.joints_m[[start, end]]
                ).all()
            ):
                continue

            length_m = float(
                np.linalg.norm(
                    pose3d_raw.joints_m[end] - pose3d_raw.joints_m[start]
                )
            )
            if (
                not np.isfinite(length_m)
                or length_m < self.min_length_m
                or length_m > self.max_length_m
            ):
                self._invalid_rejections[bone_index] += 1
                continue
            samples.append(length_m)
            profile = self._robust_profile(samples)
            if (
                profile["ready"]
                and profile["sample_count"]
                >= self.target_samples_per_bone
            ):
                self._frozen_targets[bone_index] = profile["target_length_m"]
                self._frozen_counts[bone_index] = profile["sample_count"]
                self._frozen_relative_mad[bone_index] = profile[
                    "relative_mad"
                ]

    def _robust_profile(self, values: Sequence[float]) -> dict[str, Any]:
        samples = np.asarray(values, dtype=np.float64)
        if not samples.size:
            return {
                "target_length_m": np.nan,
                "sample_count": 0,
                "outlier_count": 0,
                "relative_mad": np.nan,
                "ready": False,
            }
        initial_median = float(np.median(samples))
        gate_m = max(
            self.outlier_absolute_tolerance_m,
            self.outlier_relative_tolerance * initial_median,
        )
        inliers = samples[np.abs(samples - initial_median) <= gate_m]
        if not inliers.size:
            return {
                "target_length_m": np.nan,
                "sample_count": 0,
                "outlier_count": int(samples.size),
                "relative_mad": np.nan,
                "ready": False,
            }
        target_m = float(np.median(inliers))
        mad_m = float(np.median(np.abs(inliers - target_m)))
        relative_mad = mad_m / target_m if target_m > 0 else np.inf
        return {
            "target_length_m": target_m,
            "sample_count": int(inliers.size),
            "outlier_count": int(samples.size - inliers.size),
            "relative_mad": relative_mad,
            "ready": bool(
                inliers.size >= self.min_samples_per_bone
                and relative_mad <= self.max_relative_mad
            ),
        }

    def prior(self) -> BoneLengthPrior:
        bone_count = len(self.links)
        targets = np.full(bone_count, np.nan, dtype=np.float64)
        counts = np.zeros(bone_count, dtype=np.int64)
        relative_mad = np.full(bone_count, np.nan, dtype=np.float64)
        ready = np.zeros(bone_count, dtype=bool)
        frozen = np.isfinite(self._frozen_targets)
        for index, values in enumerate(self._samples):
            if frozen[index]:
                targets[index] = self._frozen_targets[index]
                counts[index] = self._frozen_counts[index]
                relative_mad[index] = self._frozen_relative_mad[index]
                ready[index] = True
                continue
            profile = self._robust_profile(values)
            targets[index] = profile["target_length_m"]
            counts[index] = profile["sample_count"]
            relative_mad[index] = profile["relative_mad"]
            ready[index] = profile["ready"]
        return BoneLengthPrior(
            links=self.links,
            target_lengths_m=targets,
            sample_count=counts,
            relative_mad=relative_mad,
            tolerance_ratio=self.tolerance_ratios,
            ready=ready,
            frozen=frozen,
        )

    def summary(self) -> dict[str, Any]:
        prior = self.prior()
        bones = []
        for index, (start, end) in enumerate(self.links):
            profile = self._robust_profile(self._samples[index])
            bones.append(
                {
                    "start_id": start,
                    "start_name": HALPE26_NAMES[start],
                    "end_id": end,
                    "end_name": HALPE26_NAMES[end],
                    "observation_count": len(self._samples[index]),
                    "sample_count": int(prior.sample_count[index]),
                    "outlier_count": int(profile["outlier_count"]),
                    "invalid_rejection_count": int(
                        self._invalid_rejections[index]
                    ),
                    "target_length_m": (
                        float(prior.target_lengths_m[index])
                        if np.isfinite(prior.target_lengths_m[index])
                        else None
                    ),
                    "relative_mad": (
                        float(prior.relative_mad[index])
                        if np.isfinite(prior.relative_mad[index])
                        else None
                    ),
                    "tolerance_ratio": float(
                        prior.tolerance_ratio[index]
                    ),
                    "ready": bool(prior.ready[index]),
                    "frozen": bool(prior.frozen[index]),
                }
            )
        return {
            "profile_scope": "temporary_single_person_sequence",
            "bone_count": len(self.links),
            "ready_bone_count": int(np.count_nonzero(prior.ready)),
            "frozen_bone_count": int(np.count_nonzero(prior.frozen)),
            "min_samples_per_bone": self.min_samples_per_bone,
            "target_samples_per_bone": self.target_samples_per_bone,
            "max_samples_per_bone": self.max_samples_per_bone,
            "min_keypoint_confidence": self.min_keypoint_confidence,
            "min_depth_confidence": self.min_depth_confidence,
            "max_relative_mad": self.max_relative_mad,
            "outlier_relative_tolerance": (
                self.outlier_relative_tolerance
            ),
            "outlier_absolute_tolerance_m": (
                self.outlier_absolute_tolerance_m
            ),
            "bones": bones,
        }


@dataclass
class BoneConstraintResult:
    """Constrained pose and diagnostics preserving observation provenance."""

    pose: TemporalPose3D
    corrected: np.ndarray
    correction_m: np.ndarray
    diagnostics: dict[str, Any]

    def __post_init__(self) -> None:
        count = len(HALPE26_NAMES)
        self.corrected = np.asarray(self.corrected, dtype=bool)
        self.correction_m = np.asarray(
            self.correction_m, dtype=np.float32
        )
        if self.corrected.shape != (count,):
            raise ValueError(
                f"Expected corrected shape {(count,)}, "
                f"got {self.corrected.shape}."
            )
        if self.correction_m.shape != (count,):
            raise ValueError(
                f"Expected correction_m shape {(count,)}, "
                f"got {self.correction_m.shape}."
            )

    def to_dict(self) -> dict[str, Any]:
        payload = self.pose.to_dict()
        payload["constraint"] = self.diagnostics
        for index, joint in enumerate(payload["joints"]):
            joint["corrected"] = bool(self.corrected[index])
            joint["correction_m"] = float(self.correction_m[index])
        return payload


class BoneLengthConstraint:
    """Weighted Jacobi projection with configurable observed-joint anchors."""

    def __init__(
        self,
        anchor_confidence: float = 0.55,
        iterations: int = 6,
        max_joint_correction_m: float = 0.15,
        max_predicted_correction_m: float | None = None,
        fixed_joint_indices: Sequence[int] = (19,),
        project_observed: bool = False,
    ) -> None:
        if anchor_confidence <= 0:
            raise ValueError("anchor_confidence must be positive.")
        if iterations <= 0:
            raise ValueError("iterations must be positive.")
        if max_joint_correction_m <= 0:
            raise ValueError("max_joint_correction_m must be positive.")
        if (
            max_predicted_correction_m is not None
            and max_predicted_correction_m <= 0
        ):
            raise ValueError(
                "max_predicted_correction_m must be positive."
            )
        if any(
            index < 0 or index >= len(HALPE26_NAMES)
            for index in fixed_joint_indices
        ):
            raise ValueError("fixed_joint_indices contains an invalid joint.")
        self.anchor_confidence = float(anchor_confidence)
        self.iterations = int(iterations)
        self.max_joint_correction_m = float(max_joint_correction_m)
        self.max_predicted_correction_m = float(
            max_predicted_correction_m
            if max_predicted_correction_m is not None
            else max_joint_correction_m
        )
        self.fixed_joint_indices = tuple(int(x) for x in fixed_joint_indices)
        self.project_observed = bool(project_observed)

    def apply(
        self,
        pose: TemporalPose3D,
        prior: BoneLengthPrior,
    ) -> BoneConstraintResult:
        if not prior.links:
            raise ValueError("BoneLengthPrior must contain at least one link.")
        original = pose.joints_m.astype(np.float64)
        positions = original.copy()

        predicted_movable = pose.usable & pose.predicted
        observed_movable = (
            pose.usable
            & pose.observed
            & (
                (pose.confidence < self.anchor_confidence)
                | self.project_observed
            )
        )
        movable = predicted_movable | observed_movable
        fixed_root = np.zeros(len(HALPE26_NAMES), dtype=bool)
        fixed_root[list(self.fixed_joint_indices)] = True
        fixed_root &= pose.usable
        movable[fixed_root] = False
        anchors = pose.usable & ~movable

        mobility = np.zeros(len(HALPE26_NAMES), dtype=np.float64)
        mobility[predicted_movable & ~fixed_root] = 1.0
        low_observed = observed_movable & ~fixed_root
        if np.any(low_observed):
            normalized = (
                self.anchor_confidence - pose.confidence[low_observed]
            ) / self.anchor_confidence
            mobility[low_observed] = np.maximum(
                0.05, normalized.astype(np.float64)
            )

        ready_usable = np.zeros(len(prior.links), dtype=bool)
        for index, (start, end) in enumerate(prior.links):
            ready_usable[index] = (
                prior.ready[index]
                and pose.usable[start]
                and pose.usable[end]
                and np.isfinite(positions[[start, end]]).all()
            )
        before = self._residual_statistics(positions, prior, ready_usable)
        projectable = np.asarray(
            [
                ready_usable[index]
                and mobility[start] + mobility[end] > 0
                for index, (start, end) in enumerate(prior.links)
            ],
            dtype=bool,
        )
        anchor_only = ready_usable & ~projectable
        projectable_before = self._residual_statistics(
            positions, prior, projectable
        )
        anchor_only_residual = self._residual_statistics(
            positions, prior, anchor_only
        )
        before_directions = self._unit_directions(
            positions, prior.links, ready_usable
        )
        applied_edges = np.zeros(len(prior.links), dtype=bool)

        for _ in range(self.iterations):
            snapshot = positions.copy()
            accumulated = np.zeros_like(positions)
            contribution_count = np.zeros(
                len(HALPE26_NAMES), dtype=np.float64
            )
            for bone_index, (start, end) in enumerate(prior.links):
                if not ready_usable[bone_index]:
                    continue
                start_mobility = mobility[start]
                end_mobility = mobility[end]
                mobility_sum = start_mobility + end_mobility
                if mobility_sum <= 0:
                    continue
                delta = snapshot[end] - snapshot[start]
                distance_m = float(np.linalg.norm(delta))
                if not np.isfinite(distance_m) or distance_m <= 1e-9:
                    continue
                target_m = float(prior.target_lengths_m[bone_index])
                tolerance = float(prior.tolerance_ratio[bone_index])
                lower_m = target_m * (1.0 - tolerance)
                upper_m = target_m * (1.0 + tolerance)
                if lower_m <= distance_m <= upper_m:
                    continue
                desired_m = lower_m if distance_m < lower_m else upper_m
                signed_error_m = distance_m - desired_m
                correction = signed_error_m * delta / distance_m
                if start_mobility > 0:
                    accumulated[start] += (
                        start_mobility / mobility_sum
                    ) * correction
                    contribution_count[start] += 1.0
                if end_mobility > 0:
                    accumulated[end] -= (
                        end_mobility / mobility_sum
                    ) * correction
                    contribution_count[end] += 1.0
                applied_edges[bone_index] = True
            has_updates = contribution_count > 0
            positions[has_updates] += (
                accumulated[has_updates]
                / contribution_count[has_updates, None]
            )
            positions[anchors] = original[anchors]

        offsets = positions - original
        offset_norm = np.zeros(len(HALPE26_NAMES), dtype=np.float64)
        finite_offsets = np.isfinite(offsets).all(axis=1)
        offset_norm[finite_offsets] = np.linalg.norm(
            offsets[finite_offsets], axis=1
        )
        correction_limits = np.full(
            len(HALPE26_NAMES),
            self.max_joint_correction_m,
            dtype=np.float64,
        )
        correction_limits[predicted_movable] = (
            self.max_predicted_correction_m
        )
        correction_limited = movable & (
            offset_norm > correction_limits
        )
        for index in np.flatnonzero(correction_limited):
            offsets[index] *= (
                correction_limits[index] / offset_norm[index]
            )
            positions[index] = original[index] + offsets[index]
        positions[anchors] = original[anchors]

        offsets = positions - original
        correction_m = np.zeros(len(HALPE26_NAMES), dtype=np.float64)
        finite_offsets = np.isfinite(offsets).all(axis=1)
        correction_m[finite_offsets] = np.linalg.norm(
            offsets[finite_offsets], axis=1
        )
        corrected = movable & (correction_m > 1e-6)
        after = self._residual_statistics(positions, prior, ready_usable)
        projectable_after = self._residual_statistics(
            positions, prior, projectable
        )
        after_directions = self._unit_directions(
            positions, prior.links, ready_usable
        )
        direction_stats = self._direction_statistics(
            before_directions[applied_edges],
            after_directions[applied_edges],
        )

        constrained_pose = TemporalPose3D(
            joints_m=positions,
            confidence=pose.confidence.copy(),
            usable=pose.usable.copy(),
            observed=pose.observed.copy(),
            predicted=pose.predicted.copy(),
            age_s=pose.age_s.copy(),
            reset_occurred=pose.reset_occurred,
        )
        diagnostics = {
            "calibrated_bone_count": int(np.count_nonzero(prior.ready)),
            "usable_calibrated_bone_count": int(
                np.count_nonzero(ready_usable)
            ),
            "applied_bone_count": int(np.count_nonzero(applied_edges)),
            "anchor_joint_count": int(np.count_nonzero(anchors)),
            "movable_joint_count": int(np.count_nonzero(movable)),
            "corrected_joint_count": int(np.count_nonzero(corrected)),
            "correction_limited_joint_count": int(
                np.count_nonzero(correction_limited)
            ),
            "mean_correction_m": (
                float(np.mean(correction_m[corrected]))
                if np.any(corrected)
                else 0.0
            ),
            "max_correction_m": (
                float(np.max(correction_m[corrected]))
                if np.any(corrected)
                else 0.0
            ),
            "max_anchor_displacement_m": (
                float(np.max(correction_m[anchors]))
                if np.any(anchors)
                else 0.0
            ),
            "root_joint_ids": list(self.fixed_joint_indices),
            "project_observed": self.project_observed,
            "max_root_displacement_m": (
                float(np.max(correction_m[list(self.fixed_joint_indices)]))
                if self.fixed_joint_indices
                else 0.0
            ),
            "iterations": self.iterations,
            "max_observed_correction_m": self.max_joint_correction_m,
            "max_predicted_correction_m": (
                self.max_predicted_correction_m
            ),
            "residual_before": before,
            "residual_after": after,
            "projectable_residual_before": projectable_before,
            "projectable_residual_after": projectable_after,
            "anchor_only_residual": anchor_only_residual,
            "direction_change": direction_stats,
        }
        return BoneConstraintResult(
            pose=constrained_pose,
            corrected=corrected,
            correction_m=correction_m,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _unit_directions(
        positions: np.ndarray,
        links: tuple[tuple[int, int], ...],
        enabled: np.ndarray,
    ) -> np.ndarray:
        directions = np.full((len(links), 3), np.nan, dtype=np.float64)
        for index, (start, end) in enumerate(links):
            if not enabled[index]:
                continue
            delta = positions[end] - positions[start]
            norm = float(np.linalg.norm(delta))
            if np.isfinite(norm) and norm > 1e-9:
                directions[index] = delta / norm
        return directions

    @staticmethod
    def _direction_statistics(
        before: np.ndarray,
        after: np.ndarray,
    ) -> dict[str, float | int | None]:
        valid = np.isfinite(before).all(axis=1) & np.isfinite(after).all(axis=1)
        if not np.any(valid):
            return {
                "bone_count": 0,
                "median_angle_deg": None,
                "p95_angle_deg": None,
                "max_angle_deg": None,
                "flipped_bone_count": 0,
            }
        dots = np.sum(before[valid] * after[valid], axis=1)
        clipped = np.clip(dots, -1.0, 1.0)
        angles = np.degrees(np.arccos(clipped))
        return {
            "bone_count": int(angles.size),
            "median_angle_deg": float(np.median(angles)),
            "p95_angle_deg": float(np.percentile(angles, 95)),
            "max_angle_deg": float(np.max(angles)),
            "flipped_bone_count": int(np.count_nonzero(dots < 0)),
        }

    @staticmethod
    def _residual_statistics(
        positions: np.ndarray,
        prior: BoneLengthPrior,
        enabled: np.ndarray,
    ) -> dict[str, float | int | None]:
        relative_errors: list[float] = []
        violations: list[float] = []
        for bone_index, (start, end) in enumerate(prior.links):
            if not enabled[bone_index]:
                continue
            target_m = float(prior.target_lengths_m[bone_index])
            distance_m = float(
                np.linalg.norm(positions[end] - positions[start])
            )
            if (
                not np.isfinite(distance_m)
                or not np.isfinite(target_m)
                or target_m <= 0
            ):
                continue
            relative_error = abs(distance_m - target_m) / target_m
            relative_errors.append(relative_error)
            violations.append(
                max(
                    0.0,
                    relative_error
                    - float(prior.tolerance_ratio[bone_index]),
                )
            )
        if not relative_errors:
            return {
                "bone_count": 0,
                "median_relative_error": None,
                "p95_relative_error": None,
                "max_relative_error": None,
                "violating_bone_count": 0,
                "mean_relative_violation": None,
                "max_relative_violation": None,
            }
        error_array = np.asarray(relative_errors, dtype=np.float64)
        violation_array = np.asarray(violations, dtype=np.float64)
        return {
            "bone_count": len(relative_errors),
            "median_relative_error": float(np.median(error_array)),
            "p95_relative_error": float(
                np.percentile(error_array, 95)
            ),
            "max_relative_error": float(np.max(error_array)),
            "violating_bone_count": int(
                np.count_nonzero(violation_array > 0)
            ),
            "mean_relative_violation": float(
                np.mean(violation_array)
            ),
            "max_relative_violation": float(np.max(violation_array)),
        }
