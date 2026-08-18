"""One Euro filtering and short-gap prediction for metric 3D poses."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from rgbd_avatar.pose import HALPE26_NAMES, Pose3D


def _smoothing_factor(cutoff_hz: float, dt_s: float) -> float:
    if not math.isfinite(cutoff_hz) or cutoff_hz <= 0:
        raise ValueError("cutoff_hz must be finite and positive.")
    if not math.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("dt_s must be finite and positive.")
    value = 2.0 * math.pi * cutoff_hz * dt_s
    return value / (1.0 + value)


class OneEuroFilter3D:
    """One Euro filter whose XYZ axes share a speed-dependent cutoff."""

    def __init__(
        self,
        min_cutoff_hz: float = 0.5,
        beta: float = 2.0,
        derivative_cutoff_hz: float = 1.0,
    ) -> None:
        if min_cutoff_hz <= 0:
            raise ValueError("min_cutoff_hz must be positive.")
        if beta < 0:
            raise ValueError("beta must be non-negative.")
        if derivative_cutoff_hz <= 0:
            raise ValueError("derivative_cutoff_hz must be positive.")
        self.min_cutoff_hz = float(min_cutoff_hz)
        self.beta = float(beta)
        self.derivative_cutoff_hz = float(derivative_cutoff_hz)
        self.reset()

    def reset(self) -> None:
        self._timestamp_s: float | None = None
        self._raw = np.zeros(3, dtype=np.float64)
        self._filtered = np.zeros(3, dtype=np.float64)
        self._velocity = np.zeros(3, dtype=np.float64)

    @property
    def initialized(self) -> bool:
        return self._timestamp_s is not None

    @property
    def timestamp_s(self) -> float | None:
        return self._timestamp_s

    @property
    def value(self) -> np.ndarray:
        if not self.initialized:
            raise RuntimeError("OneEuroFilter3D has not received a sample.")
        return self._filtered.copy()

    @property
    def velocity_mps(self) -> np.ndarray:
        if not self.initialized:
            raise RuntimeError("OneEuroFilter3D has not received a sample.")
        return self._velocity.copy()

    def preview_filtered_velocity(
        self,
        timestamp_s: float,
        value: np.ndarray,
    ) -> np.ndarray:
        """Return the next derivative estimate without mutating the filter."""

        timestamp = float(timestamp_s)
        sample = np.asarray(value, dtype=np.float64)
        if not math.isfinite(timestamp):
            raise ValueError("timestamp_s must be finite.")
        if sample.shape != (3,) or not np.isfinite(sample).all():
            raise ValueError("OneEuroFilter3D value must be a finite XYZ vector.")
        if self._timestamp_s is None:
            return np.zeros(3, dtype=np.float64)

        dt_s = timestamp - self._timestamp_s
        if dt_s <= 0:
            raise ValueError("One Euro timestamps must be strictly increasing.")
        raw_velocity = (sample - self._raw) / dt_s
        derivative_alpha = _smoothing_factor(
            self.derivative_cutoff_hz, dt_s
        )
        return (
            derivative_alpha * raw_velocity
            + (1.0 - derivative_alpha) * self._velocity
        )

    def update(
        self,
        timestamp_s: float,
        value: np.ndarray,
        *,
        cutoff_speed_mps: float | None = None,
    ) -> np.ndarray:
        timestamp = float(timestamp_s)
        sample = np.asarray(value, dtype=np.float64)
        if not math.isfinite(timestamp):
            raise ValueError("timestamp_s must be finite.")
        if sample.shape != (3,) or not np.isfinite(sample).all():
            raise ValueError("OneEuroFilter3D value must be a finite XYZ vector.")
        if cutoff_speed_mps is not None and (
            not math.isfinite(cutoff_speed_mps) or cutoff_speed_mps < 0
        ):
            raise ValueError("cutoff_speed_mps must be finite and non-negative.")

        if self._timestamp_s is None:
            self._timestamp_s = timestamp
            self._raw = sample.copy()
            self._filtered = sample.copy()
            self._velocity.fill(0.0)
            return self._filtered.copy()

        dt_s = timestamp - self._timestamp_s
        if dt_s <= 0:
            raise ValueError("One Euro timestamps must be strictly increasing.")

        filtered_velocity = self.preview_filtered_velocity(timestamp, sample)
        speed_mps = (
            float(np.linalg.norm(filtered_velocity))
            if cutoff_speed_mps is None
            else float(cutoff_speed_mps)
        )
        cutoff_hz = self.min_cutoff_hz + self.beta * speed_mps
        position_alpha = _smoothing_factor(cutoff_hz, dt_s)
        filtered = (
            position_alpha * sample
            + (1.0 - position_alpha) * self._filtered
        )

        self._timestamp_s = timestamp
        self._raw = sample.copy()
        self._filtered = filtered
        self._velocity = filtered_velocity
        return filtered.copy()


@dataclass
class TemporalPose3D:
    """Filtered 3D pose with explicit observed/predicted/missing states."""

    joints_m: np.ndarray
    confidence: np.ndarray
    usable: np.ndarray
    observed: np.ndarray
    predicted: np.ndarray
    age_s: np.ndarray
    reset_occurred: bool = False

    def __post_init__(self) -> None:
        count = len(HALPE26_NAMES)
        self.joints_m = np.asarray(self.joints_m, dtype=np.float32)
        self.confidence = np.asarray(self.confidence, dtype=np.float32)
        self.usable = np.asarray(self.usable, dtype=bool)
        self.observed = np.asarray(self.observed, dtype=bool)
        self.predicted = np.asarray(self.predicted, dtype=bool)
        self.age_s = np.asarray(self.age_s, dtype=np.float32)
        if self.joints_m.shape != (count, 3):
            raise ValueError(
                f"Expected temporal joints shape {(count, 3)}, "
                f"got {self.joints_m.shape}."
            )
        for name, values in (
            ("confidence", self.confidence),
            ("usable", self.usable),
            ("observed", self.observed),
            ("predicted", self.predicted),
            ("age_s", self.age_s),
        ):
            if values.shape != (count,):
                raise ValueError(
                    f"Expected {name} shape {(count,)}, got {values.shape}."
                )
        if np.any(self.observed & self.predicted):
            raise ValueError("A joint cannot be observed and predicted together.")
        if not np.array_equal(self.usable, self.observed | self.predicted):
            raise ValueError("usable must equal observed OR predicted.")

    def to_dict(self) -> dict[str, Any]:
        joints = []
        for index, name in enumerate(HALPE26_NAMES):
            joints.append(
                {
                    "id": index,
                    "name": name,
                    "xyz_m": (
                        self.joints_m[index].tolist()
                        if self.usable[index]
                        else None
                    ),
                    "confidence": float(self.confidence[index]),
                    "usable": bool(self.usable[index]),
                    "observed": bool(self.observed[index]),
                    "predicted": bool(self.predicted[index]),
                    "age_s": (
                        float(self.age_s[index])
                        if self.usable[index]
                        else None
                    ),
                }
            )
        return {
            "keypoint_format": "halpe26",
            "coordinate_system": {
                "handedness": "right",
                "x": "right",
                "y": "down",
                "z": "forward",
                "unit": "meter",
            },
            "usable_joint_count": int(np.count_nonzero(self.usable)),
            "observed_joint_count": int(np.count_nonzero(self.observed)),
            "predicted_joint_count": int(np.count_nonzero(self.predicted)),
            "reset_occurred": bool(self.reset_occurred),
            "joints": joints,
        }


class Pose3DTemporalFilter:
    """Apply per-joint 3D filters, optionally with one shared pose cutoff."""

    def __init__(
        self,
        min_cutoff_hz: float = 0.5,
        beta: float = 2.0,
        derivative_cutoff_hz: float = 1.0,
        reset_gap_s: float = 2.0,
        max_prediction_s: float = 1.1,
        min_observation_confidence: float = 0.0,
        shared_cutoff: bool = False,
        shared_speed_percentile: float = 75.0,
    ) -> None:
        if reset_gap_s <= 0:
            raise ValueError("reset_gap_s must be positive.")
        if max_prediction_s < 0:
            raise ValueError("max_prediction_s must be non-negative.")
        if min_observation_confidence < 0:
            raise ValueError(
                "min_observation_confidence must be non-negative."
            )
        if not 0 < shared_speed_percentile <= 100:
            raise ValueError("shared_speed_percentile must be in (0, 100].")
        self.reset_gap_s = float(reset_gap_s)
        self.max_prediction_s = float(max_prediction_s)
        self.min_observation_confidence = float(
            min_observation_confidence
        )
        self.shared_cutoff = bool(shared_cutoff)
        self.shared_speed_percentile = float(shared_speed_percentile)
        self._filters = [
            OneEuroFilter3D(
                min_cutoff_hz=min_cutoff_hz,
                beta=beta,
                derivative_cutoff_hz=derivative_cutoff_hz,
            )
            for _ in HALPE26_NAMES
        ]
        count = len(HALPE26_NAMES)
        self._last_observed_s = np.full(count, np.nan, dtype=np.float64)
        self._last_confidence = np.zeros(count, dtype=np.float64)
        self._last_frame_timestamp_s: float | None = None
        self.discontinuity_reset_count = 0

    def reset(self) -> None:
        for joint_filter in self._filters:
            joint_filter.reset()
        self._last_observed_s.fill(np.nan)
        self._last_confidence.fill(0.0)
        self._last_frame_timestamp_s = None

    def terminate_track(self, timestamp_s: float) -> TemporalPose3D:
        """Clear all predictions when the tracked person leaves the frame."""

        timestamp = float(timestamp_s)
        if not math.isfinite(timestamp):
            raise ValueError("timestamp_s must be finite.")
        if (
            self._last_frame_timestamp_s is not None
            and timestamp <= self._last_frame_timestamp_s
        ):
            raise ValueError(
                "Temporal pose timestamps must be strictly increasing."
            )
        self.reset()
        self._last_frame_timestamp_s = timestamp
        count = len(HALPE26_NAMES)
        return TemporalPose3D(
            joints_m=np.full((count, 3), np.nan, dtype=np.float64),
            confidence=np.zeros(count, dtype=np.float64),
            usable=np.zeros(count, dtype=bool),
            observed=np.zeros(count, dtype=bool),
            predicted=np.zeros(count, dtype=bool),
            age_s=np.full(count, np.inf, dtype=np.float64),
            reset_occurred=True,
        )

    def update(
        self,
        timestamp_s: float,
        pose: Pose3D | None,
    ) -> TemporalPose3D:
        timestamp = float(timestamp_s)
        if not math.isfinite(timestamp):
            raise ValueError("timestamp_s must be finite.")

        reset_occurred = False
        if self._last_frame_timestamp_s is not None:
            dt_s = timestamp - self._last_frame_timestamp_s
            if dt_s <= 0:
                raise ValueError(
                    "Temporal pose timestamps must be strictly increasing."
                )
            if dt_s > self.reset_gap_s:
                self.reset()
                self.discontinuity_reset_count += 1
                reset_occurred = True
        self._last_frame_timestamp_s = timestamp

        count = len(HALPE26_NAMES)
        joints = np.full((count, 3), np.nan, dtype=np.float64)
        confidence = np.zeros(count, dtype=np.float64)
        usable = np.zeros(count, dtype=bool)
        observed = np.zeros(count, dtype=bool)
        predicted = np.zeros(count, dtype=bool)
        age_s = np.full(count, np.inf, dtype=np.float64)

        if pose is None:
            observation_valid = np.zeros(count, dtype=bool)
        else:
            observation_valid = (
                pose.valid
                & np.isfinite(pose.joints_m).all(axis=1)
                & (pose.confidence >= self.min_observation_confidence)
            )

        shared_cutoff_speed_mps: float | None = None
        if self.shared_cutoff:
            preview_speeds = [
                float(
                    np.linalg.norm(
                        self._filters[index].preview_filtered_velocity(
                            timestamp,
                            pose.joints_m[index],
                        )
                    )
                )
                for index in np.flatnonzero(observation_valid)
                if pose is not None and self._filters[index].initialized
            ]
            shared_cutoff_speed_mps = (
                float(
                    np.percentile(
                        preview_speeds,
                        self.shared_speed_percentile,
                    )
                )
                if preview_speeds
                else 0.0
            )

        for index, joint_filter in enumerate(self._filters):
            if observation_valid[index]:
                assert pose is not None
                joints[index] = joint_filter.update(
                    timestamp,
                    pose.joints_m[index],
                    cutoff_speed_mps=shared_cutoff_speed_mps,
                )
                confidence[index] = float(pose.confidence[index])
                usable[index] = True
                observed[index] = True
                age_s[index] = 0.0
                self._last_observed_s[index] = timestamp
                self._last_confidence[index] = confidence[index]
                continue

            if not joint_filter.initialized:
                continue

            missing_age_s = timestamp - self._last_observed_s[index]
            if (
                self.max_prediction_s > 0
                and missing_age_s < self.max_prediction_s
            ):
                remaining = 1.0 - missing_age_s / self.max_prediction_s
                joints[index] = (
                    joint_filter.value
                    + joint_filter.velocity_mps * missing_age_s
                )
                confidence[index] = (
                    self._last_confidence[index] * max(0.0, remaining)
                )
                usable[index] = True
                predicted[index] = True
                age_s[index] = missing_age_s
            else:
                joint_filter.reset()
                self._last_observed_s[index] = np.nan
                self._last_confidence[index] = 0.0

        return TemporalPose3D(
            joints_m=joints,
            confidence=confidence,
            usable=usable,
            observed=observed,
            predicted=predicted,
            age_s=age_s,
            reset_occurred=reset_occurred,
        )
