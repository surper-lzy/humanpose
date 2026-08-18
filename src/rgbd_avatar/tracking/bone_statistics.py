"""Robust per-bone length statistics for Halpe26 metric poses."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from rgbd_avatar.pose import HALPE26_LINKS, HALPE26_NAMES


class BoneLengthAccumulator:
    """Collect observed bone lengths without applying a bone constraint."""

    def __init__(
        self,
        links: Sequence[tuple[int, int]] = HALPE26_LINKS,
        min_joint_confidence: float = 0.0,
    ) -> None:
        if min_joint_confidence < 0:
            raise ValueError("min_joint_confidence must be non-negative.")
        self.links = tuple(links)
        self.min_joint_confidence = float(min_joint_confidence)
        self._samples: list[list[float]] = [[] for _ in self.links]

    def update(
        self,
        joints_m: np.ndarray,
        valid: np.ndarray,
        confidence: np.ndarray | None = None,
    ) -> None:
        joints = np.asarray(joints_m, dtype=np.float64)
        usable = np.asarray(valid, dtype=bool)
        count = len(HALPE26_NAMES)
        if joints.shape != (count, 3):
            raise ValueError(
                f"Expected joints shape {(count, 3)}, got {joints.shape}."
            )
        if usable.shape != (count,):
            raise ValueError(
                f"Expected valid shape {(count,)}, got {usable.shape}."
            )
        if confidence is None:
            scores = np.ones(count, dtype=np.float64)
        else:
            scores = np.asarray(confidence, dtype=np.float64)
            if scores.shape != (count,):
                raise ValueError(
                    f"Expected confidence shape {(count,)}, got {scores.shape}."
                )

        for sample_list, (start, end) in zip(
            self._samples, self.links, strict=True
        ):
            if (
                not usable[start]
                or not usable[end]
                or scores[start] < self.min_joint_confidence
                or scores[end] < self.min_joint_confidence
                or not np.isfinite(joints[[start, end]]).all()
            ):
                continue
            length_m = float(np.linalg.norm(joints[end] - joints[start]))
            if math_is_positive_finite(length_m):
                sample_list.append(length_m)

    def summary(self) -> dict[str, Any]:
        bones: list[dict[str, Any]] = []
        all_samples: list[float] = []
        relative_mads: list[float] = []
        for (start, end), values in zip(
            self.links, self._samples, strict=True
        ):
            samples = np.asarray(values, dtype=np.float64)
            record: dict[str, Any] = {
                "start_id": start,
                "start_name": HALPE26_NAMES[start],
                "end_id": end,
                "end_name": HALPE26_NAMES[end],
                "sample_count": int(samples.size),
            }
            if samples.size:
                median = float(np.median(samples))
                mad = float(np.median(np.abs(samples - median)))
                record.update(
                    {
                        "median_m": median,
                        "mad_m": mad,
                        "relative_mad": (
                            mad / median if median > 0 else None
                        ),
                        "mean_m": float(np.mean(samples)),
                        "std_m": float(np.std(samples)),
                        "p05_m": float(np.percentile(samples, 5)),
                        "p95_m": float(np.percentile(samples, 95)),
                    }
                )
                if median > 0:
                    relative_mads.append(mad / median)
                all_samples.extend(values)
            else:
                record.update(
                    {
                        "median_m": None,
                        "mad_m": None,
                        "relative_mad": None,
                        "mean_m": None,
                        "std_m": None,
                        "p05_m": None,
                        "p95_m": None,
                    }
                )
            bones.append(record)
        return {
            "bone_count": len(self.links),
            "bones_with_samples": len(relative_mads),
            "total_sample_count": len(all_samples),
            "median_relative_mad": (
                float(np.median(relative_mads))
                if relative_mads
                else None
            ),
            "mean_relative_mad": (
                float(np.mean(relative_mads))
                if relative_mads
                else None
            ),
            "bones": bones,
        }


def math_is_positive_finite(value: float) -> bool:
    """Keep the hot update loop readable without accepting zero-length bones."""

    return bool(np.isfinite(value) and value > 0)
