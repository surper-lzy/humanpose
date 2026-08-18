"""Dependency-free, occlusion-aware RGB-D identity tracking.

The tracker remains usable by the isolated comparison tool and is also an
opt-in identity backend for the integrated multi-person live pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from rgbd_avatar.pose import Pose2D


_UNMATCHABLE_COST = 1_000_000.0


@dataclass(frozen=True)
class ShadowIdentityConfig:
    """Association and lifecycle parameters for the shadow experiment."""

    min_confirmed_hits: int = 1
    normal_missing_s: float = 0.35
    occluded_missing_s: float = 1.0
    overlap_iou: float = 0.25
    ambiguity_margin: float = 0.08
    max_match_cost: float = 1.15
    max_center_distance_ratio: float = 1.50
    max_root_distance_m: float = 0.90
    max_appearance_distance: float = 0.65
    velocity_alpha: float = 0.55
    appearance_alpha: float = 0.15

    def __post_init__(self) -> None:
        if self.min_confirmed_hits <= 0:
            raise ValueError("min_confirmed_hits must be positive.")
        positive = (
            self.normal_missing_s,
            self.occluded_missing_s,
            self.max_match_cost,
            self.max_center_distance_ratio,
            self.max_root_distance_m,
        )
        if any(not math.isfinite(value) or value <= 0 for value in positive):
            raise ValueError("Shadow identity thresholds must be positive.")
        if self.occluded_missing_s < self.normal_missing_s:
            raise ValueError(
                "occluded_missing_s must be at least normal_missing_s."
            )
        unit_values = (
            self.overlap_iou,
            self.ambiguity_margin,
            self.max_appearance_distance,
            self.velocity_alpha,
            self.appearance_alpha,
        )
        if any(not 0 <= value <= 1 for value in unit_values):
            raise ValueError("Shadow identity ratios must be in [0, 1].")


@dataclass(frozen=True)
class ShadowIdentityObservation:
    """One observed pose passed to the isolated identity tracker."""

    observation_id: int
    pose2d: Pose2D
    root_camera_m: np.ndarray | None
    appearance: np.ndarray | None

    def __post_init__(self) -> None:
        root = self.root_camera_m
        if root is not None:
            root = np.asarray(root, dtype=np.float64)
            if root.shape != (3,) or not np.isfinite(root).all():
                raise ValueError("root_camera_m must contain three finite values.")
        appearance = self.appearance
        if appearance is not None:
            appearance = _normalise_descriptor(appearance)
            if appearance.size == 0:
                raise ValueError("appearance must not be empty.")
        object.__setattr__(self, "root_camera_m", root)
        object.__setattr__(self, "appearance", appearance)


@dataclass(frozen=True)
class ShadowIdentityAssignment:
    """Shadow identity selected for one current observation."""

    observation_id: int
    shadow_id: int
    state: str
    match_cost: float | None
    appearance_frozen: bool


@dataclass(frozen=True)
class ShadowIdentityFrame:
    """Identity-only result produced without changing the live pose result."""

    assignments: tuple[ShadowIdentityAssignment, ...]
    predicted_shadow_ids: tuple[int, ...]
    removed_shadow_ids: tuple[int, ...]
    overlap_observation_ids: tuple[int, ...]
    ambiguous_observation_ids: tuple[int, ...]
    timing_ms: float


@dataclass
class _ShadowTrack:
    shadow_id: int
    pose2d: Pose2D
    root_camera_m: np.ndarray | None
    appearance: np.ndarray | None
    last_timestamp_s: float
    last_observed_s: float
    center_velocity_px_s: np.ndarray
    root_velocity_m_s: np.ndarray
    hits: int = 1
    occlusion_grace_until_s: float = 0.0
    missing_since_s: float | None = None


def _normalise_descriptor(values: np.ndarray) -> np.ndarray:
    descriptor = np.asarray(values, dtype=np.float32).reshape(-1)
    if not np.isfinite(descriptor).all():
        raise ValueError("Appearance descriptors must be finite.")
    norm = float(np.linalg.norm(descriptor))
    if norm <= 1e-12:
        return np.zeros_like(descriptor)
    return descriptor / norm


def _bbox_iou(first: np.ndarray, second: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = np.asarray(first, dtype=np.float64)
    bx1, by1, bx2, by2 = np.asarray(second, dtype=np.float64)
    width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = width * height
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _bbox_center_and_diagonal(bbox: np.ndarray) -> tuple[np.ndarray, float]:
    x1, y1, x2, y2 = np.asarray(bbox, dtype=np.float64)
    size = np.maximum(np.array([x2 - x1, y2 - y1]), 1.0)
    center = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5])
    return center, float(np.linalg.norm(size))


def _translated_pose(pose: Pose2D, offset_xy: np.ndarray) -> Pose2D:
    offset = np.asarray(offset_xy, dtype=np.float32)
    return Pose2D(
        keypoints=pose.keypoints + offset[None, :],
        scores=pose.scores.copy(),
        bbox_xyxy=pose.bbox_xyxy
        + np.array([offset[0], offset[1], offset[0], offset[1]]),
        bbox_score=pose.bbox_score,
    )


def _keypoint_distance_ratio(first: Pose2D, second: Pose2D) -> float | None:
    common = (
        (first.scores >= 0.3)
        & (second.scores >= 0.3)
        & np.isfinite(first.keypoints).all(axis=1)
        & np.isfinite(second.keypoints).all(axis=1)
    )
    if np.count_nonzero(common) < 4:
        return None
    _, first_diagonal = _bbox_center_and_diagonal(first.bbox_xyxy)
    _, second_diagonal = _bbox_center_and_diagonal(second.bbox_xyxy)
    scale = max(1.0, 0.5 * (first_diagonal + second_diagonal))
    distance = np.linalg.norm(
        first.keypoints[common] - second.keypoints[common],
        axis=1,
    )
    return float(np.median(distance) / scale)


def _appearance_distance(
    first: np.ndarray | None,
    second: np.ndarray | None,
) -> float | None:
    if first is None or second is None or first.shape != second.shape:
        return None
    if not np.any(first) or not np.any(second):
        return None
    return float(np.clip(1.0 - np.dot(first, second), 0.0, 2.0))


def extract_upper_body_hsv_descriptor(
    rgb_bgr: np.ndarray,
    pose2d: Pose2D,
) -> np.ndarray | None:
    """Extract a cheap appearance descriptor from a conservative torso crop."""

    image = np.asarray(rgb_bgr, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("rgb_bgr must have shape HxWx3.")
    height, width = image.shape[:2]
    x1, y1, x2, y2 = np.asarray(pose2d.bbox_xyxy, dtype=np.float64)
    bbox_width = max(1.0, x2 - x1)
    bbox_height = max(1.0, y2 - y1)

    torso_ids = np.asarray((5, 6, 11, 12), dtype=np.int64)
    torso_valid = (
        pose2d.scores[torso_ids] >= 0.3
    ) & np.isfinite(pose2d.keypoints[torso_ids]).all(axis=1)
    torso_points = pose2d.keypoints[torso_ids[torso_valid]]
    if len(torso_points) >= 2:
        crop_x1 = float(np.min(torso_points[:, 0]) - 0.12 * bbox_width)
        crop_x2 = float(np.max(torso_points[:, 0]) + 0.12 * bbox_width)
        crop_y1 = float(np.min(torso_points[:, 1]) - 0.08 * bbox_height)
        crop_y2 = float(np.max(torso_points[:, 1]) + 0.10 * bbox_height)
    else:
        crop_x1 = x1 + 0.20 * bbox_width
        crop_x2 = x2 - 0.20 * bbox_width
        crop_y1 = y1 + 0.15 * bbox_height
        crop_y2 = y1 + 0.65 * bbox_height

    left = max(0, int(math.floor(crop_x1)))
    right = min(width, int(math.ceil(crop_x2)))
    top = max(0, int(math.floor(crop_y1)))
    bottom = min(height, int(math.ceil(crop_y2)))
    if right - left < 4 or bottom - top < 4:
        return None

    hsv = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [16, 8],
        [0, 180, 0, 256],
    ).reshape(-1)
    total = float(np.sum(histogram))
    if total <= 0:
        return None
    # Square-rooting an L1 histogram gives a Hellinger-style embedding whose
    # cosine distance is stable and cheap to compute.
    return _normalise_descriptor(np.sqrt(histogram / total))


class ShadowRGBDIdentityTracker:
    """Track identities using motion, depth, appearance and overlap gates."""

    def __init__(self, config: ShadowIdentityConfig | None = None) -> None:
        self.config = config or ShadowIdentityConfig()
        self._tracks: dict[int, _ShadowTrack] = {}
        self._next_id = 1

    @property
    def active_shadow_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._tracks))

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    def _predict(
        self,
        track: _ShadowTrack,
        timestamp_s: float,
    ) -> tuple[Pose2D, np.ndarray | None]:
        dt = max(0.0, min(timestamp_s - track.last_timestamp_s, 0.5))
        predicted_pose = _translated_pose(
            track.pose2d,
            track.center_velocity_px_s * dt,
        )
        predicted_root = (
            track.root_camera_m + track.root_velocity_m_s * dt
            if track.root_camera_m is not None
            else None
        )
        return predicted_pose, predicted_root

    def _cost(
        self,
        track: _ShadowTrack,
        observation: ShadowIdentityObservation,
        timestamp_s: float,
    ) -> float:
        predicted_pose, predicted_root = self._predict(track, timestamp_s)
        predicted_center, predicted_diagonal = _bbox_center_and_diagonal(
            predicted_pose.bbox_xyxy
        )
        observed_center, observed_diagonal = _bbox_center_and_diagonal(
            observation.pose2d.bbox_xyxy
        )
        scale = max(1.0, 0.5 * (predicted_diagonal + observed_diagonal))
        center_ratio = float(
            np.linalg.norm(predicted_center - observed_center) / scale
        )
        iou = _bbox_iou(
            predicted_pose.bbox_xyxy,
            observation.pose2d.bbox_xyxy,
        )
        root_distance = (
            float(np.linalg.norm(predicted_root - observation.root_camera_m))
            if predicted_root is not None
            and observation.root_camera_m is not None
            else None
        )
        appearance_distance = _appearance_distance(
            track.appearance,
            observation.appearance,
        )

        if center_ratio > self.config.max_center_distance_ratio and iou == 0:
            return _UNMATCHABLE_COST
        if (
            root_distance is not None
            and root_distance > self.config.max_root_distance_m
        ):
            return _UNMATCHABLE_COST
        if (
            appearance_distance is not None
            and appearance_distance > self.config.max_appearance_distance
            and iou < self.config.overlap_iou
        ):
            return _UNMATCHABLE_COST

        terms: list[tuple[float, float]] = [
            (0.15, 1.0 - iou),
            (
                0.25,
                min(
                    2.0,
                    center_ratio / self.config.max_center_distance_ratio,
                ),
            ),
        ]
        keypoint_distance = _keypoint_distance_ratio(
            predicted_pose,
            observation.pose2d,
        )
        if keypoint_distance is not None:
            terms.append((0.15, min(2.0, keypoint_distance / 0.60)))
        if root_distance is not None:
            terms.append(
                (
                    0.20,
                    min(2.0, root_distance / self.config.max_root_distance_m),
                )
            )
        if appearance_distance is not None:
            terms.append(
                (
                    0.25,
                    min(
                        2.0,
                        appearance_distance
                        / self.config.max_appearance_distance,
                    ),
                )
            )
        weight = sum(item_weight for item_weight, _ in terms)
        return float(
            sum(item_weight * value for item_weight, value in terms) / weight
        )

    def _overlap_ids(
        self,
        observations: list[ShadowIdentityObservation],
    ) -> set[int]:
        overlapping: set[int] = set()
        for first_index, first in enumerate(observations):
            for second in observations[first_index + 1 :]:
                if (
                    _bbox_iou(
                        first.pose2d.bbox_xyxy,
                        second.pose2d.bbox_xyxy,
                    )
                    >= self.config.overlap_iou
                ):
                    overlapping.add(first.observation_id)
                    overlapping.add(second.observation_id)
        return overlapping

    @staticmethod
    def _assignment_margin(
        costs: np.ndarray,
        row: int,
        column: int,
    ) -> float:
        selected = float(costs[row, column])
        alternatives = [
            float(value)
            for index, value in enumerate(costs[row])
            if index != column and value < _UNMATCHABLE_COST
        ]
        alternatives.extend(
            float(costs[index, column])
            for index in range(costs.shape[0])
            if index != row and costs[index, column] < _UNMATCHABLE_COST
        )
        if not alternatives:
            return math.inf
        return min(alternatives) - selected

    def _new_track(
        self,
        observation: ShadowIdentityObservation,
        timestamp_s: float,
        overlapping: bool,
    ) -> _ShadowTrack:
        track = _ShadowTrack(
            shadow_id=self._next_id,
            pose2d=observation.pose2d,
            root_camera_m=(
                observation.root_camera_m.copy()
                if observation.root_camera_m is not None
                else None
            ),
            appearance=(
                observation.appearance.copy()
                if observation.appearance is not None
                else None
            ),
            last_timestamp_s=timestamp_s,
            last_observed_s=timestamp_s,
            center_velocity_px_s=np.zeros(2, dtype=np.float64),
            root_velocity_m_s=np.zeros(3, dtype=np.float64),
            occlusion_grace_until_s=(
                timestamp_s + self.config.occluded_missing_s
                if overlapping
                else 0.0
            ),
        )
        self._tracks[track.shadow_id] = track
        self._next_id += 1
        return track

    def _update_track(
        self,
        track: _ShadowTrack,
        observation: ShadowIdentityObservation,
        timestamp_s: float,
        *,
        freeze_appearance: bool,
    ) -> None:
        dt = max(timestamp_s - track.last_observed_s, 1e-3)
        previous_center, _ = _bbox_center_and_diagonal(track.pose2d.bbox_xyxy)
        current_center, _ = _bbox_center_and_diagonal(
            observation.pose2d.bbox_xyxy
        )
        measured_velocity = (current_center - previous_center) / dt
        alpha = self.config.velocity_alpha
        track.center_velocity_px_s = (
            (1.0 - alpha) * track.center_velocity_px_s
            + alpha * measured_velocity
        )
        if (
            track.root_camera_m is not None
            and observation.root_camera_m is not None
        ):
            measured_root_velocity = (
                observation.root_camera_m - track.root_camera_m
            ) / dt
            track.root_velocity_m_s = (
                (1.0 - alpha) * track.root_velocity_m_s
                + alpha * measured_root_velocity
            )
        elif observation.root_camera_m is not None:
            track.root_velocity_m_s = np.zeros(3, dtype=np.float64)

        if not freeze_appearance and observation.appearance is not None:
            if track.appearance is None:
                track.appearance = observation.appearance.copy()
            else:
                appearance_alpha = self.config.appearance_alpha
                track.appearance = _normalise_descriptor(
                    (1.0 - appearance_alpha) * track.appearance
                    + appearance_alpha * observation.appearance
                )
        track.pose2d = observation.pose2d
        if observation.root_camera_m is not None:
            track.root_camera_m = observation.root_camera_m.copy()
        track.last_timestamp_s = timestamp_s
        track.last_observed_s = timestamp_s
        track.missing_since_s = None
        track.hits += 1
        if freeze_appearance:
            track.occlusion_grace_until_s = (
                timestamp_s + self.config.occluded_missing_s
            )

    def update(
        self,
        observations: list[ShadowIdentityObservation],
        timestamp_s: float,
    ) -> ShadowIdentityFrame:
        """Update identities while preserving ambiguous occluded tracks."""

        started = time.perf_counter()
        if not math.isfinite(timestamp_s) or timestamp_s < 0:
            raise ValueError("timestamp_s must be finite and non-negative.")
        observation_ids = [item.observation_id for item in observations]
        if len(observation_ids) != len(set(observation_ids)):
            raise ValueError("observation_id must be unique within one frame.")
        overlap_ids = self._overlap_ids(observations)
        track_ids = sorted(self._tracks)
        had_tracks = bool(track_ids)
        assignments: list[ShadowIdentityAssignment] = []
        matched_tracks: set[int] = set()
        matched_observations: set[int] = set()
        ambiguous_observations: set[int] = set()

        if track_ids and observations:
            costs = np.full(
                (len(track_ids), len(observations)),
                _UNMATCHABLE_COST,
                dtype=np.float64,
            )
            for row, shadow_id in enumerate(track_ids):
                for column, observation in enumerate(observations):
                    costs[row, column] = self._cost(
                        self._tracks[shadow_id],
                        observation,
                        timestamp_s,
                    )
            rows, columns = linear_sum_assignment(costs)
            for row_value, column_value in zip(rows, columns, strict=True):
                row = int(row_value)
                column = int(column_value)
                cost = float(costs[row, column])
                if cost >= _UNMATCHABLE_COST or cost > self.config.max_match_cost:
                    continue
                observation = observations[column]
                margin = self._assignment_margin(costs, row, column)
                ambiguous = (
                    observation.observation_id in overlap_ids
                    and margin < self.config.ambiguity_margin
                )
                if ambiguous:
                    ambiguous_observations.add(observation.observation_id)
                    continue
                shadow_id = track_ids[row]
                track = self._tracks[shadow_id]
                freeze_appearance = observation.observation_id in overlap_ids
                self._update_track(
                    track,
                    observation,
                    timestamp_s,
                    freeze_appearance=freeze_appearance,
                )
                matched_tracks.add(shadow_id)
                matched_observations.add(column)
                state = (
                    "confirmed"
                    if track.hits >= self.config.min_confirmed_hits
                    else "tentative"
                )
                assignments.append(
                    ShadowIdentityAssignment(
                        observation_id=observation.observation_id,
                        shadow_id=shadow_id,
                        state=state,
                        match_cost=cost,
                        appearance_frozen=freeze_appearance,
                    )
                )

        for column, observation in enumerate(observations):
            if column in matched_observations:
                continue
            if observation.observation_id in ambiguous_observations:
                continue
            overlapping = observation.observation_id in overlap_ids
            # Do not create a new identity in the middle of an ambiguous
            # overlap when existing tracks are already being preserved.
            if overlapping and had_tracks:
                ambiguous_observations.add(observation.observation_id)
                continue
            track = self._new_track(observation, timestamp_s, overlapping)
            matched_tracks.add(track.shadow_id)
            state = (
                "confirmed"
                if track.hits >= self.config.min_confirmed_hits
                else "tentative"
            )
            assignments.append(
                ShadowIdentityAssignment(
                    observation_id=observation.observation_id,
                    shadow_id=track.shadow_id,
                    state=state,
                    match_cost=None,
                    appearance_frozen=overlapping,
                )
            )

        predicted_ids: list[int] = []
        removed_ids: list[int] = []
        for shadow_id in sorted(set(self._tracks) - matched_tracks):
            track = self._tracks[shadow_id]
            if track.missing_since_s is None:
                track.missing_since_s = timestamp_s
            allowed_missing_s = (
                self.config.occluded_missing_s
                if timestamp_s <= track.occlusion_grace_until_s
                else self.config.normal_missing_s
            )
            if timestamp_s - track.missing_since_s > allowed_missing_s:
                removed_ids.append(shadow_id)
            else:
                predicted_ids.append(shadow_id)
        for shadow_id in removed_ids:
            del self._tracks[shadow_id]

        assignments.sort(key=lambda item: item.observation_id)
        return ShadowIdentityFrame(
            assignments=tuple(assignments),
            predicted_shadow_ids=tuple(predicted_ids),
            removed_shadow_ids=tuple(removed_ids),
            overlap_observation_ids=tuple(sorted(overlap_ids)),
            ambiguous_observation_ids=tuple(
                sorted(ambiguous_observations)
            ),
            timing_ms=(time.perf_counter() - started) * 1000.0,
        )
