"""Pure topology decisions for ambiguous RGB-D joint surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class TopologyCandidate:
    """A local surface candidate detached from point-cloud implementation."""

    token: int
    xyz_m: np.ndarray
    score: float
    person_quality: float | None = None

    def __post_init__(self) -> None:
        point = np.asarray(self.xyz_m, dtype=np.float64)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("Topology candidate XYZ must be finite shape (3,).")
        if not np.isfinite(self.score):
            raise ValueError("Topology candidate score must be finite.")
        if self.person_quality is not None and not np.isfinite(
            self.person_quality
        ):
            raise ValueError("person_quality must be finite or None.")
        object.__setattr__(self, "xyz_m", point)
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(
            self,
            "person_quality",
            (
                float(self.person_quality)
                if self.person_quality is not None
                else None
            ),
        )


@dataclass(frozen=True)
class TopologySelection:
    """Selected candidate token per joint plus search diagnostics."""

    selected_tokens: dict[int, int | None]
    objective: float
    evaluated_combination_count: int
    feasible_combination_count: int


def _distance(
    first: TopologyCandidate,
    second: TopologyCandidate,
) -> float:
    return float(np.linalg.norm(first.xyz_m - second.xyz_m))


def select_face_core_candidates(
    candidates: Mapping[int, Sequence[TopologyCandidate]],
    *,
    joint_ids: tuple[int, int, int] = (0, 1, 2),
    candidate_limit: int = 5,
    min_present: int = 2,
    missing_score: float = 0.25,
    depth_tolerance_m: float = 0.10,
    depth_tolerance_ratio: float = 0.04,
    nose_eye_max_length_m: float = 0.15,
    eye_eye_max_length_m: float = 0.18,
    neck_depth_m: float | None = None,
    neck_far_tolerance_m: float = 0.15,
) -> TopologySelection | None:
    """Jointly choose nose and eye surfaces with metric face consistency."""

    if candidate_limit <= 0:
        raise ValueError("candidate_limit must be positive.")
    if not 1 <= min_present <= len(joint_ids):
        raise ValueError("min_present is incompatible with joint_ids.")
    if missing_score < 0:
        raise ValueError("missing_score must be non-negative.")
    if (
        depth_tolerance_m <= 0
        or depth_tolerance_ratio < 0
        or nose_eye_max_length_m <= 0
        or eye_eye_max_length_m <= 0
        or neck_far_tolerance_m <= 0
    ):
        raise ValueError("Face topology thresholds are invalid.")

    options: list[list[TopologyCandidate | None]] = []
    for joint_id in joint_ids:
        ranked = sorted(
            candidates.get(joint_id, ()),
            key=lambda candidate: candidate.score,
            reverse=True,
        )[:candidate_limit]
        options.append([*ranked, None])

    evaluated = 0
    feasible = 0
    best_key: tuple[float, int] | None = None
    best_selection: dict[int, int | None] | None = None
    pair_limits = {
        frozenset((joint_ids[0], joint_ids[1])): nose_eye_max_length_m,
        frozenset((joint_ids[0], joint_ids[2])): nose_eye_max_length_m,
        frozenset((joint_ids[1], joint_ids[2])): eye_eye_max_length_m,
    }

    for combination in product(*options):
        evaluated += 1
        present = [
            (joint_id, candidate)
            for joint_id, candidate in zip(
                joint_ids,
                combination,
                strict=True,
            )
            if candidate is not None
        ]
        if len(present) < min_present:
            continue
        depths = np.asarray(
            [candidate.xyz_m[2] for _, candidate in present],
            dtype=np.float64,
        )
        reference_depth = float(np.median(depths))
        depth_tolerance = max(
            depth_tolerance_m,
            depth_tolerance_ratio * reference_depth,
        )
        depth_spread = float(np.max(depths) - np.min(depths))
        if depth_spread > depth_tolerance:
            continue
        if neck_depth_m is not None and any(
            candidate.xyz_m[2] > neck_depth_m + neck_far_tolerance_m
            for _, candidate in present
        ):
            continue

        pair_valid = True
        for first_index in range(len(present)):
            for second_index in range(first_index + 1, len(present)):
                first_id, first = present[first_index]
                second_id, second = present[second_index]
                limit = pair_limits[frozenset((first_id, second_id))]
                if _distance(first, second) > limit:
                    pair_valid = False
                    break
            if not pair_valid:
                break
        if not pair_valid:
            continue

        feasible += 1
        present_count = len(present)
        objective = sum(candidate.score for _, candidate in present)
        objective += missing_score * (len(joint_ids) - present_count)
        objective -= 0.15 * depth_spread / max(depth_tolerance, 1e-12)
        key = (objective, present_count)
        if best_key is None or key > best_key:
            best_key = key
            best_selection = {
                joint_id: (
                    candidate.token if candidate is not None else None
                )
                for joint_id, candidate in zip(
                    joint_ids,
                    combination,
                    strict=True,
                )
            }

    if best_key is None or best_selection is None:
        return None
    return TopologySelection(
        selected_tokens=best_selection,
        objective=best_key[0],
        evaluated_combination_count=evaluated,
        feasible_combination_count=feasible,
    )


def select_foot_group_candidates(
    candidates: Mapping[int, Sequence[TopologyCandidate]],
    *,
    ankle_id: int,
    big_toe_id: int,
    small_toe_id: int,
    heel_id: int,
    knee_xyz_m: np.ndarray | None,
    candidate_limit: int = 6,
    ankle_min_score_ratio: float = 0.75,
    min_person_quality: float = 0.25,
    missing_score: float = 0.25,
    compactness_weight: float = 0.15,
    knee_ankle_max_length_m: float = 0.75,
    ankle_toe_max_length_m: float = 0.35,
    ankle_heel_max_length_m: float = 0.22,
    toe_pair_max_length_m: float = 0.20,
) -> TopologySelection | None:
    """Choose one coherent ankle/toe/heel surface group.

    Low person-quality surfaces are treated as likely floor/background. A
    missing leaf remains a legal choice so a ground point is never required
    merely to complete the skeleton.
    """

    if candidate_limit <= 0:
        raise ValueError("candidate_limit must be positive.")
    if not 0 < ankle_min_score_ratio <= 1:
        raise ValueError("ankle_min_score_ratio must be in (0, 1].")
    if not 0 <= min_person_quality <= 1:
        raise ValueError("min_person_quality must be in [0, 1].")
    if missing_score < 0 or compactness_weight < 0:
        raise ValueError(
            "missing_score and compactness_weight must be non-negative."
        )
    if any(
        value <= 0
        for value in (
            knee_ankle_max_length_m,
            ankle_toe_max_length_m,
            ankle_heel_max_length_m,
            toe_pair_max_length_m,
        )
    ):
        raise ValueError("Foot topology metric thresholds must be positive.")

    def person_surface(candidate: TopologyCandidate) -> bool:
        return (
            candidate.person_quality is None
            or candidate.person_quality >= min_person_quality
        )

    ranked_ankles = sorted(
        candidates.get(ankle_id, ()),
        key=lambda candidate: candidate.score,
        reverse=True,
    )
    if not ranked_ankles:
        return None
    best_ankle_score = ranked_ankles[0].score
    ankle_options = [
        candidate
        for candidate in ranked_ankles[:candidate_limit]
        if candidate.score
        >= best_ankle_score * ankle_min_score_ratio
        and person_surface(candidate)
    ]
    if not ankle_options:
        return None

    leaf_ids = (big_toe_id, small_toe_id, heel_id)
    leaf_options: list[list[TopologyCandidate | None]] = []
    for joint_id in leaf_ids:
        ranked = sorted(
            (
                candidate
                for candidate in candidates.get(joint_id, ())
                if person_surface(candidate)
            ),
            key=lambda candidate: candidate.score,
            reverse=True,
        )[:candidate_limit]
        leaf_options.append([*ranked, None])

    knee = (
        np.asarray(knee_xyz_m, dtype=np.float64)
        if knee_xyz_m is not None
        else None
    )
    if knee is not None and (
        knee.shape != (3,) or not np.isfinite(knee).all()
    ):
        raise ValueError("knee_xyz_m must be finite shape (3,) or None.")

    evaluated = 0
    feasible = 0
    best_key: tuple[float, int] | None = None
    best_selection: dict[int, int | None] | None = None
    for ankle in ankle_options:
        if (
            knee is not None
            and np.linalg.norm(ankle.xyz_m - knee)
            > knee_ankle_max_length_m
        ):
            continue
        for leaves in product(*leaf_options):
            evaluated += 1
            big_toe, small_toe, heel = leaves
            if (
                big_toe is not None
                and _distance(ankle, big_toe)
                > ankle_toe_max_length_m
            ):
                continue
            if (
                small_toe is not None
                and _distance(ankle, small_toe)
                > ankle_toe_max_length_m
            ):
                continue
            if (
                heel is not None
                and _distance(ankle, heel)
                > ankle_heel_max_length_m
            ):
                continue
            if (
                big_toe is not None
                and small_toe is not None
                and _distance(big_toe, small_toe)
                > toe_pair_max_length_m
            ):
                continue

            feasible += 1
            present_leaves = [
                candidate
                for candidate in leaves
                if candidate is not None
            ]
            objective = ankle.score
            objective += sum(
                candidate.score for candidate in present_leaves
            )
            objective += missing_score * (
                len(leaf_ids) - len(present_leaves)
            )
            normalized_lengths = []
            if big_toe is not None:
                normalized_lengths.append(
                    _distance(ankle, big_toe)
                    / ankle_toe_max_length_m
                )
            if small_toe is not None:
                normalized_lengths.append(
                    _distance(ankle, small_toe)
                    / ankle_toe_max_length_m
                )
            if heel is not None:
                normalized_lengths.append(
                    _distance(ankle, heel)
                    / ankle_heel_max_length_m
                )
            if big_toe is not None and small_toe is not None:
                normalized_lengths.append(
                    _distance(big_toe, small_toe)
                    / toe_pair_max_length_m
                )
            objective -= compactness_weight * sum(normalized_lengths)
            key = (objective, len(present_leaves))
            if best_key is None or key > best_key:
                best_key = key
                best_selection = {
                    ankle_id: ankle.token,
                    **{
                        joint_id: (
                            candidate.token
                            if candidate is not None
                            else None
                        )
                        for joint_id, candidate in zip(
                            leaf_ids,
                            leaves,
                            strict=True,
                        )
                    },
                }

    if best_key is None or best_selection is None:
        return None
    return TopologySelection(
        selected_tokens=best_selection,
        objective=best_key[0],
        evaluated_combination_count=evaluated,
        feasible_combination_count=feasible,
    )


def bilateral_length_outliers(
    *,
    center_xyz_m: np.ndarray,
    left_xyz_m: np.ndarray | None,
    right_xyz_m: np.ndarray | None,
    max_length_m: float,
    asymmetry_ratio: float,
) -> tuple[bool, bool, dict[str, float | None]]:
    """Detect one implausibly long side while preserving symmetric poses."""

    if max_length_m <= 0:
        raise ValueError("max_length_m must be positive.")
    if asymmetry_ratio <= 1:
        raise ValueError("asymmetry_ratio must exceed 1.")
    center = np.asarray(center_xyz_m, dtype=np.float64)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("center_xyz_m must be finite shape (3,).")

    def length(point: np.ndarray | None) -> float | None:
        if point is None:
            return None
        value = np.asarray(point, dtype=np.float64)
        if value.shape != (3,) or not np.isfinite(value).all():
            return None
        return float(np.linalg.norm(value - center))

    left_length = length(left_xyz_m)
    right_length = length(right_xyz_m)
    left_outlier = bool(
        left_length is not None
        and left_length > max_length_m
        and (
            right_length is None
            or left_length > asymmetry_ratio * right_length
        )
    )
    right_outlier = bool(
        right_length is not None
        and right_length > max_length_m
        and (
            left_length is None
            or right_length > asymmetry_ratio * left_length
        )
    )
    return left_outlier, right_outlier, {
        "left_length_m": left_length,
        "right_length_m": right_length,
    }
