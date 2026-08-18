"""Build a deliberately stylized stick-figure avatar from Halpe26 joints."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rgbd_avatar.pose import HALPE26_NAMES

from .procedural import (
    CapsulePrimitive,
    EllipsoidPrimitive,
    ProceduralAvatarFrame,
)


Color = tuple[float, float, float]


@dataclass(frozen=True)
class StickFigureConfig:
    """Metric proportions and friendly non-photorealistic colors."""

    default_shoulder_width_m: float = 0.42
    minimum_shoulder_width_m: float = 0.25
    maximum_shoulder_width_m: float = 0.65
    rod_radius_ratio: float = 0.055
    torso_radius_ratio: float = 0.065
    joint_radius_ratio: float = 0.058
    head_radius_ratio: float = 0.26
    minimum_segment_length_m: float = 0.02
    rod_color: Color = (0.015, 0.015, 0.015)
    torso_color: Color = (0.015, 0.015, 0.015)
    joint_color: Color = (0.015, 0.015, 0.015)
    head_color: Color = (0.005, 0.005, 0.005)


def _validate(
    joints_m: np.ndarray,
    usable: np.ndarray,
    config: StickFigureConfig,
) -> tuple[np.ndarray, np.ndarray]:
    joints = np.asarray(joints_m, dtype=np.float64)
    mask = np.asarray(usable, dtype=bool)
    count = len(HALPE26_NAMES)
    if joints.shape != (count, 3):
        raise ValueError(f"joints_m must have shape ({count}, 3).")
    if mask.shape != (count,):
        raise ValueError(f"usable must have shape ({count},).")
    numeric = np.asarray(
        (
            config.default_shoulder_width_m,
            config.minimum_shoulder_width_m,
            config.maximum_shoulder_width_m,
            config.rod_radius_ratio,
            config.torso_radius_ratio,
            config.joint_radius_ratio,
            config.head_radius_ratio,
            config.minimum_segment_length_m,
        ),
        dtype=np.float64,
    )
    if not np.isfinite(numeric).all() or np.any(numeric <= 0):
        raise ValueError("Stick-figure configuration must be positive.")
    if config.minimum_shoulder_width_m > config.maximum_shoulder_width_m:
        raise ValueError(
            "minimum_shoulder_width_m must not exceed "
            "maximum_shoulder_width_m."
        )
    return joints, mask & np.isfinite(joints).all(axis=1)


def _scale(
    joints: np.ndarray,
    usable: np.ndarray,
    config: StickFigureConfig,
) -> float:
    estimates: list[float] = []
    if usable[5] and usable[6]:
        estimates.append(float(np.linalg.norm(joints[6] - joints[5])))
    if usable[11] and usable[12]:
        estimates.append(1.25 * float(np.linalg.norm(joints[12] - joints[11])))
    estimates = [value for value in estimates if 0.15 <= value <= 0.8]
    value = float(np.median(estimates)) if estimates else (
        config.default_shoulder_width_m
    )
    return float(
        np.clip(
            value,
            config.minimum_shoulder_width_m,
            config.maximum_shoulder_width_m,
        )
    )


def _capsule(
    output: list[CapsulePrimitive],
    name: str,
    start: np.ndarray,
    end: np.ndarray,
    radius_m: float,
    color: Color,
    minimum_length_m: float,
) -> None:
    if not (np.isfinite(start).all() and np.isfinite(end).all()):
        return
    if np.linalg.norm(end - start) < minimum_length_m:
        return
    output.append(
        CapsulePrimitive(
            name=name,
            start_m=start.copy(),
            end_m=end.copy(),
            radius_m=radius_m,
            color=color,
        )
    )


def _sphere(
    output: list[EllipsoidPrimitive],
    name: str,
    center: np.ndarray,
    radius_m: float,
    color: Color,
) -> None:
    if not np.isfinite(center).all():
        return
    output.append(
        EllipsoidPrimitive(
            name=name,
            center_m=center.copy(),
            rotation=np.eye(3),
            radii_m=np.full(3, radius_m, dtype=np.float64),
            color=color,
        )
    )


def build_stick_figure_avatar(
    joints_m: np.ndarray,
    usable: np.ndarray,
    *,
    ground_height_m: float | None = None,
    config: StickFigureConfig | None = None,
) -> ProceduralAvatarFrame:
    """Create rods and joint balls that follow a metric Halpe26 pose."""

    settings = config or StickFigureConfig()
    joints, valid = _validate(joints_m, usable, settings)
    if ground_height_m is not None and not np.isfinite(ground_height_m):
        raise ValueError("ground_height_m must be finite when supplied.")
    if not np.any(valid):
        return ProceduralAvatarFrame()

    scale = _scale(joints, valid, settings)
    rod_radius = settings.rod_radius_ratio * scale
    torso_radius = settings.torso_radius_ratio * scale
    joint_radius = settings.joint_radius_ratio * scale
    capsules: list[CapsulePrimitive] = []
    ellipsoids: list[EllipsoidPrimitive] = []

    body_segments = (
        ("left_upper_arm", 5, 7),
        ("left_forearm", 7, 9),
        ("right_upper_arm", 6, 8),
        ("right_forearm", 8, 10),
        ("left_thigh", 11, 13),
        ("left_shin", 13, 15),
        ("right_thigh", 12, 14),
        ("right_shin", 14, 16),
    )
    for name, start_index, end_index in body_segments:
        if valid[start_index] and valid[end_index]:
            _capsule(
                capsules,
                name,
                joints[start_index],
                joints[end_index],
                rod_radius,
                settings.rod_color,
                settings.minimum_segment_length_m,
            )

    shoulder_center = None
    if valid[5] and valid[6]:
        shoulder_center = 0.5 * (joints[5] + joints[6])
        _capsule(
            capsules,
            "shoulder_bar",
            joints[5],
            joints[6],
            torso_radius,
            settings.torso_color,
            settings.minimum_segment_length_m,
        )
    elif valid[18]:
        shoulder_center = joints[18]

    hip_center = None
    if valid[11] and valid[12]:
        hip_center = 0.5 * (joints[11] + joints[12])
        _capsule(
            capsules,
            "hip_bar",
            joints[11],
            joints[12],
            torso_radius,
            settings.torso_color,
            settings.minimum_segment_length_m,
        )
    elif valid[19]:
        hip_center = joints[19]

    if shoulder_center is not None and hip_center is not None:
        _capsule(
            capsules,
            "torso_axis",
            hip_center,
            shoulder_center,
            torso_radius,
            settings.torso_color,
            settings.minimum_segment_length_m,
        )
    if shoulder_center is not None and valid[18]:
        _capsule(
            capsules,
            "neck",
            shoulder_center,
            joints[18],
            rod_radius,
            settings.torso_color,
            settings.minimum_segment_length_m,
        )

    for side, ankle_index, toe_indices, heel_index in (
        ("left", 15, (20, 22), 24),
        ("right", 16, (21, 23), 25),
    ):
        if not valid[ankle_index]:
            continue
        foot_points = [
            joints[index]
            for index in (*toe_indices, heel_index)
            if valid[index]
        ]
        if not foot_points:
            continue
        foot_center = np.mean(foot_points, axis=0)
        if ground_height_m is not None:
            foot_center[2] = max(
                foot_center[2],
                ground_height_m + rod_radius,
            )
        _capsule(
            capsules,
            f"{side}_foot",
            joints[ankle_index],
            foot_center,
            rod_radius,
            settings.rod_color,
            settings.minimum_segment_length_m,
        )

    for index in (5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16):
        if valid[index]:
            _sphere(
                ellipsoids,
                HALPE26_NAMES[index],
                joints[index],
                joint_radius,
                settings.joint_color,
            )
    if hip_center is not None:
        _sphere(
            ellipsoids,
            "hip_center",
            hip_center,
            joint_radius,
            settings.joint_color,
        )
    if shoulder_center is not None:
        _sphere(
            ellipsoids,
            "shoulder_center",
            shoulder_center,
            joint_radius,
            settings.joint_color,
        )

    if valid[17] and valid[18]:
        head_vector = joints[17] - joints[18]
        measured = float(np.linalg.norm(head_vector))
        head_radius = float(
            np.clip(
                0.42 * measured,
                0.18 * scale,
                settings.head_radius_ratio * scale,
            )
        )
        direction = head_vector / max(measured, 1e-8)
        center = joints[17] - 0.80 * head_radius * direction
        _sphere(
            ellipsoids,
            "head",
            center,
            head_radius,
            settings.head_color,
        )
    elif valid[0]:
        _sphere(
            ellipsoids,
            "head",
            joints[0],
            settings.head_radius_ratio * scale,
            settings.head_color,
        )

    return ProceduralAvatarFrame(
        capsules=tuple(capsules),
        ellipsoids=tuple(ellipsoids),
    )
