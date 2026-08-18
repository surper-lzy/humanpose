"""Build a lightweight mannequin from a ground-aligned Halpe26 pose.

The output is deliberately renderer-independent.  Tapered capsules describe
limbs; ellipsoids describe the head, trunk, hands, feet, and joint covers.
Input coordinates must use the viewer convention: +X right, +Y forward,
+Z up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rgbd_avatar.pose import HALPE26_NAMES


Color = tuple[float, float, float]

NEUTRAL_COLOR: Color = (0.72, 0.74, 0.76)


@dataclass(frozen=True)
class CapsulePrimitive:
    """A possibly tapered limb between two metric endpoints."""

    name: str
    start_m: np.ndarray
    end_m: np.ndarray
    radius_m: float
    color: Color
    end_radius_m: float | None = None

    @property
    def length_m(self) -> float:
        return float(np.linalg.norm(self.end_m - self.start_m))

    @property
    def resolved_end_radius_m(self) -> float:
        return (
            self.radius_m
            if self.end_radius_m is None
            else self.end_radius_m
        )


@dataclass(frozen=True)
class EllipsoidPrimitive:
    """An oriented ellipsoid with radii along its local XYZ axes."""

    name: str
    center_m: np.ndarray
    rotation: np.ndarray
    radii_m: np.ndarray
    color: Color


@dataclass(frozen=True)
class ProceduralAvatarFrame:
    """Renderer-independent primitives for one pose frame."""

    capsules: tuple[CapsulePrimitive, ...] = ()
    ellipsoids: tuple[EllipsoidPrimitive, ...] = ()

    @property
    def primitive_count(self) -> int:
        return len(self.capsules) + len(self.ellipsoids)


@dataclass(frozen=True)
class ProceduralAvatarConfig:
    """Conservative proportions relative to estimated shoulder width."""

    default_shoulder_width_m: float = 0.42
    min_shoulder_width_m: float = 0.25
    max_shoulder_width_m: float = 0.65
    upper_arm_radius_ratio: float = 0.11
    upper_arm_end_radius_ratio: float = 0.085
    forearm_radius_ratio: float = 0.085
    forearm_end_radius_ratio: float = 0.058
    thigh_radius_ratio: float = 0.16
    thigh_end_radius_ratio: float = 0.115
    shin_radius_ratio: float = 0.115
    shin_end_radius_ratio: float = 0.072
    joint_radius_ratio: float = 0.095
    minimum_segment_length_m: float = 0.025


def _unit(vector: np.ndarray, *, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if np.isfinite(norm) and norm > 1e-8:
        return vector / norm
    return fallback.astype(np.float64, copy=True)


def _body_frame(up_hint: np.ndarray, lateral_hint: np.ndarray) -> np.ndarray:
    """Return a proper local-to-world rotation: lateral, forward, up."""

    up = _unit(up_hint, fallback=np.array([0.0, 0.0, 1.0]))
    lateral_projected = lateral_hint - np.dot(lateral_hint, up) * up
    if np.linalg.norm(lateral_projected) <= 1e-8:
        reference = (
            np.array([1.0, 0.0, 0.0])
            if abs(up[0]) < 0.9
            else np.array([0.0, 1.0, 0.0])
        )
        lateral_projected = reference - np.dot(reference, up) * up
    lateral = _unit(
        lateral_projected,
        fallback=np.array([1.0, 0.0, 0.0]),
    )
    forward = _unit(
        np.cross(up, lateral),
        fallback=np.array([0.0, 1.0, 0.0]),
    )
    lateral = _unit(np.cross(forward, up), fallback=lateral)
    return np.column_stack((lateral, forward, up))


def _foot_frame(forward_hint: np.ndarray) -> np.ndarray:
    up = np.array([0.0, 0.0, 1.0])
    forward_projected = forward_hint.copy()
    forward_projected[2] = 0.0
    forward = _unit(
        forward_projected,
        fallback=np.array([0.0, 1.0, 0.0]),
    )
    lateral = _unit(
        np.cross(forward, up),
        fallback=np.array([1.0, 0.0, 0.0]),
    )
    return np.column_stack((lateral, forward, up))


def _segment_frame(direction: np.ndarray) -> np.ndarray:
    """Return a proper rotation whose local Z follows a body segment."""

    z_axis = _unit(direction, fallback=np.array([0.0, 0.0, 1.0]))
    helper = (
        np.array([0.0, 0.0, 1.0])
        if abs(z_axis[2]) < 0.9
        else np.array([1.0, 0.0, 0.0])
    )
    x_axis = _unit(
        np.cross(helper, z_axis),
        fallback=np.array([1.0, 0.0, 0.0]),
    )
    y_axis = _unit(np.cross(z_axis, x_axis), fallback=np.array([0.0, 1.0, 0.0]))
    return np.column_stack((x_axis, y_axis, z_axis))


def _validate_inputs(
    joints_m: np.ndarray,
    usable: np.ndarray,
    config: ProceduralAvatarConfig,
) -> tuple[np.ndarray, np.ndarray]:
    joints = np.asarray(joints_m, dtype=np.float64)
    mask = np.asarray(usable, dtype=bool)
    expected_count = len(HALPE26_NAMES)
    if joints.shape != (expected_count, 3):
        raise ValueError(
            f"joints_m must have shape ({expected_count}, 3), got "
            f"{joints.shape}."
        )
    if mask.shape != (expected_count,):
        raise ValueError(
            f"usable must have shape ({expected_count},), got {mask.shape}."
        )
    numeric_values = np.array(
        [
            config.default_shoulder_width_m,
            config.min_shoulder_width_m,
            config.max_shoulder_width_m,
            config.upper_arm_radius_ratio,
            config.upper_arm_end_radius_ratio,
            config.forearm_radius_ratio,
            config.forearm_end_radius_ratio,
            config.thigh_radius_ratio,
            config.thigh_end_radius_ratio,
            config.shin_radius_ratio,
            config.shin_end_radius_ratio,
            config.joint_radius_ratio,
            config.minimum_segment_length_m,
        ],
        dtype=np.float64,
    )
    if not np.isfinite(numeric_values).all() or np.any(numeric_values <= 0):
        raise ValueError("Procedural avatar configuration must be positive.")
    if config.min_shoulder_width_m > config.max_shoulder_width_m:
        raise ValueError(
            "min_shoulder_width_m must not exceed max_shoulder_width_m."
        )
    return joints, mask & np.isfinite(joints).all(axis=1)


def _estimate_scale(
    joints: np.ndarray,
    usable: np.ndarray,
    config: ProceduralAvatarConfig,
) -> float:
    candidates: list[float] = []
    if usable[5] and usable[6]:
        candidates.append(float(np.linalg.norm(joints[6] - joints[5])))
    if usable[11] and usable[12]:
        # Adult hip breadth is usually smaller than shoulder breadth.
        candidates.append(1.25 * float(np.linalg.norm(joints[12] - joints[11])))
    plausible = [value for value in candidates if 0.15 <= value <= 0.8]
    estimate = (
        float(np.median(plausible))
        if plausible
        else config.default_shoulder_width_m
    )
    return float(
        np.clip(
            estimate,
            config.min_shoulder_width_m,
            config.max_shoulder_width_m,
        )
    )


def _append_capsule(
    output: list[CapsulePrimitive],
    *,
    name: str,
    start_index: int,
    end_index: int,
    radius_m: float,
    end_radius_m: float,
    color: Color,
    joints: np.ndarray,
    usable: np.ndarray,
    minimum_length_m: float,
) -> None:
    if not (usable[start_index] and usable[end_index]):
        return
    start = joints[start_index].copy()
    end = joints[end_index].copy()
    if np.linalg.norm(end - start) < minimum_length_m:
        return
    output.append(
        CapsulePrimitive(
            name=name,
            start_m=start,
            end_m=end,
            radius_m=radius_m,
            color=color,
            end_radius_m=end_radius_m,
        )
    )


def _append_joint(
    output: list[EllipsoidPrimitive],
    *,
    name: str,
    index: int,
    radius_m: float,
    color: Color,
    joints: np.ndarray,
    usable: np.ndarray,
) -> None:
    if not usable[index]:
        return
    output.append(
        EllipsoidPrimitive(
            name=name,
            center_m=joints[index].copy(),
            rotation=np.eye(3),
            radii_m=np.full(3, radius_m, dtype=np.float64),
            color=color,
        )
    )


def _append_hand(
    output: list[EllipsoidPrimitive],
    *,
    side: str,
    elbow_index: int,
    wrist_index: int,
    scale_m: float,
    joints: np.ndarray,
    usable: np.ndarray,
) -> None:
    if not (usable[elbow_index] and usable[wrist_index]):
        return
    direction = joints[wrist_index] - joints[elbow_index]
    forward = _unit(direction, fallback=np.array([0.0, 0.0, -1.0]))
    half_length = 0.20 * scale_m
    output.append(
        EllipsoidPrimitive(
            name=f"{side}_hand",
            center_m=joints[wrist_index] + 0.17 * scale_m * forward,
            rotation=_segment_frame(forward),
            radii_m=np.array(
                [0.09 * scale_m, 0.055 * scale_m, half_length]
            ),
            color=NEUTRAL_COLOR,
        )
    )


def _append_foot(
    output: list[EllipsoidPrimitive],
    *,
    side: str,
    ankle_index: int,
    toe_indices: tuple[int, int],
    heel_index: int,
    scale_m: float,
    color: Color,
    joints: np.ndarray,
    usable: np.ndarray,
    ground_height_m: float | None,
) -> None:
    if not usable[ankle_index]:
        return
    toes = [joints[index] for index in toe_indices if usable[index]]
    heel = joints[heel_index] if usable[heel_index] else None
    if not toes and heel is None:
        return

    toe_center = np.mean(toes, axis=0) if toes else joints[ankle_index]
    heel_center = heel if heel is not None else joints[ankle_index]
    forward_hint = toe_center - heel_center
    measured_length = float(np.linalg.norm(forward_hint[:2]))
    half_length = np.clip(
        0.58 * measured_length,
        0.22 * scale_m,
        0.42 * scale_m,
    )
    half_width = 0.13 * scale_m
    half_height = 0.075 * scale_m
    center = 0.5 * (toe_center + heel_center)
    if ground_height_m is not None:
        center[2] = max(center[2], ground_height_m + half_height)
    output.append(
        EllipsoidPrimitive(
            name=f"{side}_foot",
            center_m=center,
            rotation=_foot_frame(forward_hint),
            # Local Y follows heel-to-toe, so it carries the long radius.
            radii_m=np.array(
                [half_width, half_length, half_height],
                dtype=np.float64,
            ),
            color=color,
        )
    )


def build_procedural_avatar(
    joints_m: np.ndarray,
    usable: np.ndarray,
    *,
    ground_height_m: float | None = None,
    config: ProceduralAvatarConfig | None = None,
) -> ProceduralAvatarFrame:
    """Create mannequin primitives from display/ground-space Halpe26 joints.

    Missing joints remove only the affected body part.  If a calibrated ground
    height is supplied, feet are prevented from penetrating that plane; their
    measured horizontal placement remains unchanged.
    """

    settings = config or ProceduralAvatarConfig()
    joints, valid = _validate_inputs(joints_m, usable, settings)
    if ground_height_m is not None and not np.isfinite(ground_height_m):
        raise ValueError("ground_height_m must be finite when supplied.")
    if not np.any(valid):
        return ProceduralAvatarFrame()

    scale = _estimate_scale(joints, valid, settings)
    capsules: list[CapsulePrimitive] = []
    ellipsoids: list[EllipsoidPrimitive] = []

    limb_specs = (
        (
            "left_upper_arm",
            5,
            7,
            settings.upper_arm_radius_ratio,
            settings.upper_arm_end_radius_ratio,
        ),
        (
            "left_forearm",
            7,
            9,
            settings.forearm_radius_ratio,
            settings.forearm_end_radius_ratio,
        ),
        (
            "right_upper_arm",
            6,
            8,
            settings.upper_arm_radius_ratio,
            settings.upper_arm_end_radius_ratio,
        ),
        (
            "right_forearm",
            8,
            10,
            settings.forearm_radius_ratio,
            settings.forearm_end_radius_ratio,
        ),
        (
            "left_thigh",
            11,
            13,
            settings.thigh_radius_ratio,
            settings.thigh_end_radius_ratio,
        ),
        (
            "left_shin",
            13,
            15,
            settings.shin_radius_ratio,
            settings.shin_end_radius_ratio,
        ),
        (
            "right_thigh",
            12,
            14,
            settings.thigh_radius_ratio,
            settings.thigh_end_radius_ratio,
        ),
        (
            "right_shin",
            14,
            16,
            settings.shin_radius_ratio,
            settings.shin_end_radius_ratio,
        ),
    )
    for name, start, end, start_radius_ratio, end_radius_ratio in limb_specs:
        _append_capsule(
            capsules,
            name=name,
            start_index=start,
            end_index=end,
            radius_m=start_radius_ratio * scale,
            end_radius_m=end_radius_ratio * scale,
            color=NEUTRAL_COLOR,
            joints=joints,
            usable=valid,
            minimum_length_m=settings.minimum_segment_length_m,
        )

    if valid[17] and valid[18]:
        neck_end = 0.72 * joints[18] + 0.28 * joints[17]
        if np.linalg.norm(neck_end - joints[18]) >= (
            settings.minimum_segment_length_m
        ):
            capsules.append(
                CapsulePrimitive(
                    name="neck",
                    start_m=joints[18].copy(),
                    end_m=neck_end,
                    radius_m=0.105 * scale,
                    end_radius_m=0.095 * scale,
                    color=NEUTRAL_COLOR,
                )
            )

    shoulder_valid = valid[5] and valid[6]
    hip_valid = valid[11] and valid[12]
    if shoulder_valid:
        shoulder_center = 0.5 * (joints[5] + joints[6])
        lateral_hint = joints[6] - joints[5]
    else:
        shoulder_center = joints[18] if valid[18] else None
        lateral_hint = np.array([1.0, 0.0, 0.0])
    if hip_valid:
        hip_center = 0.5 * (joints[11] + joints[12])
        if not shoulder_valid:
            lateral_hint = joints[12] - joints[11]
    else:
        hip_center = joints[19] if valid[19] else None

    body_rotation = np.eye(3)
    torso_length = 1.25 * scale
    if shoulder_center is not None and hip_center is not None:
        up_hint = shoulder_center - hip_center
        measured_torso_length = float(np.linalg.norm(up_hint))
        if measured_torso_length >= settings.minimum_segment_length_m:
            torso_length = measured_torso_length
            body_rotation = _body_frame(up_hint, lateral_hint)
            ellipsoids.append(
                EllipsoidPrimitive(
                    name="torso",
                    center_m=(
                        0.58 * shoulder_center + 0.42 * hip_center
                    ),
                    rotation=body_rotation,
                    radii_m=np.array(
                        [0.56 * scale, 0.235 * scale, 0.43 * torso_length]
                    ),
                    color=NEUTRAL_COLOR,
                )
            )
            ellipsoids.append(
                EllipsoidPrimitive(
                    name="abdomen",
                    center_m=(
                        0.25 * shoulder_center + 0.75 * hip_center
                    ),
                    rotation=body_rotation,
                    radii_m=np.array(
                        [0.39 * scale, 0.20 * scale, 0.29 * torso_length]
                    ),
                    color=NEUTRAL_COLOR,
                )
            )

    if hip_valid:
        pelvis_width = float(np.linalg.norm(joints[12] - joints[11]))
        ellipsoids.append(
            EllipsoidPrimitive(
                name="pelvis",
                center_m=0.5 * (joints[11] + joints[12]),
                rotation=body_rotation,
                radii_m=np.array(
                    [
                        max(0.58 * pelvis_width, 0.28 * scale),
                        0.25 * scale,
                        0.18 * torso_length,
                    ]
                ),
                color=NEUTRAL_COLOR,
            )
        )

    if valid[17] and valid[18]:
        neck_to_head = joints[17] - joints[18]
        head_length = float(np.linalg.norm(neck_to_head))
        if head_length >= settings.minimum_segment_length_m:
            ellipsoids.append(
                EllipsoidPrimitive(
                    name="head",
                    center_m=0.52 * joints[17] + 0.48 * joints[18],
                    rotation=_body_frame(neck_to_head, lateral_hint),
                    radii_m=np.array(
                        [0.23 * scale, 0.21 * scale, 0.52 * head_length]
                    ),
                    color=NEUTRAL_COLOR,
                )
            )

    joint_specs = (
        ("left_shoulder", 5, 0.90),
        ("left_elbow", 7, 0.75),
        ("right_shoulder", 6, 0.90),
        ("right_elbow", 8, 0.75),
        ("left_hip", 11, 0.95),
        ("left_knee", 13, 0.82),
        ("left_ankle", 15, 0.60),
        ("right_hip", 12, 0.95),
        ("right_knee", 14, 0.82),
        ("right_ankle", 16, 0.60),
    )
    for name, index, radius_multiplier in joint_specs:
        _append_joint(
            ellipsoids,
            name=name,
            index=index,
            radius_m=(
                settings.joint_radius_ratio * scale * radius_multiplier
            ),
            color=NEUTRAL_COLOR,
            joints=joints,
            usable=valid,
        )

    _append_hand(
        ellipsoids,
        side="left",
        elbow_index=7,
        wrist_index=9,
        scale_m=scale,
        joints=joints,
        usable=valid,
    )
    _append_hand(
        ellipsoids,
        side="right",
        elbow_index=8,
        wrist_index=10,
        scale_m=scale,
        joints=joints,
        usable=valid,
    )

    _append_foot(
        ellipsoids,
        side="left",
        ankle_index=15,
        toe_indices=(20, 22),
        heel_index=24,
        scale_m=scale,
        color=NEUTRAL_COLOR,
        joints=joints,
        usable=valid,
        ground_height_m=ground_height_m,
    )
    _append_foot(
        ellipsoids,
        side="right",
        ankle_index=16,
        toe_indices=(21, 23),
        heel_index=25,
        scale_m=scale,
        color=NEUTRAL_COLOR,
        joints=joints,
        usable=valid,
        ground_height_m=ground_height_m,
    )

    return ProceduralAvatarFrame(
        capsules=tuple(capsules),
        ellipsoids=tuple(ellipsoids),
    )
