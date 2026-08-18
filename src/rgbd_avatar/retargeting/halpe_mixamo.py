"""Direct fixed-proportion Halpe26/Hand21 to Mixamo analytical IK."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from rgbd_avatar.avatar.mixamo_asset import MixamoAsset
from rgbd_avatar.retargeting.halpe_smpl import HalpeSMPLRetargetProfile


_REQUIRED_BONES = (
    "Hips", "Spine", "Spine1", "Spine2", "Neck", "Head",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
    "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
)

_SCALE_SEGMENTS = (
    ("left_thigh", 11, 13, "LeftUpLeg", "LeftLeg"),
    ("left_shin", 13, 15, "LeftLeg", "LeftFoot"),
    ("right_thigh", 12, 14, "RightUpLeg", "RightLeg"),
    ("right_shin", 14, 16, "RightLeg", "RightFoot"),
    ("left_upper_arm", 5, 7, "LeftArm", "LeftForeArm"),
    ("left_forearm", 7, 9, "LeftForeArm", "LeftHand"),
    ("right_upper_arm", 6, 8, "RightArm", "RightForeArm"),
    ("right_forearm", 8, 10, "RightForeArm", "RightHand"),
)


def _unit(value: np.ndarray) -> np.ndarray | None:
    vector = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if vector.shape != (3,) or not np.isfinite(vector).all() or norm <= 1e-8:
        return None
    return vector / norm


def _frame(primary_value: np.ndarray, secondary_value: np.ndarray) -> np.ndarray | None:
    primary = _unit(primary_value)
    secondary = _unit(secondary_value)
    if primary is None or secondary is None:
        return None
    secondary = _unit(secondary - np.dot(secondary, primary) * primary)
    if secondary is None:
        return None
    tertiary = _unit(np.cross(primary, secondary))
    if tertiary is None:
        return None
    secondary = np.cross(tertiary, primary)
    result = np.column_stack((primary, secondary, tertiary))
    return result if np.linalg.det(result) > 0.0 else None


def _body_basis(up_value: np.ndarray, lateral_value: np.ndarray) -> np.ndarray | None:
    up = _unit(up_value)
    lateral = _unit(lateral_value)
    if up is None or lateral is None:
        return None
    lateral = _unit(lateral - np.dot(lateral, up) * up)
    if lateral is None:
        return None
    depth = _unit(np.cross(up, lateral))
    if depth is None:
        return None
    lateral = np.cross(depth, up)
    result = np.column_stack((lateral, depth, up))
    return result if np.linalg.det(result) > 0.0 else None


def _limited_rotation(
    previous: np.ndarray,
    target: np.ndarray,
    maximum_step_rad: float,
    response: float,
) -> np.ndarray:
    delta = Rotation.from_matrix(previous.T @ target).as_rotvec()
    angle = float(np.linalg.norm(delta))
    if angle > maximum_step_rad:
        delta *= maximum_step_rad / angle
    delta *= response
    return previous @ Rotation.from_rotvec(delta).as_matrix()


@dataclass(frozen=True)
class MixamoHandObservation:
    joints_display_m: np.ndarray
    confidence: np.ndarray
    valid: np.ndarray

    def __post_init__(self) -> None:
        joints = np.asarray(self.joints_display_m, dtype=np.float64)
        confidence = np.asarray(self.confidence, dtype=np.float64)
        valid = np.asarray(self.valid, dtype=bool)
        if joints.shape != (21, 3) or confidence.shape != (21,) or valid.shape != (21,):
            raise ValueError("Mixamo Hand21 observation arrays have invalid shapes.")
        if np.any(valid) and not np.isfinite(joints[valid]).all():
            raise ValueError("Valid Mixamo hand joints must be finite.")
        object.__setattr__(self, "joints_display_m", joints)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "valid", valid)


@dataclass(frozen=True)
class MixamoIKConfig:
    maximum_rotation_speed_deg_s: float = 180.0
    rotation_response: float = 0.78
    minimum_segment_confidence: float = 0.35
    minimum_hand_confidence: float = 0.12

    def __post_init__(self) -> None:
        if not math.isfinite(self.maximum_rotation_speed_deg_s) or self.maximum_rotation_speed_deg_s <= 0:
            raise ValueError("Mixamo maximum rotation speed must be positive.")
        if not 0.0 < self.rotation_response <= 1.0:
            raise ValueError("Mixamo rotation response must be in (0, 1].")
        if not 0.0 <= self.minimum_segment_confidence <= 1.0:
            raise ValueError("Mixamo segment confidence must be in [0, 1].")
        if not 0.0 <= self.minimum_hand_confidence <= 1.0:
            raise ValueError("Mixamo hand confidence must be in [0, 1].")


@dataclass(frozen=True)
class MixamoIKFrame:
    bone_global_m: np.ndarray
    root_display_m: np.ndarray
    rejected_segments: tuple[str, ...]
    held_bones: tuple[str, ...]

    def __post_init__(self) -> None:
        matrices = np.asarray(self.bone_global_m, dtype=np.float64)
        root = np.asarray(self.root_display_m, dtype=np.float64)
        if matrices.ndim != 3 or matrices.shape[1:] != (4, 4):
            raise ValueError("Mixamo IK matrices must have shape Bx4x4.")
        if root.shape != (3,) or not np.isfinite(matrices).all() or not np.isfinite(root).all():
            raise ValueError("Mixamo IK output must be finite.")
        object.__setattr__(self, "bone_global_m", matrices)
        object.__setattr__(self, "root_display_m", root)


def estimate_mixamo_scale(
    asset: MixamoAsset,
    joints_sequence: Sequence[np.ndarray],
    confidence_sequence: Sequence[np.ndarray],
    usable_sequence: Sequence[np.ndarray],
    predicted_sequence: Sequence[np.ndarray],
    profile: HalpeSMPLRetargetProfile,
) -> float:
    """Estimate one robust person-to-avatar scale from reliable limb lengths."""

    if not (
        len(joints_sequence) == len(confidence_sequence)
        == len(usable_sequence) == len(predicted_sequence)
    ):
        raise ValueError("Mixamo scale sequences must have equal length.")
    index = asset.bone_index
    bind_positions = asset.bind_global_m[:, :3, 3]
    ratios: list[float] = []
    stature_ratios: list[float] = []
    bind_foot_center = 0.5 * (
        bind_positions[index["LeftToeBase"]]
        + bind_positions[index["RightToeBase"]]
    )
    bind_stature = float(
        np.linalg.norm(
            bind_positions[index["HeadTop_End"]] - bind_foot_center
        )
    )
    for joints, confidence, usable, predicted in zip(
        joints_sequence,
        confidence_sequence,
        usable_sequence,
        predicted_sequence,
        strict=True,
    ):
        points = np.asarray(joints, dtype=np.float64)
        scores = np.asarray(confidence, dtype=np.float64)
        valid = np.asarray(usable, dtype=bool)
        predicted_mask = np.asarray(predicted, dtype=bool)
        if (
            valid[17]
            and valid[15]
            and valid[16]
            and not (predicted_mask[17] or predicted_mask[15] or predicted_mask[16])
            and min(scores[17], scores[15], scores[16]) >= 0.35
        ):
            foot_center = 0.5 * (points[15] + points[16])
            observed_stature = float(np.linalg.norm(points[17] - foot_center))
            if 1.20 <= observed_stature <= 2.20:
                stature_ratios.append(observed_stature / bind_stature)
        for name, start, end, bone, child in _SCALE_SEGMENTS:
            if profile.classify(name, points, scores, valid, predicted_mask) != "inlier":
                continue
            if predicted_mask[start] or predicted_mask[end]:
                continue
            observed = float(np.linalg.norm(points[end] - points[start]))
            rest = float(np.linalg.norm(bind_positions[index[child]] - bind_positions[index[bone]]))
            if rest > 1e-6:
                ratios.append(observed / rest)
    if len(stature_ratios) >= 3:
        # This character has intentionally stylized limb proportions.  A
        # height-derived uniform scale preserves the authored silhouette;
        # limb ratios are still gathered above to validate the observations.
        scale = float(np.median(np.asarray(stature_ratios, dtype=np.float64)))
        if not 0.65 <= scale <= 1.50:
            raise ValueError(f"Implausible Mixamo metric scale {scale:.4f}.")
        return scale
    if len(ratios) < 8:
        raise ValueError("Too few reliable limb observations to scale Mixamo.")
    values = np.asarray(ratios, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad > 1e-8:
        values = values[np.abs(values - median) <= 3.5 * 1.4826 * mad]
    scale = float(np.median(values))
    if not 0.65 <= scale <= 1.50:
        raise ValueError(f"Implausible Mixamo metric scale {scale:.4f}.")
    return scale


class MixamoAnalyticalIK:
    """Stateful analytical IK with bind-pose proportions and temporal holds."""

    def __init__(
        self,
        asset: MixamoAsset,
        profile: HalpeSMPLRetargetProfile,
        *,
        scale: float,
        config: MixamoIKConfig = MixamoIKConfig(),
    ) -> None:
        self.asset = asset
        self.profile = profile
        self.scale = float(scale)
        self.config = config
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("Mixamo scale must be finite and positive.")
        missing = set(_REQUIRED_BONES) - set(asset.bone_names)
        if missing:
            raise ValueError(f"Mixamo skeleton is missing bones: {sorted(missing)}")
        self.index = asset.bone_index
        self.bind_positions = asset.bind_global_m[:, :3, 3]
        self.bind_rotations = asset.bind_global_m[:, :3, :3]
        self.local_offsets = np.zeros((len(asset.bone_names), 3), dtype=np.float64)
        self.local_rotations = np.zeros((len(asset.bone_names), 3, 3), dtype=np.float64)
        self.local_rotations[0] = self.bind_rotations[0]
        for bone in range(1, len(asset.bone_names)):
            parent = int(asset.parent_indices[bone])
            parent_rotation = self.bind_rotations[parent]
            self.local_offsets[bone] = parent_rotation.T @ (
                self.bind_positions[bone] - self.bind_positions[parent]
            )
            self.local_rotations[bone] = parent_rotation.T @ self.bind_rotations[bone]
        self.bind_body_basis = _body_basis(
            self.bind_positions[self.index["Neck"]] - self.bind_positions[self.index["Hips"]],
            self.bind_positions[self.index["LeftArm"]] - self.bind_positions[self.index["RightArm"]],
        )
        if self.bind_body_basis is None:
            raise ValueError("Mixamo bind pose has a degenerate torso frame.")
        self.bind_frames = self._build_bind_frames()
        self.previous_rotations: np.ndarray | None = None

    def _build_bind_frames(self) -> dict[str, np.ndarray]:
        p = self.bind_positions
        i = self.index
        body_depth = self.bind_body_basis[:, 1]
        body_up = self.bind_body_basis[:, 2]
        specs = {
            "LeftShoulder": ("LeftArm", body_up),
            "RightShoulder": ("RightArm", body_up),
            "LeftArm": ("LeftForeArm", body_depth),
            "RightArm": ("RightForeArm", body_depth),
            "LeftForeArm": ("LeftHand", body_depth),
            "RightForeArm": ("RightHand", body_depth),
            "LeftUpLeg": ("LeftLeg", body_depth),
            "RightUpLeg": ("RightLeg", body_depth),
            "LeftLeg": ("LeftFoot", body_depth),
            "RightLeg": ("RightFoot", body_depth),
            "Neck": ("Head", body_depth),
        }
        frames: dict[str, np.ndarray] = {}
        for bone, (child, secondary) in specs.items():
            frame = _frame(p[i[child]] - p[i[bone]], secondary)
            if frame is None:
                raise ValueError(f"Mixamo bind frame is degenerate for {bone}.")
            frames[bone] = frame
        # Foot is an ankle node: Foot -> ToeBase slopes downward even though
        # the authored sole is flat. Halpe heel -> toe describes the sole, so
        # compare it with the bind bone vector projected onto the ground plane.
        for foot, toe in (
            ("LeftFoot", "LeftToeBase"),
            ("RightFoot", "RightToeBase"),
        ):
            sole_forward = p[i[toe]] - p[i[foot]]
            sole_forward = sole_forward - np.dot(sole_forward, body_up) * body_up
            frame = _frame(sole_forward, body_up)
            if frame is None:
                raise ValueError(
                    f"Mixamo bind sole frame is degenerate for {foot}."
                )
            frames[foot] = frame
        # The hand frame uses the middle and index/pinky metacarpal roots.
        for side in ("Left", "Right"):
            bone = f"{side}Hand"
            primary = p[i[f"{side}HandMiddle1"]] - p[i[bone]]
            lateral = p[i[f"{side}HandIndex1"]] - p[i[f"{side}HandPinky1"]]
            frame = _frame(primary, lateral)
            if frame is None:
                raise ValueError(f"Mixamo bind hand frame is degenerate for {side}.")
            frames[bone] = frame
        return frames

    def reset(self) -> None:
        self.previous_rotations = None

    def solve(
        self,
        joints_display_m: np.ndarray,
        confidence: np.ndarray,
        usable: np.ndarray,
        predicted: np.ndarray,
        *,
        delta_time_s: float,
        hands: Mapping[str, MixamoHandObservation] | None = None,
    ) -> MixamoIKFrame | None:
        joints = np.asarray(joints_display_m, dtype=np.float64)
        scores = np.asarray(confidence, dtype=np.float64)
        valid = np.asarray(usable, dtype=bool) & np.isfinite(joints).all(axis=1)
        predicted_mask = np.asarray(predicted, dtype=bool)
        if joints.shape != (26, 3) or scores.shape != (26,) or valid.shape != (26,):
            raise ValueError("Mixamo IK Halpe arrays have invalid shapes.")
        if predicted_mask.shape != (26,):
            raise ValueError("Mixamo IK predicted mask has invalid shape.")
        if not math.isfinite(delta_time_s) or delta_time_s <= 0:
            raise ValueError("Mixamo IK delta time must be finite and positive.")
        root = joints[19] if valid[19] else (
            0.5 * (joints[11] + joints[12]) if valid[11] and valid[12] else None
        )
        if root is None or not np.isfinite(root).all():
            self.reset()
            return None

        rejected: list[str] = []
        held: list[str] = []

        def accepted(name: str) -> bool:
            state = self.profile.classify(name, joints, scores, valid, predicted_mask)
            if state == "invalid":
                rejected.append(name)
                return False
            return True

        body_basis = None
        if accepted("torso") and (accepted("hip_axis") or accepted("shoulder_axis")):
            lateral = np.zeros(3, dtype=np.float64)
            if valid[11] and valid[12]:
                direction = _unit(joints[11] - joints[12])
                if direction is not None:
                    lateral += direction
            if valid[5] and valid[6]:
                direction = _unit(joints[5] - joints[6])
                if direction is not None:
                    lateral += direction
            body_basis = _body_basis(joints[18] - root, lateral)

        previous = self.previous_rotations
        rotations = np.zeros_like(self.bind_rotations)
        positions = np.zeros_like(self.bind_positions)
        maximum_step = math.radians(
            self.config.maximum_rotation_speed_deg_s * delta_time_s
        )
        if body_basis is not None:
            root_target_rotation = (
                body_basis @ self.bind_body_basis.T @ self.bind_rotations[0]
            )
            if previous is not None:
                root_target_rotation = _limited_rotation(
                    previous[0], root_target_rotation,
                    maximum_step, self.config.rotation_response,
                )
            rotations[0] = root_target_rotation
        elif previous is not None:
            rotations[0] = previous[0]
            held.append("Hips")
        else:
            rotations[0] = self.bind_rotations[0]
        positions[0] = root

        target_frames: dict[str, np.ndarray] = {}

        def add_frame(
            bone: str,
            primary: np.ndarray,
            secondary: np.ndarray,
            segment: str | None = None,
        ) -> None:
            if segment is not None and not accepted(segment):
                return
            frame = _frame(primary, secondary)
            if frame is not None:
                target_frames[bone] = frame
            else:
                held.append(bone)

        body_depth = body_basis[:, 1] if body_basis is not None else rotations[0] @ self.bind_body_basis[:, 1]
        body_up = body_basis[:, 2] if body_basis is not None else rotations[0] @ self.bind_body_basis[:, 2]
        if valid[18] and valid[5]:
            add_frame("LeftShoulder", joints[5] - joints[18], body_up, "shoulder_axis")
        if valid[18] and valid[6]:
            add_frame("RightShoulder", joints[6] - joints[18], body_up, "shoulder_axis")
        if valid[5] and valid[7]:
            secondary = joints[9] - joints[7] if valid[9] else body_depth
            add_frame("LeftArm", joints[7] - joints[5], secondary, "left_upper_arm")
        if valid[6] and valid[8]:
            secondary = joints[10] - joints[8] if valid[10] else body_depth
            add_frame("RightArm", joints[8] - joints[6], secondary, "right_upper_arm")
        if valid[7] and valid[9]:
            add_frame("LeftForeArm", joints[9] - joints[7], body_depth, "left_forearm")
        if valid[8] and valid[10]:
            add_frame("RightForeArm", joints[10] - joints[8], body_depth, "right_forearm")
        if valid[11] and valid[13]:
            secondary = joints[15] - joints[13] if valid[15] else body_depth
            add_frame("LeftUpLeg", joints[13] - joints[11], secondary, "left_thigh")
        if valid[12] and valid[14]:
            secondary = joints[16] - joints[14] if valid[16] else body_depth
            add_frame("RightUpLeg", joints[14] - joints[12], secondary, "right_thigh")
        if valid[13] and valid[15]:
            add_frame("LeftLeg", joints[15] - joints[13], body_depth, "left_shin")
        if valid[14] and valid[16]:
            add_frame("RightLeg", joints[16] - joints[14], body_depth, "right_shin")
        if valid[18] and valid[17]:
            add_frame("Neck", joints[17] - joints[18], body_depth, "neck_head")
        if valid[24] and valid[20] and valid[22]:
            toe = 0.5 * (joints[20] + joints[22])
            add_frame("LeftFoot", toe - joints[24], body_up, "left_foot_big")
        if valid[25] and valid[21] and valid[23]:
            toe = 0.5 * (joints[21] + joints[23])
            add_frame("RightFoot", toe - joints[25], body_up, "right_foot_big")

        for side, bone in (("left", "LeftHand"), ("right", "RightHand")):
            hand = hands.get(side) if hands is not None else None
            if hand is None:
                continue
            required = np.asarray((0, 5, 9, 17), dtype=np.int64)
            if not np.all(hand.valid[required]) or np.min(hand.confidence[required]) < self.config.minimum_hand_confidence:
                held.append(bone)
                continue
            add_frame(
                bone,
                hand.joints_display_m[9] - hand.joints_display_m[0],
                hand.joints_display_m[5] - hand.joints_display_m[17],
            )

        for bone in range(1, len(self.asset.bone_names)):
            parent = int(self.asset.parent_indices[bone])
            positions[bone] = positions[parent] + (
                rotations[parent] @ self.local_offsets[bone] * self.scale
            )
            name = self.asset.bone_names[bone]
            if name in target_frames:
                target_rotation = (
                    target_frames[name]
                    @ self.bind_frames[name].T
                    @ self.bind_rotations[bone]
                )
                if previous is not None:
                    target_rotation = _limited_rotation(
                        previous[bone], target_rotation,
                        maximum_step, self.config.rotation_response,
                    )
                rotations[bone] = target_rotation
            elif previous is not None and name in self.bind_frames:
                previous_parent = int(self.asset.parent_indices[bone])
                previous_local = previous[previous_parent].T @ previous[bone]
                rotations[bone] = rotations[parent] @ previous_local
                held.append(name)
            else:
                rotations[bone] = rotations[parent] @ self.local_rotations[bone]

        matrices = np.tile(np.eye(4), (len(self.asset.bone_names), 1, 1))
        matrices[:, :3, :3] = self.scale * rotations
        matrices[:, :3, 3] = positions
        self.previous_rotations = rotations.copy()
        return MixamoIKFrame(
            bone_global_m=matrices,
            root_display_m=root,
            rejected_segments=tuple(dict.fromkeys(rejected)),
            held_bones=tuple(dict.fromkeys(held)),
        )


__all__ = [
    "MixamoAnalyticalIK",
    "MixamoHandObservation",
    "MixamoIKConfig",
    "MixamoIKFrame",
    "estimate_mixamo_scale",
]
