"""Fit and cache SMPL meshes from metric Halpe26 joint sequences."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Literal, Sequence

import numpy as np

from rgbd_avatar.retargeting import RetargetedSMPLTargets


# SMPL is X horizontal, Y up, Z depth. The project display space is X right,
# Y forward, Z up. This is a proper +90 degree rotation around X.
DISPLAY_FROM_SMPL = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)

# (Halpe26 index, SMPL index). SMPL output joints 0..23 are the body rig;
# indices beyond 23 are vertex-selected convenience landmarks.
HALPE_TO_SMPL: tuple[tuple[int, int], ...] = (
    (0, 24),   # nose vertex
    (1, 26),   # left eye vertex
    (2, 25),   # right eye vertex
    (3, 28),   # left ear vertex
    (4, 27),   # right ear vertex
    (11, 1),   # left hip
    (12, 2),   # right hip
    (13, 4),   # left knee
    (14, 5),   # right knee
    (15, 7),   # left ankle
    (16, 8),   # right ankle
    (18, 12),  # neck
    (5, 16),   # left shoulder
    (6, 17),   # right shoulder
    (7, 18),   # left elbow
    (8, 19),   # right elbow
    (9, 20),   # left wrist
    (10, 21),  # right wrist
)

# Corresponding Halpe and SMPL bones used only to estimate one shared metric
# scale. Twist and pose do not change these lengths.
SCALE_BONES: tuple[tuple[int, int, int, int], ...] = (
    (11, 13, 1, 4),
    (13, 15, 4, 7),
    (12, 14, 2, 5),
    (14, 16, 5, 8),
    (5, 7, 16, 18),
    (7, 9, 18, 20),
    (6, 8, 17, 19),
    (8, 10, 19, 21),
    (11, 12, 1, 2),
    (5, 6, 16, 17),
)

# (Hand21 start/end, SMPL start/end, relative weight).  MCP/palm-base axes
# are intentionally used instead of articulated fingertips.  This makes hand
# orientation insensitive to whether the person opens or curls their fingers.
SMPL_HAND_DIRECTION_TARGETS: dict[
    str, tuple[tuple[int, int, int, int, float], ...]
] = {
    "left": (
        (0, 9, 20, 37, 1.0),    # wrist -> middle MCP / middle landmark
        (17, 5, 39, 36, 1.0),   # pinky MCP -> index MCP (palm lateral)
        (0, 2, 20, 35, 0.45),   # thumb MCP disambiguates palm handedness
    ),
    "right": (
        (0, 9, 21, 42, 1.0),
        (17, 5, 44, 41, 1.0),
        (0, 2, 21, 40, 0.45),
    ),
}

# (Halpe ankle, Halpe heel, Halpe big toe, Halpe small toe,
#  SMPL ankle, SMPL heel landmark, SMPL big-toe landmark,
#  SMPL small-toe landmark).  As with hands, heel-to-toe vectors constrain
# orientation only; detected foot dimensions cannot stretch the avatar.
SMPL_FOOT_DIRECTION_TARGETS: tuple[
    tuple[int, int, int, int, int, int, int, int], ...
] = (
    (15, 24, 20, 22, 7, 31, 29, 30),
    (16, 25, 21, 23, 8, 34, 32, 33),
)

# These terminal joints are held at their neutral local rotations. Their
# parents (ankles and wrists) still rotate freely, so the rigid foot/hand can
# follow an observed direction without curling or folding its local shape.
RIGID_END_EFFECTOR_JOINTS: tuple[int, ...] = (10, 11, 22, 23)


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest without loading a file at once."""

    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class SMPLFitConfig:
    iterations: int = 160
    learning_rate: float = 0.045
    huber_delta_m: float = 0.035
    pose_prior_weight: float = 4e-4
    spine_pose_weight: float = 5e-2
    temporal_pose_weight: float = 1.5e-3
    temporal_orient_weight: float = 5e-4
    predicted_weight_scale: float = 0.25
    minimum_joint_weight: float = 0.05
    face_target_weight_scale: float = 0.35
    hand_target_weight_scale: float = 0.50
    end_effector_direction_weight: float = 0.02
    rigid_end_effectors: bool = True
    minimum_hand_confidence: float = 0.15
    minimum_target_count: int = 10
    early_stop_delta: float = 1e-8
    early_stop_patience: int = 12

    def __post_init__(self) -> None:
        positive = (
            self.iterations,
            self.learning_rate,
            self.huber_delta_m,
            self.pose_prior_weight,
            self.spine_pose_weight,
            self.temporal_pose_weight,
            self.temporal_orient_weight,
            self.minimum_joint_weight,
            self.face_target_weight_scale,
            self.hand_target_weight_scale,
            self.end_effector_direction_weight,
            self.minimum_hand_confidence,
            self.minimum_target_count,
            self.early_stop_delta,
            self.early_stop_patience,
        )
        if any(not math.isfinite(float(value)) or value <= 0 for value in positive):
            raise ValueError("SMPL fitting parameters must be positive.")
        if not 0 < self.predicted_weight_scale <= 1:
            raise ValueError("predicted_weight_scale must be in (0, 1].")
        if not isinstance(self.rigid_end_effectors, bool):
            raise ValueError("rigid_end_effectors must be boolean.")


@dataclass(frozen=True)
class SMPLTarget:
    smpl_joint_indices: np.ndarray
    points_native_m: np.ndarray
    weights: np.ndarray
    smpl_direction_pairs: np.ndarray
    directions_native: np.ndarray
    direction_weights: np.ndarray

    def __post_init__(self) -> None:
        indices = np.asarray(self.smpl_joint_indices, dtype=np.int64)
        points = np.asarray(self.points_native_m, dtype=np.float64)
        weights = np.asarray(self.weights, dtype=np.float64)
        pairs = np.asarray(self.smpl_direction_pairs, dtype=np.int64)
        directions = np.asarray(self.directions_native, dtype=np.float64)
        direction_weights = np.asarray(self.direction_weights, dtype=np.float64)
        if indices.ndim != 1 or points.shape != (len(indices), 3):
            raise ValueError("SMPL positional targets have invalid shapes.")
        if weights.shape != (len(indices),):
            raise ValueError("SMPL positional target weights have invalid shape.")
        if pairs.shape != (len(pairs), 2):
            raise ValueError("SMPL direction pairs must have shape Nx2.")
        if directions.shape != (len(pairs), 3):
            raise ValueError("SMPL direction targets must have shape Nx3.")
        if direction_weights.shape != (len(pairs),):
            raise ValueError("SMPL direction weights have invalid shape.")
        if np.any(indices < 0) or np.any(pairs < 0):
            raise ValueError("SMPL target indices must be non-negative.")
        if (
            not np.isfinite(points).all()
            or not np.isfinite(weights).all()
            or not np.isfinite(directions).all()
            or not np.isfinite(direction_weights).all()
        ):
            raise ValueError("SMPL targets must be finite.")
        if np.any(weights <= 0) or np.any(direction_weights <= 0):
            raise ValueError("SMPL target weights must be positive.")
        if len(directions) and not np.allclose(
            np.linalg.norm(directions, axis=1),
            1.0,
            atol=1e-6,
        ):
            raise ValueError("SMPL direction targets must be unit vectors.")
        object.__setattr__(self, "smpl_joint_indices", indices)
        object.__setattr__(self, "points_native_m", points)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "smpl_direction_pairs", pairs)
        object.__setattr__(self, "directions_native", directions)
        object.__setattr__(self, "direction_weights", direction_weights)

    @property
    def count(self) -> int:
        return len(self.smpl_joint_indices)

    @property
    def direction_count(self) -> int:
        return len(self.smpl_direction_pairs)


@dataclass(frozen=True)
class SMPLHandObservation:
    side: Literal["left", "right"]
    joints_display_m: np.ndarray
    confidence: np.ndarray
    valid: np.ndarray

    def __post_init__(self) -> None:
        joints = np.asarray(self.joints_display_m, dtype=np.float64)
        confidence = np.asarray(self.confidence, dtype=np.float64)
        valid = np.asarray(self.valid, dtype=bool)
        if self.side not in SMPL_HAND_DIRECTION_TARGETS:
            raise ValueError(f"Unsupported hand side: {self.side!r}.")
        if joints.shape != (21, 3):
            raise ValueError("SMPL hand observation joints must be 21x3.")
        if confidence.shape != (21,) or valid.shape != (21,):
            raise ValueError("SMPL hand confidence and valid must have length 21.")
        if np.any(valid) and not np.isfinite(joints[valid]).all():
            raise ValueError("Valid SMPL hand observations must be finite.")
        object.__setattr__(self, "joints_display_m", joints)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "valid", valid)


@dataclass(frozen=True)
class SMPLFrameFit:
    body_pose: np.ndarray
    global_orient: np.ndarray
    translation_native_m: np.ndarray
    vertices_display_m: np.ndarray
    joints_display_m: np.ndarray
    target_count: int
    error_mean_m: float
    error_p95_m: float
    error_max_m: float
    iterations: int


@dataclass(frozen=True)
class SMPLSequenceCache:
    frame_indices: np.ndarray
    present: np.ndarray
    vertices_display_m: np.ndarray
    joints_display_m: np.ndarray
    faces: np.ndarray
    body_pose: np.ndarray
    global_orient: np.ndarray
    translation_native_m: np.ndarray
    target_counts: np.ndarray
    error_mean_m: np.ndarray
    error_p95_m: np.ndarray
    error_max_m: np.ndarray
    scale: float
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        frame_indices = np.asarray(self.frame_indices, dtype=np.int64)
        present = np.asarray(self.present, dtype=bool)
        vertices = np.asarray(self.vertices_display_m, dtype=np.float32)
        joints = np.asarray(self.joints_display_m, dtype=np.float32)
        faces = np.asarray(self.faces, dtype=np.int32)
        frame_count = len(frame_indices)
        if frame_count == 0 or np.any(np.diff(frame_indices) <= 0):
            raise ValueError("SMPL cache frame indices must increase.")
        if present.shape != (frame_count,):
            raise ValueError("SMPL present mask has the wrong shape.")
        if (
            vertices.ndim != 3
            or vertices.shape[0] != frame_count
            or vertices.shape[2] != 3
        ):
            raise ValueError("SMPL cache vertices must have shape FxVx3.")
        if joints.shape != (frame_count, 24, 3):
            raise ValueError("SMPL cache joints must have shape Fx24x3.")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("SMPL cache faces must have shape Tx3.")
        if faces.size == 0 or faces.min() < 0 or faces.max() >= vertices.shape[1]:
            raise ValueError("SMPL cache contains invalid face indices.")
        if np.any(present) and (
            not np.isfinite(vertices[present]).all()
            or not np.isfinite(joints[present]).all()
        ):
            raise ValueError("Present SMPL frames must be finite.")
        if np.any(~present) and (
            not np.isnan(vertices[~present]).all()
            or not np.isnan(joints[~present]).all()
        ):
            raise ValueError("Absent SMPL frames must contain NaN geometry.")
        for name, array, shape in (
            ("body_pose", self.body_pose, (frame_count, 69)),
            ("global_orient", self.global_orient, (frame_count, 3)),
            ("translation_native_m", self.translation_native_m, (frame_count, 3)),
            ("target_counts", self.target_counts, (frame_count,)),
            ("error_mean_m", self.error_mean_m, (frame_count,)),
            ("error_p95_m", self.error_p95_m, (frame_count,)),
            ("error_max_m", self.error_max_m, (frame_count,)),
        ):
            if np.asarray(array).shape != shape:
                raise ValueError(f"SMPL cache {name} has the wrong shape.")
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("SMPL cache scale must be positive.")

        object.__setattr__(self, "frame_indices", frame_indices)
        object.__setattr__(self, "present", present)
        object.__setattr__(self, "vertices_display_m", vertices)
        object.__setattr__(self, "joints_display_m", joints)
        object.__setattr__(self, "faces", faces)

    def save(self, path: str | Path) -> None:
        output = Path(path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{output.stem}.",
                suffix=".npz",
                dir=output.parent,
                delete=False,
            ) as file:
                np.savez_compressed(
                    file,
                    schema_version=np.asarray(1, dtype=np.int32),
                    frame_indices=self.frame_indices,
                    present=self.present,
                    vertices_display_m=self.vertices_display_m,
                    joints_display_m=self.joints_display_m,
                    faces=self.faces,
                    body_pose=np.asarray(self.body_pose, dtype=np.float32),
                    global_orient=np.asarray(self.global_orient, dtype=np.float32),
                    translation_native_m=np.asarray(
                        self.translation_native_m,
                        dtype=np.float32,
                    ),
                    target_counts=np.asarray(self.target_counts, dtype=np.int16),
                    error_mean_m=np.asarray(self.error_mean_m, dtype=np.float32),
                    error_p95_m=np.asarray(self.error_p95_m, dtype=np.float32),
                    error_max_m=np.asarray(self.error_max_m, dtype=np.float32),
                    scale=np.asarray(self.scale, dtype=np.float64),
                    metadata_json=np.asarray(
                        json.dumps(self.metadata, sort_keys=True),
                    ),
                )
                file.flush()
                os.fsync(file.fileno())
                temporary_path = Path(file.name)
            os.replace(temporary_path, output)
            temporary_path = None
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    @classmethod
    def load(cls, path: str | Path) -> "SMPLSequenceCache":
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"SMPL sequence cache not found: {source}")
        with np.load(source, allow_pickle=False) as payload:
            if int(payload["schema_version"]) != 1:
                raise ValueError("Unsupported SMPL cache schema version.")
            metadata = json.loads(str(payload["metadata_json"]))
            return cls(
                frame_indices=payload["frame_indices"].copy(),
                present=payload["present"].copy(),
                vertices_display_m=payload["vertices_display_m"].copy(),
                joints_display_m=payload["joints_display_m"].copy(),
                faces=payload["faces"].copy(),
                body_pose=payload["body_pose"].copy(),
                global_orient=payload["global_orient"].copy(),
                translation_native_m=payload["translation_native_m"].copy(),
                target_counts=payload["target_counts"].copy(),
                error_mean_m=payload["error_mean_m"].copy(),
                error_p95_m=payload["error_p95_m"].copy(),
                error_max_m=payload["error_max_m"].copy(),
                scale=float(payload["scale"]),
                metadata=metadata,
            )


def smpl_to_display(points_native_m: np.ndarray) -> np.ndarray:
    points = np.asarray(points_native_m)
    if points.shape[-1] != 3:
        raise ValueError("SMPL points must end in XYZ.")
    return points @ DISPLAY_FROM_SMPL.T


def display_to_smpl(points_display_m: np.ndarray) -> np.ndarray:
    points = np.asarray(points_display_m)
    if points.shape[-1] != 3:
        raise ValueError("Display points must end in XYZ.")
    return points @ DISPLAY_FROM_SMPL


def build_smpl_target(
    joints_display_m: np.ndarray,
    confidence: np.ndarray,
    usable: np.ndarray,
    predicted: np.ndarray,
    *,
    config: SMPLFitConfig,
    hand_observations: Sequence[SMPLHandObservation] = (),
    retargeted_body: RetargetedSMPLTargets | None = None,
) -> SMPLTarget:
    joints = np.asarray(joints_display_m, dtype=np.float64)
    scores = np.asarray(confidence, dtype=np.float64)
    valid = np.asarray(usable, dtype=bool) & np.isfinite(joints).all(axis=1)
    predicted_mask = np.asarray(predicted, dtype=bool)
    if joints.shape != (26, 3) or scores.shape != (26,):
        raise ValueError("Halpe target arrays have invalid shapes.")
    if valid.shape != (26,) or predicted_mask.shape != (26,):
        raise ValueError("Halpe target masks have invalid shapes.")

    target_indices: list[int] = (
        retargeted_body.smpl_joint_indices.tolist()
        if retargeted_body is not None
        else []
    )
    target_points: list[np.ndarray] = (
        [point.copy() for point in retargeted_body.points_display_m]
        if retargeted_body is not None
        else []
    )
    target_weights: list[float] = (
        retargeted_body.weights.tolist()
        if retargeted_body is not None
        else []
    )
    direction_pairs: list[tuple[int, int]] = (
        [tuple(pair) for pair in retargeted_body.smpl_direction_pairs]
        if retargeted_body is not None
        else []
    )
    target_directions: list[np.ndarray] = (
        [direction.copy() for direction in retargeted_body.directions_display]
        if retargeted_body is not None
        else []
    )
    direction_weights: list[float] = (
        retargeted_body.direction_weights.tolist()
        if retargeted_body is not None
        else []
    )

    def append_direction(
        smpl_start: int,
        smpl_end: int,
        observed_start: np.ndarray,
        observed_end: np.ndarray,
        weight: float,
        *,
        minimum_length_m: float,
        maximum_length_m: float,
    ) -> None:
        vector = np.asarray(observed_end) - np.asarray(observed_start)
        length = float(np.linalg.norm(vector))
        if not minimum_length_m <= length <= maximum_length_m:
            return
        direction_pairs.append((smpl_start, smpl_end))
        target_directions.append(vector / length)
        direction_weights.append(max(config.minimum_joint_weight, weight))

    if retargeted_body is None:
        for halpe_index, smpl_index in HALPE_TO_SMPL:
            if not valid[halpe_index]:
                continue
            weight = float(
                np.clip(
                    scores[halpe_index],
                    config.minimum_joint_weight,
                    1.0,
                )
            )
            if predicted_mask[halpe_index]:
                weight *= config.predicted_weight_scale
            if halpe_index <= 4:
                weight *= config.face_target_weight_scale
            target_indices.append(smpl_index)
            target_points.append(joints[halpe_index])
            target_weights.append(weight)

        for (
            ankle_index,
            heel_index,
            big_toe_index,
            small_toe_index,
            _smpl_ankle_index,
            smpl_heel_index,
            smpl_big_toe_index,
            smpl_small_toe_index,
        ) in SMPL_FOOT_DIRECTION_TARGETS:
            if not (valid[ankle_index] and valid[heel_index]):
                continue
            foot_indices = (ankle_index, heel_index)
            foot_weight = float(np.mean(scores[list(foot_indices)])) * 0.65
            if any(predicted_mask[index] for index in foot_indices):
                foot_weight *= config.predicted_weight_scale
            for toe_index, smpl_toe_index in (
                (big_toe_index, smpl_big_toe_index),
                (small_toe_index, smpl_small_toe_index),
            ):
                if not valid[toe_index]:
                    continue
                toe_weight = 0.5 * (
                    foot_weight
                    + float(scores[toe_index])
                    * (
                        config.predicted_weight_scale
                        if predicted_mask[toe_index]
                        else 1.0
                    )
                )
                append_direction(
                    smpl_heel_index,
                    smpl_toe_index,
                    joints[heel_index],
                    joints[toe_index],
                    toe_weight,
                    minimum_length_m=0.04,
                    maximum_length_m=0.45,
                )

    for observation in hand_observations:
        for (
            hand_start,
            hand_end,
            smpl_start,
            smpl_end,
            relative_weight,
        ) in SMPL_HAND_DIRECTION_TARGETS[observation.side]:
            confidence = float(
                min(
                    observation.confidence[hand_start],
                    observation.confidence[hand_end],
                )
            )
            if (
                not (
                    observation.valid[hand_start]
                    and observation.valid[hand_end]
                )
                or confidence < config.minimum_hand_confidence
            ):
                continue
            append_direction(
                smpl_start,
                smpl_end,
                observation.joints_display_m[hand_start],
                observation.joints_display_m[hand_end],
                confidence
                * config.hand_target_weight_scale
                * relative_weight,
                minimum_length_m=0.015,
                maximum_length_m=0.18,
            )

    native_directions = (
        display_to_smpl(np.asarray(target_directions, dtype=np.float64))
        if target_directions
        else np.empty((0, 3), dtype=np.float64)
    )
    return SMPLTarget(
        smpl_joint_indices=np.asarray(target_indices, dtype=np.int64),
        points_native_m=(
            display_to_smpl(np.asarray(target_points))
            if target_points
            else np.empty((0, 3), dtype=np.float64)
        ),
        weights=np.asarray(target_weights, dtype=np.float64),
        smpl_direction_pairs=np.asarray(direction_pairs, dtype=np.int64).reshape(-1, 2),
        directions_native=native_directions,
        direction_weights=np.asarray(direction_weights, dtype=np.float64),
    )


def estimate_smpl_scale(
    joints_display_sequence: Sequence[np.ndarray],
    usable_sequence: Sequence[np.ndarray],
    rest_joints_native_m: np.ndarray,
) -> float:
    rest = np.asarray(rest_joints_native_m, dtype=np.float64)
    if rest.shape[0] < 24 or rest.shape[1] != 3:
        raise ValueError("SMPL rest joints must contain at least 24 XYZ rows.")
    ratios: list[float] = []
    for joints_value, usable_value in zip(
        joints_display_sequence,
        usable_sequence,
    ):
        joints = np.asarray(joints_value, dtype=np.float64)
        usable = np.asarray(usable_value, dtype=bool)
        for halpe_start, halpe_end, smpl_start, smpl_end in SCALE_BONES:
            if not (usable[halpe_start] and usable[halpe_end]):
                continue
            target_length = float(
                np.linalg.norm(joints[halpe_end] - joints[halpe_start])
            )
            rest_length = float(np.linalg.norm(rest[smpl_end] - rest[smpl_start]))
            if 0.05 < target_length < 0.8 and rest_length > 0.02:
                ratio = target_length / rest_length
                if 0.65 <= ratio <= 1.45:
                    ratios.append(ratio)
    if len(ratios) < 8:
        raise ValueError("Too few plausible bones to estimate SMPL scale.")
    return float(np.clip(np.median(ratios), 0.75, 1.35))


class SMPLSequenceFitter:
    """Differentiable joint fitting with temporal warm starts."""

    def __init__(
        self,
        model: Any,
        *,
        scale: float,
        device: str,
        config: SMPLFitConfig,
        betas: np.ndarray | None = None,
    ) -> None:
        import torch

        self.torch = torch
        self.model = model
        self.scale = float(scale)
        self.device = device
        self.config = config
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        beta_values = (
            np.zeros(int(model.num_betas), dtype=np.float32)
            if betas is None
            else np.asarray(betas, dtype=np.float32)
        )
        if beta_values.shape != (int(model.num_betas),):
            raise ValueError(
                f"Expected {model.num_betas} SMPL betas, "
                f"got {beta_values.shape}."
            )
        if not np.isfinite(beta_values).all():
            raise ValueError("SMPL betas must be finite.")
        self.betas = torch.as_tensor(
            beta_values,
            dtype=torch.float32,
            device=device,
        ).reshape(1, -1)
        self.zero_body_pose = torch.zeros((1, 69), device=device)
        self.zero_orient = torch.zeros((1, 3), device=device)

    def fit(
        self,
        target: SMPLTarget,
        previous: SMPLFrameFit | None,
    ) -> SMPLFrameFit:
        torch = self.torch
        config = self.config
        if target.count < config.minimum_target_count:
            raise ValueError(
                f"SMPL target has only {target.count} joints; "
                f"need {config.minimum_target_count}."
            )

        body_initial = (
            previous.body_pose.copy()
            if previous is not None
            else np.zeros(69, dtype=np.float32)
        )
        end_effector_pose_indices = np.asarray(
            [
                component
                for smpl_joint in RIGID_END_EFFECTOR_JOINTS
                for component in range(
                    3 * (smpl_joint - 1),
                    3 * (smpl_joint - 1) + 3,
                )
            ],
            dtype=np.int64,
        )
        if config.rigid_end_effectors:
            body_initial[end_effector_pose_indices] = 0.0
        orient_initial = (
            previous.global_orient
            if previous is not None
            else np.zeros(3, dtype=np.float32)
        )
        body_pose = torch.nn.Parameter(
            torch.as_tensor(body_initial, device=self.device).reshape(1, 69).clone()
        )
        if config.rigid_end_effectors:
            gradient_mask = torch.ones((1, 69), device=self.device)
            gradient_mask[:, end_effector_pose_indices.tolist()] = 0.0
            body_pose.register_hook(lambda gradient: gradient * gradient_mask)
        global_orient = torch.nn.Parameter(
            torch.as_tensor(orient_initial, device=self.device).reshape(1, 3).clone()
        )
        joint_indices = torch.as_tensor(
            target.smpl_joint_indices,
            dtype=torch.long,
            device=self.device,
        )
        target_points = torch.as_tensor(
            target.points_native_m,
            dtype=torch.float32,
            device=self.device,
        )
        weights = torch.as_tensor(
            target.weights,
            dtype=torch.float32,
            device=self.device,
        )
        direction_pairs = torch.as_tensor(
            target.smpl_direction_pairs,
            dtype=torch.long,
            device=self.device,
        )
        target_directions = torch.as_tensor(
            target.directions_native,
            dtype=torch.float32,
            device=self.device,
        )
        direction_weights = torch.as_tensor(
            target.direction_weights,
            dtype=torch.float32,
            device=self.device,
        )

        with torch.no_grad():
            initial_output = self.model(
                betas=self.betas,
                body_pose=body_pose,
                global_orient=global_orient,
                return_verts=False,
            )
            initial_targets = (
                self.scale * initial_output.joints[0, joint_indices]
            )
            translation_initial = torch.sum(
                weights[:, None] * (target_points - initial_targets),
                dim=0,
            ) / torch.sum(weights)
        translation = torch.nn.Parameter(
            translation_initial.reshape(1, 3).clone()
        )

        previous_body = (
            torch.as_tensor(previous.body_pose, device=self.device).reshape(1, 69)
            if previous is not None
            else None
        )
        previous_orient = (
            torch.as_tensor(previous.global_orient, device=self.device).reshape(1, 3)
            if previous is not None
            else None
        )
        optimizer = torch.optim.Adam(
            (body_pose, global_orient, translation),
            lr=config.learning_rate,
        )
        prior_weights = torch.ones((1, 69), device=self.device)
        # Spine/collar rotations are underconstrained by positional joints.
        for smpl_joint in (3, 6, 9, 13, 14, 22, 23):
            start = 3 * (smpl_joint - 1)
            prior_weights[:, start : start + 3] = 2.5
        spine_pose_indices = torch.as_tensor(
            [
                component
                for smpl_joint in (3, 6, 9)
                for component in range(
                    3 * (smpl_joint - 1),
                    3 * (smpl_joint - 1) + 3,
                )
            ],
            dtype=torch.long,
            device=self.device,
        )

        previous_loss: float | None = None
        stable_steps = 0
        used_iterations = config.iterations
        for iteration in range(config.iterations):
            optimizer.zero_grad(set_to_none=True)
            output = self.model(
                betas=self.betas,
                body_pose=body_pose,
                global_orient=global_orient,
                return_verts=False,
            )
            predicted = (
                self.scale * output.joints[0, joint_indices]
                + translation[0]
            )
            residual = predicted - target_points
            absolute = torch.abs(residual)
            delta = config.huber_delta_m
            huber = torch.where(
                absolute <= delta,
                0.5 * residual.square() / delta,
                absolute - 0.5 * delta,
            )
            data_loss = torch.sum(
                weights[:, None] * huber
            ) / (3.0 * torch.sum(weights))
            loss = data_loss + config.pose_prior_weight * torch.mean(
                prior_weights * body_pose.square()
            )
            if target.direction_count:
                model_vectors = (
                    output.joints[0, direction_pairs[:, 1]]
                    - output.joints[0, direction_pairs[:, 0]]
                )
                model_directions = torch.nn.functional.normalize(
                    model_vectors,
                    dim=1,
                    eps=1e-8,
                )
                cosine_distance = 1.0 - torch.sum(
                    model_directions * target_directions,
                    dim=1,
                ).clamp(-1.0, 1.0)
                direction_loss = torch.sum(
                    direction_weights * cosine_distance
                ) / torch.sum(direction_weights)
                loss = loss + (
                    config.end_effector_direction_weight * direction_loss
                )
            # Halpe provides shoulders, hips and neck, but no observations for
            # SMPL spine1/2/3.  A dedicated prior prevents those three latent
            # joints from forming an arbitrary S-curve while global orientation
            # and the hip joints still represent rigid torso lean.
            loss = loss + config.spine_pose_weight * torch.mean(
                body_pose[:, spine_pose_indices].square()
            )
            if previous_body is not None and previous_orient is not None:
                loss = loss + config.temporal_pose_weight * torch.mean(
                    (body_pose - previous_body).square()
                )
                loss = loss + config.temporal_orient_weight * torch.mean(
                    (global_orient - previous_orient).square()
                )
            loss.backward()
            optimizer.step()

            loss_value = float(loss.detach().cpu())
            if (
                previous_loss is not None
                and abs(previous_loss - loss_value)
                < config.early_stop_delta
            ):
                stable_steps += 1
                if stable_steps >= config.early_stop_patience:
                    used_iterations = iteration + 1
                    break
            else:
                stable_steps = 0
            previous_loss = loss_value

        with torch.no_grad():
            output = self.model(
                betas=self.betas,
                body_pose=body_pose,
                global_orient=global_orient,
                return_verts=True,
            )
            joints_native = (
                self.scale * output.joints[0, :24] + translation[0]
            )
            vertices_native = (
                self.scale * output.vertices[0] + translation[0]
            )
            fitted_targets = (
                self.scale * output.joints[0, joint_indices]
                + translation[0]
            )
            errors = torch.linalg.norm(
                fitted_targets - target_points,
                dim=1,
            ).cpu().numpy()

        return SMPLFrameFit(
            body_pose=body_pose.detach().cpu().numpy()[0].astype(np.float32),
            global_orient=(
                global_orient.detach().cpu().numpy()[0].astype(np.float32)
            ),
            translation_native_m=(
                translation.detach().cpu().numpy()[0].astype(np.float32)
            ),
            vertices_display_m=smpl_to_display(
                vertices_native.cpu().numpy()
            ).astype(np.float32),
            joints_display_m=smpl_to_display(
                joints_native.cpu().numpy()
            ).astype(np.float32),
            target_count=target.count,
            error_mean_m=float(np.mean(errors)),
            error_p95_m=float(np.percentile(errors, 95)),
            error_max_m=float(np.max(errors)),
            iterations=used_iterations,
        )
