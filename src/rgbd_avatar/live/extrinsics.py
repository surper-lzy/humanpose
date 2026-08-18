"""Application-space camera extrinsics used by the live avatar pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np


def rotation_zyx_from_degrees(
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float,
) -> np.ndarray:
    """Return ``Rz(yaw) @ Ry(pitch) @ Rx(roll)`` for column vectors."""

    angles = np.asarray((roll_deg, pitch_deg, yaw_deg), dtype=np.float64)
    if not np.isfinite(angles).all():
        raise ValueError("Roll, pitch, and yaw must be finite.")
    roll, pitch, yaw = np.deg2rad(angles)
    rotation_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(roll), -np.sin(roll)],
            [0.0, np.sin(roll), np.cos(roll)],
        ],
        dtype=np.float64,
    )
    rotation_y = np.array(
        [
            [np.cos(pitch), 0.0, np.sin(pitch)],
            [0.0, 1.0, 0.0],
            [-np.sin(pitch), 0.0, np.cos(pitch)],
        ],
        dtype=np.float64,
    )
    rotation_z = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return rotation_z @ rotation_y @ rotation_x


@dataclass(frozen=True)
class ApplicationExtrinsics:
    """Rigid transform ``p_application = R @ p_camera + t`` in metres."""

    roll_deg: float
    pitch_deg: float
    yaw_deg: float
    translation_m: np.ndarray

    def __post_init__(self) -> None:
        angles = np.asarray(
            (self.roll_deg, self.pitch_deg, self.yaw_deg),
            dtype=np.float64,
        )
        translation = np.asarray(self.translation_m, dtype=np.float64)
        if not np.isfinite(angles).all():
            raise ValueError("Application extrinsic angles must be finite.")
        if translation.shape != (3,) or not np.isfinite(translation).all():
            raise ValueError("translation_m must be a finite XYZ vector.")
        object.__setattr__(self, "roll_deg", float(self.roll_deg))
        object.__setattr__(self, "pitch_deg", float(self.pitch_deg))
        object.__setattr__(self, "yaw_deg", float(self.yaw_deg))
        object.__setattr__(self, "translation_m", translation.copy())

    @property
    def rotation_application_from_camera(self) -> np.ndarray:
        return rotation_zyx_from_degrees(
            self.roll_deg,
            self.pitch_deg,
            self.yaw_deg,
        )

    @property
    def matrix(self) -> np.ndarray:
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = self.rotation_application_from_camera
        transform[:3, 3] = self.translation_m
        return transform

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> "ApplicationExtrinsics":
        if not isinstance(values, Mapping):
            raise ValueError("live.application_extrinsics must be a mapping.")
        if values.get("transform") != "application_from_camera":
            raise ValueError(
                "application_extrinsics.transform must be "
                "application_from_camera."
            )
        if values.get("euler_order") != "ZYX":
            raise ValueError("Only ZYX application extrinsics are supported.")
        translation = values.get("translation_m")
        if not isinstance(translation, (list, tuple)) or len(translation) != 3:
            raise ValueError(
                "application_extrinsics.translation_m must contain XYZ."
            )
        return cls(
            roll_deg=float(values["roll_deg"]),
            pitch_deg=float(values["pitch_deg"]),
            yaw_deg=float(values["yaw_deg"]),
            translation_m=np.asarray(translation, dtype=np.float64),
        )

    @classmethod
    def from_rotation_translation(
        cls,
        rotation_application_from_camera: np.ndarray,
        translation_m: np.ndarray,
    ) -> "ApplicationExtrinsics":
        """Build from a proper rotation while retaining the public ZYX form."""

        rotation = np.asarray(
            rotation_application_from_camera,
            dtype=np.float64,
        )
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
            raise ValueError("rotation must be a finite 3x3 matrix.")
        if not np.allclose(
            rotation.T @ rotation,
            np.eye(3),
            atol=1e-7,
        ):
            raise ValueError("rotation must be orthonormal.")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-7):
            raise ValueError("rotation must have determinant +1.")

        sin_pitch = float(np.clip(-rotation[2, 0], -1.0, 1.0))
        pitch = math.asin(sin_pitch)
        if abs(math.cos(pitch)) > 1e-7:
            roll = math.atan2(rotation[2, 1], rotation[2, 2])
            yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        else:
            # ZYX has one free angle at gimbal lock. Fix roll to zero and
            # recover the equivalent yaw; the resulting matrix stays exact.
            roll = 0.0
            yaw = math.atan2(-rotation[0, 1], rotation[1, 1])
        return cls(
            roll_deg=math.degrees(roll),
            pitch_deg=math.degrees(pitch),
            yaw_deg=math.degrees(yaw),
            translation_m=np.asarray(translation_m, dtype=np.float64),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "transform": "application_from_camera",
            "euler_order": "ZYX",
            "roll_deg": self.roll_deg,
            "pitch_deg": self.pitch_deg,
            "yaw_deg": self.yaw_deg,
            "translation_m": self.translation_m.tolist(),
        }

    def transform_points(self, points_camera_m: np.ndarray) -> np.ndarray:
        points = np.asarray(points_camera_m, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Expected Nx3 camera points, got {points.shape}.")
        finite = np.isfinite(points).all(axis=1)
        transformed = np.full_like(points, np.nan)
        transformed[finite] = (
            points[finite] @ self.rotation_application_from_camera.T
            + self.translation_m
        )
        return transformed
