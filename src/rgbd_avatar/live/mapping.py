"""Map live metric camera joints into an authored 3DGS placement."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rgbd_avatar.pose import HALPE26_NAMES
from rgbd_avatar.scene import SceneAlignment


# Upright optical camera C: +X right, +Y down, +Z forward.
# Live avatar world L: +X right, +Y forward, +Z up.
OPTICAL_UPRIGHT_L_FROM_C = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
    dtype=np.float64,
)
HALPE_FOOT_INDICES = np.asarray((15, 16, 20, 21, 22, 23, 24, 25))


def _proper_rotation(value: np.ndarray) -> np.ndarray:
    rotation = np.asarray(value, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("rotation_l_from_c must be a finite 3x3 matrix.")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError("rotation_l_from_c must be orthonormal.")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError("rotation_l_from_c must have determinant +1.")
    return rotation


@dataclass(frozen=True)
class LiveSceneMapper:
    """Place live Halpe26 joints at the saved 3DGS spawn and axes."""

    alignment: SceneAlignment
    mode: str = "root_locked"
    rotation_l_from_c: np.ndarray = None  # type: ignore[assignment]
    origin_camera_m: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.alignment, SceneAlignment):
            raise TypeError("alignment must be a SceneAlignment.")
        if self.mode not in ("root_locked", "fixed_origin"):
            raise ValueError("mode must be root_locked or fixed_origin.")
        rotation_value = (
            OPTICAL_UPRIGHT_L_FROM_C
            if self.rotation_l_from_c is None
            else self.rotation_l_from_c
        )
        rotation = _proper_rotation(rotation_value)
        origin = None
        if self.origin_camera_m is not None:
            origin = np.asarray(self.origin_camera_m, dtype=np.float64)
            if origin.shape != (3,) or not np.isfinite(origin).all():
                raise ValueError("origin_camera_m must be a finite XYZ vector.")
        if self.mode == "fixed_origin" and origin is None:
            raise ValueError("fixed_origin mode requires origin_camera_m.")
        object.__setattr__(self, "rotation_l_from_c", rotation)
        object.__setattr__(self, "origin_camera_m", origin)

    def map_joints(
        self,
        joints_camera_m: np.ndarray,
        usable: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return G joints and the C-space point mapped to the scene spawn."""

        count = len(HALPE26_NAMES)
        joints_c = np.asarray(joints_camera_m, dtype=np.float64)
        valid = np.asarray(usable, dtype=bool)
        if joints_c.shape != (count, 3) or valid.shape != (count,):
            raise ValueError("Live Halpe joints/usable must have shapes (26,3)/(26,).")
        valid &= np.isfinite(joints_c).all(axis=1)
        if not np.any(valid):
            raise ValueError("No usable live joint can be mapped.")

        if self.mode == "root_locked":
            feet = HALPE_FOOT_INDICES[valid[HALPE_FOOT_INDICES]]
            if len(feet) < 2:
                raise ValueError(
                    "root_locked mapping requires at least two usable foot joints."
                )
            anchor_camera = np.median(joints_c[feet], axis=0)
        else:
            assert self.origin_camera_m is not None
            anchor_camera = self.origin_camera_m

        joints_l = np.full_like(joints_c, np.nan)
        joints_l[valid] = (
            (joints_c[valid] - anchor_camera)
            @ self.rotation_l_from_c.T
        )
        joints_g = np.full_like(joints_l, np.nan)
        joints_g[valid] = (
            self.alignment.spawn_point_g
            + self.alignment.scale_g_per_m
            * (joints_l[valid] @ self.alignment.rotation_g_from_w.T)
        )
        return joints_g.astype(np.float32), anchor_camera.copy()
