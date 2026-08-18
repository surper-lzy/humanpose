"""Cached Gaussian RGB/expected-depth views used for precise scene picking."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GaussianAlignmentView:
    """A rendered 3DGS image plus enough data to unproject visible pixels."""

    rgb_uint8: np.ndarray
    expected_depth_g: np.ndarray
    alpha: np.ndarray
    intrinsic_matrix: np.ndarray
    camera_to_world_g: np.ndarray
    camera_name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        rgb = np.asarray(self.rgb_uint8, dtype=np.uint8)
        depth = np.asarray(self.expected_depth_g, dtype=np.float32)
        alpha = np.asarray(self.alpha, dtype=np.float32)
        intrinsic = np.asarray(self.intrinsic_matrix, dtype=np.float64)
        camera_to_world = np.asarray(self.camera_to_world_g, dtype=np.float64)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Gaussian alignment RGB must have shape HxWx3.")
        if depth.shape != rgb.shape[:2] or alpha.shape != rgb.shape[:2]:
            raise ValueError("Gaussian alignment depth/alpha shape differs from RGB.")
        if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
            raise ValueError("Gaussian alignment K must be finite shape 3x3.")
        if camera_to_world.shape != (4, 4) or not np.isfinite(camera_to_world).all():
            raise ValueError("Gaussian alignment camera_to_world must be finite shape 4x4.")
        if not np.allclose(camera_to_world[3], [0, 0, 0, 1], atol=1e-8):
            raise ValueError("Gaussian alignment camera_to_world must be homogeneous.")
        if abs(float(np.linalg.det(camera_to_world[:3, :3])) - 1.0) > 1e-5:
            raise ValueError("Gaussian alignment camera rotation must be proper.")
        if not math.isfinite(float(np.linalg.det(intrinsic))) or abs(float(np.linalg.det(intrinsic))) < 1e-12:
            raise ValueError("Gaussian alignment K must be invertible.")
        if not np.isfinite(depth).all() or np.any(depth < 0):
            raise ValueError("Gaussian expected depth must be finite and non-negative.")
        if not np.isfinite(alpha).all() or np.any(alpha < -1e-5) or np.any(alpha > 1.0001):
            raise ValueError("Gaussian alpha must be finite and lie in [0,1].")
        if not isinstance(self.camera_name, str) or not self.camera_name:
            raise ValueError("Gaussian alignment camera_name cannot be empty.")
        if not isinstance(self.metadata, dict):
            raise ValueError("Gaussian alignment metadata must be a dictionary.")
        object.__setattr__(self, "rgb_uint8", rgb)
        object.__setattr__(self, "expected_depth_g", depth)
        object.__setattr__(self, "alpha", np.clip(alpha, 0.0, 1.0))
        object.__setattr__(self, "intrinsic_matrix", intrinsic)
        object.__setattr__(self, "camera_to_world_g", camera_to_world)

    @property
    def height(self) -> int:
        return int(self.rgb_uint8.shape[0])

    @property
    def width(self) -> int:
        return int(self.rgb_uint8.shape[1])

    def unproject_pixel(
        self,
        pixel_xy: tuple[int, int] | list[int] | np.ndarray,
        *,
        patch_radius: int = 2,
        minimum_alpha: float = 0.05,
    ) -> tuple[np.ndarray, float, float]:
        """Return world point, robust local depth, and local median alpha."""

        pixel = np.asarray(pixel_xy, dtype=np.int64)
        if pixel.shape != (2,):
            raise ValueError("pixel_xy must contain integer X,Y.")
        x, y = int(pixel[0]), int(pixel[1])
        if not 0 <= x < self.width or not 0 <= y < self.height:
            raise ValueError(f"Pixel {(x, y)} lies outside the rendered view.")
        if patch_radius < 0:
            raise ValueError("patch_radius must be non-negative.")
        if not math.isfinite(minimum_alpha) or not 0 <= minimum_alpha <= 1:
            raise ValueError("minimum_alpha must lie in [0,1].")
        x0, x1 = max(0, x - patch_radius), min(self.width, x + patch_radius + 1)
        y0, y1 = max(0, y - patch_radius), min(self.height, y + patch_radius + 1)
        local_depth = self.expected_depth_g[y0:y1, x0:x1]
        local_alpha = self.alpha[y0:y1, x0:x1]
        keep = (local_alpha >= minimum_alpha) & (local_depth > 0.0)
        if not np.any(keep):
            raise ValueError(
                f"Pixel {(x, y)} has no visible Gaussian depth with alpha >= {minimum_alpha}."
            )
        depth = float(np.median(local_depth[keep]))
        alpha = float(np.median(local_alpha[keep]))
        camera_ray = np.linalg.inv(self.intrinsic_matrix) @ np.array(
            [float(x), float(y), 1.0],
            dtype=np.float64,
        )
        point_camera = camera_ray * depth
        point_world = (
            self.camera_to_world_g[:3, :3] @ point_camera
            + self.camera_to_world_g[:3, 3]
        )
        return point_world, depth, alpha

    def intersect_pixel_with_plane(
        self,
        pixel_xy: tuple[int, int] | list[int] | np.ndarray,
        *,
        plane_normal_g: Any,
        plane_offset_g: float,
    ) -> np.ndarray:
        """Intersect a camera pixel ray with a world-space plane."""

        pixel = np.asarray(pixel_xy, dtype=np.int64)
        if pixel.shape != (2,):
            raise ValueError("pixel_xy must contain integer X,Y.")
        x, y = int(pixel[0]), int(pixel[1])
        if not 0 <= x < self.width or not 0 <= y < self.height:
            raise ValueError(f"Pixel {(x, y)} lies outside the rendered view.")
        normal = np.asarray(plane_normal_g, dtype=np.float64)
        if normal.shape != (3,) or not np.isfinite(normal).all():
            raise ValueError("plane_normal_g must be a finite XYZ vector.")
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 1e-12:
            raise ValueError("plane_normal_g must be non-zero.")
        normal = normal / normal_norm
        offset = float(plane_offset_g) / normal_norm
        if not math.isfinite(offset):
            raise ValueError("plane_offset_g must be finite.")

        camera_ray = np.linalg.inv(self.intrinsic_matrix) @ np.array(
            [float(x), float(y), 1.0],
            dtype=np.float64,
        )
        origin = self.camera_to_world_g[:3, 3]
        direction = self.camera_to_world_g[:3, :3] @ camera_ray
        denominator = float(np.dot(normal, direction))
        if abs(denominator) <= 1e-12:
            raise ValueError(f"Pixel {(x, y)} ray is parallel to the plane.")
        ray_distance = -float(np.dot(normal, origin) + offset) / denominator
        if ray_distance <= 0.0:
            raise ValueError(f"Pixel {(x, y)} plane intersection is behind camera.")
        return origin + ray_distance * direction

    def save(self, path: str | Path) -> None:
        output = Path(path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
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
                    rgb_uint8=self.rgb_uint8,
                    expected_depth_g=self.expected_depth_g,
                    alpha=self.alpha,
                    intrinsic_matrix=self.intrinsic_matrix,
                    camera_to_world_g=self.camera_to_world_g,
                    camera_name=np.asarray(self.camera_name),
                    metadata_json=np.asarray(
                        json.dumps(
                            self.metadata,
                            ensure_ascii=False,
                            allow_nan=False,
                            sort_keys=True,
                        )
                    ),
                )
                file.flush()
                os.fsync(file.fileno())
                temporary = Path(file.name)
            os.replace(temporary, output)
            temporary = None
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    @classmethod
    def load(cls, path: str | Path) -> "GaussianAlignmentView":
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Gaussian alignment view not found: {source}")
        with np.load(source, allow_pickle=False) as payload:
            if int(payload["schema_version"].item()) != 1:
                raise ValueError("Unsupported Gaussian alignment view schema.")
            return cls(
                rgb_uint8=payload["rgb_uint8"].copy(),
                expected_depth_g=payload["expected_depth_g"].copy(),
                alpha=payload["alpha"].copy(),
                intrinsic_matrix=payload["intrinsic_matrix"].copy(),
                camera_to_world_g=payload["camera_to_world_g"].copy(),
                camera_name=str(payload["camera_name"].item()),
                metadata=json.loads(str(payload["metadata_json"].item())),
            )
