"""Pinhole-camera deprojection for aligned depth images."""

from __future__ import annotations

import numpy as np

from rgbd_avatar.camera import CameraIntrinsics


def deproject_pixel(
    u: float,
    v: float,
    z_m: float,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """Convert an aligned pixel and metric depth to camera-frame XYZ."""
    if not np.isfinite(z_m) or z_m <= 0:
        raise ValueError(f"Depth must be finite and positive, got {z_m}.")

    x_m = (float(u) - intrinsics.cx) * float(z_m) / intrinsics.fx
    y_m = (float(v) - intrinsics.cy) * float(z_m) / intrinsics.fy
    return np.array([x_m, y_m, float(z_m)], dtype=np.float32)


def depth_to_organized_point_cloud(
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsics,
    min_depth_m: float = 0.2,
    max_depth_m: float = 8.0,
) -> np.ndarray:
    """Convert aligned depth to an HxWx3 metric point cloud.

    Invalid pixels are represented by NaN triplets so image-space indexing is
    preserved.
    """
    depth = np.asarray(depth_m, dtype=np.float32)
    expected_shape = (intrinsics.height, intrinsics.width)
    if depth.shape != expected_shape:
        raise ValueError(
            f"Expected depth shape {expected_shape}, got {depth.shape}."
        )
    if min_depth_m <= 0 or max_depth_m <= min_depth_m:
        raise ValueError("Invalid metric depth range.")

    rows, columns = np.indices(depth.shape, dtype=np.float32)
    valid = (
        np.isfinite(depth)
        & (depth >= min_depth_m)
        & (depth <= max_depth_m)
    )
    x = (columns - intrinsics.cx) * depth / intrinsics.fx
    y = (rows - intrinsics.cy) * depth / intrinsics.fy
    points = np.stack((x, y, depth), axis=-1).astype(np.float32)
    points[~valid] = np.nan
    return points
