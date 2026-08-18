"""Depth-image loading with explicit metric conversion."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_depth_m(path: str | Path, depth_scale: float) -> np.ndarray:
    """Read a single-channel depth image and convert it to float32 meters."""
    if depth_scale <= 0:
        raise ValueError("depth_scale must be positive.")

    depth_path = Path(path).expanduser().resolve()
    if not depth_path.is_file():
        raise FileNotFoundError(f"Depth image not found: {depth_path}")

    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise RuntimeError(f"OpenCV failed to read depth image: {depth_path}")
    if depth_raw.ndim != 2:
        raise ValueError(
            f"Expected a single-channel depth image, got {depth_raw.shape}."
        )
    if not np.issubdtype(depth_raw.dtype, np.integer):
        raise TypeError(
            f"Expected integer raw depth values, got {depth_raw.dtype}."
        )

    return depth_raw.astype(np.float32) * np.float32(depth_scale)
