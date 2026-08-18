"""Robust local depth sampling around image-space keypoints."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DepthSample:
    depth_m: float
    confidence: float
    valid_count: int
    total_count: int
    median_absolute_deviation_m: float
    used_nearest_cluster: bool


def _select_nearest_supported_cluster(
    values: np.ndarray,
    edge_gap_m: float,
    min_cluster_fraction: float,
    expected_depth_m: float | None = None,
    max_expected_depth_delta_m: float = 0.45,
) -> tuple[np.ndarray | None, bool]:
    """Choose a supported local depth mode, optionally using track history.

    The legacy behavior selects the nearest supported mode.  When an expected
    depth is supplied, the supported mode closest to that depth is selected.
    A mode too far from a valid expectation is rejected so a fully occluding
    foreground surface cannot silently replace the tracked person's depth.
    """
    sorted_values = np.sort(values)
    gap_indices = np.flatnonzero(np.diff(sorted_values) >= edge_gap_m)
    if gap_indices.size == 0:
        if (
            expected_depth_m is not None
            and abs(float(np.median(sorted_values)) - expected_depth_m)
            > max_expected_depth_delta_m
        ):
            return None, False
        return sorted_values, False

    clusters = np.split(sorted_values, gap_indices + 1)
    minimum_size = max(
        3, int(np.ceil(sorted_values.size * min_cluster_fraction))
    )
    supported = [cluster for cluster in clusters if cluster.size >= minimum_size]
    if expected_depth_m is None:
        if supported:
            return supported[0], True
        return sorted_values, False

    candidates = supported or [sorted_values]
    selected = min(
        candidates,
        key=lambda cluster: abs(float(np.median(cluster)) - expected_depth_m),
    )
    if (
        abs(float(np.median(selected)) - expected_depth_m)
        > max_expected_depth_delta_m
    ):
        return None, bool(supported)
    return selected, bool(supported)


def sample_joint_depth(
    depth_m: np.ndarray,
    u: float,
    v: float,
    radius: int = 3,
    min_depth_m: float = 0.2,
    max_depth_m: float = 8.0,
    max_mad_m: float = 0.08,
    depth_edge_gap_m: float = 0.15,
    min_cluster_fraction: float = 0.2,
    expected_depth_m: float | None = None,
    max_expected_depth_delta_m: float = 0.45,
) -> DepthSample | None:
    """Sample a keypoint's depth using a robust local depth mode.

    ``expected_depth_m`` is intentionally optional so all existing callers
    retain nearest-cluster behavior.  Multi-person tracking can pass the
    previous depth of the same joint to keep overlapping people on separate
    depth layers without constructing a point cloud.
    """
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Expected HxW depth, got {depth.shape}.")
    if radius < 0:
        raise ValueError("radius must be non-negative.")
    if max_expected_depth_delta_m <= 0:
        raise ValueError("max_expected_depth_delta_m must be positive.")
    if expected_depth_m is not None:
        expected_depth_m = float(expected_depth_m)
        if not np.isfinite(expected_depth_m):
            raise ValueError("expected_depth_m must be finite or None.")

    center_u = int(round(float(u)))
    center_v = int(round(float(v)))
    height, width = depth.shape
    if center_u < 0 or center_u >= width or center_v < 0 or center_v >= height:
        return None

    x1 = max(0, center_u - radius)
    x2 = min(width, center_u + radius + 1)
    y1 = max(0, center_v - radius)
    y2 = min(height, center_v + radius + 1)
    window = depth[y1:y2, x1:x2]
    valid_mask = (
        np.isfinite(window)
        & (window >= min_depth_m)
        & (window <= max_depth_m)
    )
    values = window[valid_mask]
    if values.size == 0:
        return None

    selected_values, used_nearest_cluster = _select_nearest_supported_cluster(
        values,
        edge_gap_m=depth_edge_gap_m,
        min_cluster_fraction=min_cluster_fraction,
        expected_depth_m=expected_depth_m,
        max_expected_depth_delta_m=max_expected_depth_delta_m,
    )
    if selected_values is None:
        return None
    median = float(np.median(selected_values))
    mad = float(np.median(np.abs(selected_values - median)))
    valid_ratio = float(selected_values.size / window.size)
    consistency = max(0.0, 1.0 - mad / max_mad_m)
    confidence = float(np.clip(valid_ratio * consistency, 0.0, 1.0))
    return DepthSample(
        depth_m=median,
        confidence=confidence,
        valid_count=int(selected_values.size),
        total_count=int(window.size),
        median_absolute_deviation_m=mad,
        used_nearest_cluster=used_nearest_cluster,
    )
