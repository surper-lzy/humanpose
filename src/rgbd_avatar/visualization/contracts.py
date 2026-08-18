"""Validated external-data contracts used by the 3D sequence viewer."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from rgbd_avatar.depth import GroundPlaneEstimate
from rgbd_avatar.io import load_json_mapping


LOGGER = logging.getLogger(__name__)


def verify_manifest_camera(
    jsonl_path: Path,
    camera: dict[str, Any],
    *,
    allow_mismatch: bool,
) -> None:
    """Reject replay with calibration different from the processing run."""

    manifest_path = jsonl_path.parent / "manifest.json"
    if not manifest_path.is_file():
        LOGGER.warning(
            "No manifest.json beside %s; camera calibration cannot be verified.",
            jsonl_path,
        )
        return
    manifest = load_json_mapping(manifest_path)
    snapshot = manifest.get("camera")
    if not isinstance(snapshot, dict):
        LOGGER.warning(
            "%s predates camera snapshots. Use the same camera YAML that "
            "created this result.",
            manifest_path,
        )
        return

    differences: list[str] = []
    for key in ("rgb_width", "rgb_height", "depth_width", "depth_height"):
        if int(snapshot.get(key, -1)) != int(camera[key]):
            differences.append(
                f"{key}: manifest={snapshot.get(key)!r}, yaml={camera[key]!r}"
            )
    for key in ("depth_scale", "min_depth_m", "max_depth_m"):
        recorded = float(snapshot.get(key, np.nan))
        current = float(camera[key])
        if not np.isclose(recorded, current, rtol=1e-9, atol=1e-12):
            differences.append(f"{key}: manifest={recorded!r}, yaml={current!r}")
    for key in ("align_depth_to_rgb", "images_undistorted"):
        if bool(snapshot.get(key)) != bool(camera[key]):
            differences.append(
                f"{key}: manifest={snapshot.get(key)!r}, yaml={camera[key]!r}"
            )
    recorded_intrinsics = snapshot.get("intrinsics", {})
    for key in ("fx", "fy", "cx", "cy"):
        recorded = float(recorded_intrinsics.get(key, np.nan))
        current = float(camera["intrinsics"][key])
        if not np.isclose(recorded, current, rtol=1e-9, atol=1e-9):
            differences.append(
                f"intrinsics.{key}: manifest={recorded!r}, yaml={current!r}"
            )

    if not differences:
        LOGGER.info("Verified camera calibration against %s.", manifest_path)
        return
    message = (
        "Camera configuration differs from the processing manifest: "
        + "; ".join(differences)
    )
    if not allow_mismatch:
        raise ValueError(
            message
            + ". Pass --allow-camera-config-mismatch only for an intentional "
            "diagnostic."
        )
    LOGGER.warning("%s", message)


def load_ground_alignment(
    jsonl_path: Path,
    *,
    ground_plane_path: Path | None,
    disabled: bool,
) -> tuple[GroundPlaneEstimate | None, np.ndarray | None]:
    """Load the calibrated camera-to-ground transform when available."""

    if disabled:
        LOGGER.info("Ground alignment disabled by command line.")
        return None, None
    explicit = ground_plane_path is not None
    path = (
        ground_plane_path.expanduser().resolve()
        if ground_plane_path is not None
        else jsonl_path.parent / "ground_plane.json"
    )
    if not path.is_file():
        if explicit:
            raise FileNotFoundError(f"Ground calibration not found: {path}")
        LOGGER.warning(
            "No ground_plane.json beside %s; using optical camera axes.",
            jsonl_path,
        )
        return None, None
    estimate = GroundPlaneEstimate.from_mapping(load_json_mapping(path))
    transform = estimate.camera_to_ground_transform()
    LOGGER.info(
        "Loaded ground alignment from %s: camera_height=%.4f m "
        "tilt=%.2f deg residual_p95=%.4f m.",
        path,
        estimate.camera_height_m,
        estimate.tilt_from_camera_up_deg,
        estimate.residual_p95_m,
    )
    return estimate, transform


__all__ = ["load_ground_alignment", "verify_manifest_camera"]
