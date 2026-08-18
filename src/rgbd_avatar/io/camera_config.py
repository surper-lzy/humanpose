"""Load the shared aligned RGB-D camera calibration contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rgbd_avatar.camera import CameraIntrinsics

from .serialization import load_yaml_mapping


def load_camera_config(
    path: str | Path,
) -> tuple[dict[str, Any], CameraIntrinsics]:
    resolved = Path(path).expanduser().resolve()
    payload = load_yaml_mapping(resolved)
    camera = payload.get("camera")
    if not isinstance(camera, dict):
        raise ValueError(f"Expected a camera mapping in {resolved}.")
    try:
        intrinsics_payload = camera["intrinsics"]
        width = int(camera["depth_width"])
        height = int(camera["depth_height"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Incomplete camera calibration in {resolved}.") from error
    if not isinstance(intrinsics_payload, dict):
        raise ValueError(f"Expected camera.intrinsics in {resolved}.")
    intrinsics = CameraIntrinsics(
        **intrinsics_payload,
        width=width,
        height=height,
    )
    return camera, intrinsics
