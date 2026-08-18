"""Small read-only COLMAP binary camera loader for 3DGS preview views."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
from typing import BinaryIO

import numpy as np


_CAMERA_MODELS: dict[int, tuple[str, int]] = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


def _read_exact(file: BinaryIO, count: int) -> bytes:
    data = file.read(count)
    if len(data) != count:
        raise ValueError("Unexpected end of COLMAP binary file.")
    return data


def _unpack(file: BinaryIO, format_string: str) -> tuple[object, ...]:
    size = struct.calcsize("<" + format_string)
    return struct.unpack("<" + format_string, _read_exact(file, size))


def quaternion_wxyz_to_rotation(qvec: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(qvec, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("COLMAP quaternion must be finite wxyz shape (4,).")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("COLMAP quaternion must be non-zero.")
    w, x, y, z = quaternion / norm
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class ColmapCamera:
    camera_id: int
    model: str
    width: int
    height: int
    parameters: np.ndarray

    def __post_init__(self) -> None:
        parameters = np.asarray(self.parameters, dtype=np.float64)
        if self.camera_id <= 0:
            raise ValueError("COLMAP camera ID must be positive.")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("COLMAP image dimensions must be positive.")
        if not np.isfinite(parameters).all():
            raise ValueError("COLMAP camera parameters must be finite.")
        object.__setattr__(self, "parameters", parameters)

    def intrinsic_matrix(self, *, width: int, height: int) -> np.ndarray:
        if width <= 0 or height <= 0:
            raise ValueError("Target image dimensions must be positive.")
        if self.model == "PINHOLE" and len(self.parameters) == 4:
            fx, fy, cx, cy = self.parameters
        elif self.model == "SIMPLE_PINHOLE" and len(self.parameters) == 3:
            focal, cx, cy = self.parameters
            fx = fy = focal
        else:
            raise ValueError(
                f"Gaussian alignment views require an undistorted PINHOLE camera; got {self.model}."
            )
        scale_x = width / self.width
        scale_y = height / self.height
        return np.array(
            [
                [fx * scale_x, 0.0, cx * scale_x],
                [0.0, fy * scale_y, cy * scale_y],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class ColmapImage:
    image_id: int
    camera_id: int
    name: str
    quaternion_wxyz: np.ndarray
    translation: np.ndarray

    def __post_init__(self) -> None:
        quaternion = np.asarray(self.quaternion_wxyz, dtype=np.float64)
        translation = np.asarray(self.translation, dtype=np.float64)
        if self.image_id <= 0 or self.camera_id <= 0:
            raise ValueError("COLMAP image and camera IDs must be positive.")
        if not self.name:
            raise ValueError("COLMAP image name cannot be empty.")
        if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
            raise ValueError("COLMAP image quaternion must be finite shape (4,).")
        if translation.shape != (3,) or not np.isfinite(translation).all():
            raise ValueError("COLMAP image translation must be finite shape (3,).")
        object.__setattr__(self, "quaternion_wxyz", quaternion)
        object.__setattr__(self, "translation", translation)

    @property
    def world_to_camera(self) -> np.ndarray:
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = quaternion_wxyz_to_rotation(self.quaternion_wxyz)
        transform[:3, 3] = self.translation
        return transform

    @property
    def camera_to_world(self) -> np.ndarray:
        rotation = quaternion_wxyz_to_rotation(self.quaternion_wxyz)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation.T
        transform[:3, 3] = -rotation.T @ self.translation
        return transform


def read_cameras_binary(path: str | Path) -> dict[int, ColmapCamera]:
    source = Path(path).expanduser().resolve()
    cameras: dict[int, ColmapCamera] = {}
    with source.open("rb") as file:
        count = int(_unpack(file, "Q")[0])
        for _ in range(count):
            camera_id, model_id, width, height = _unpack(file, "iiQQ")
            if int(model_id) not in _CAMERA_MODELS:
                raise ValueError(f"Unsupported COLMAP camera model ID {model_id}.")
            model, parameter_count = _CAMERA_MODELS[int(model_id)]
            parameters = np.asarray(
                _unpack(file, "d" * parameter_count),
                dtype=np.float64,
            )
            camera = ColmapCamera(
                camera_id=int(camera_id),
                model=model,
                width=int(width),
                height=int(height),
                parameters=parameters,
            )
            if camera.camera_id in cameras:
                raise ValueError(f"Duplicate COLMAP camera ID {camera.camera_id}.")
            cameras[camera.camera_id] = camera
    if not cameras:
        raise ValueError(f"No COLMAP cameras found in {source}.")
    return cameras


def read_images_binary(path: str | Path) -> list[ColmapImage]:
    source = Path(path).expanduser().resolve()
    images: list[ColmapImage] = []
    with source.open("rb") as file:
        count = int(_unpack(file, "Q")[0])
        for _ in range(count):
            properties = _unpack(file, "idddddddi")
            image_id = int(properties[0])
            quaternion = np.asarray(properties[1:5], dtype=np.float64)
            translation = np.asarray(properties[5:8], dtype=np.float64)
            camera_id = int(properties[8])
            name_bytes = bytearray()
            while True:
                character = _read_exact(file, 1)
                if character == b"\x00":
                    break
                name_bytes.extend(character)
            point_count = int(_unpack(file, "Q")[0])
            file.seek(24 * point_count, 1)
            images.append(
                ColmapImage(
                    image_id=image_id,
                    camera_id=camera_id,
                    name=name_bytes.decode("utf-8"),
                    quaternion_wxyz=quaternion,
                    translation=translation,
                )
            )
    if not images:
        raise ValueError(f"No COLMAP images found in {source}.")
    return sorted(images, key=lambda image: image.name)


def load_sparse_cameras(
    sparse_directory: str | Path,
) -> tuple[dict[int, ColmapCamera], list[ColmapImage]]:
    root = Path(sparse_directory).expanduser().resolve()
    cameras = read_cameras_binary(root / "cameras.bin")
    images = read_images_binary(root / "images.bin")
    unknown = sorted({image.camera_id for image in images} - set(cameras))
    if unknown:
        raise ValueError(f"COLMAP images reference unknown camera IDs: {unknown}")
    return cameras, images
