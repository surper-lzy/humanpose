"""Device-neutral contracts for live RGB-D capture and pose transport."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Protocol, runtime_checkable

import numpy as np

from rgbd_avatar.camera import CameraIntrinsics
from rgbd_avatar.pose import HALPE26_NAMES


@dataclass(frozen=True)
class RGBDFrame:
    """One depth-to-color-aligned frame produced by any camera adapter."""

    rgb_bgr: np.ndarray
    depth_m: np.ndarray
    intrinsics: CameraIntrinsics
    timestamp_ns: int
    frame_number: int
    source_id: str

    def __post_init__(self) -> None:
        rgb = np.asarray(self.rgb_bgr, dtype=np.uint8)
        depth = np.asarray(self.depth_m, dtype=np.float32)
        expected_shape = (self.intrinsics.height, self.intrinsics.width)
        if rgb.shape != (*expected_shape, 3):
            raise ValueError(
                f"rgb_bgr must have shape {(*expected_shape, 3)}, got {rgb.shape}."
            )
        if depth.shape != expected_shape:
            raise ValueError(
                f"depth_m must have shape {expected_shape}, got {depth.shape}."
            )
        if np.any(np.isfinite(depth) & (depth < 0.0)):
            raise ValueError("depth_m cannot contain a negative finite depth.")
        if int(self.timestamp_ns) < 0 or int(self.frame_number) < 0:
            raise ValueError("Frame timestamp and number must be non-negative.")
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError("source_id must be a non-empty string.")
        object.__setattr__(self, "rgb_bgr", rgb)
        object.__setattr__(self, "depth_m", depth)
        object.__setattr__(self, "timestamp_ns", int(self.timestamp_ns))
        object.__setattr__(self, "frame_number", int(self.frame_number))


@runtime_checkable
class RGBDSource(Protocol):
    """Minimal plugin boundary implemented by each physical camera SDK."""

    @property
    def source_id(self) -> str: ...

    def start(self) -> None: ...

    def read(self, timeout_ms: int = 1000) -> RGBDFrame: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class LivePosePacket:
    """Small device-independent message sent from pose to 3DGS service."""

    frame_number: int
    timestamp_ns: int
    source_id: str
    joints_g: np.ndarray
    confidence: np.ndarray
    usable: np.ndarray
    mapping_mode: str

    def __post_init__(self) -> None:
        count = len(HALPE26_NAMES)
        joints = np.asarray(self.joints_g, dtype=np.float32)
        confidence = np.asarray(self.confidence, dtype=np.float32)
        usable = np.asarray(self.usable, dtype=bool)
        if joints.shape != (count, 3):
            raise ValueError(f"joints_g must have shape {(count, 3)}.")
        if confidence.shape != (count,) or usable.shape != (count,):
            raise ValueError("confidence and usable must have shape (26,).")
        if not np.isfinite(joints[usable]).all():
            raise ValueError("Every usable joint must have finite G coordinates.")
        if not np.isfinite(confidence).all() or np.any(confidence < 0.0):
            raise ValueError("confidence must be finite and non-negative.")
        if int(self.frame_number) < 0 or int(self.timestamp_ns) < 0:
            raise ValueError("Packet frame number and timestamp must be non-negative.")
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError("source_id must be a non-empty string.")
        if self.mapping_mode not in ("root_locked", "fixed_origin"):
            raise ValueError("mapping_mode must be root_locked or fixed_origin.")
        object.__setattr__(self, "frame_number", int(self.frame_number))
        object.__setattr__(self, "timestamp_ns", int(self.timestamp_ns))
        object.__setattr__(self, "joints_g", joints)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "usable", usable)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "keypoint_format": "halpe26",
            "coordinate_system": "3dgs_world",
            "frame_number": self.frame_number,
            "timestamp_ns": self.timestamp_ns,
            "source_id": self.source_id,
            "mapping_mode": self.mapping_mode,
            "confidence": self.confidence.tolist(),
            "usable": self.usable.tolist(),
            "joints_g": [
                self.joints_g[index].tolist() if self.usable[index] else None
                for index in range(len(HALPE26_NAMES))
            ],
        }

    def to_json_bytes(self) -> bytes:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "LivePosePacket":
        if int(payload.get("schema_version", -1)) != 1:
            raise ValueError("Unsupported live pose schema_version.")
        if payload.get("keypoint_format") != "halpe26":
            raise ValueError("Live pose packet keypoint format must be halpe26.")
        if payload.get("coordinate_system") != "3dgs_world":
            raise ValueError("Live pose packet must contain 3DGS-world joints.")
        usable = np.asarray(payload["usable"], dtype=bool)
        joints = np.full((len(HALPE26_NAMES), 3), np.nan, dtype=np.float32)
        joint_payload = payload["joints_g"]
        if not isinstance(joint_payload, list) or len(joint_payload) != len(joints):
            raise ValueError("joints_g must contain 26 entries.")
        for index in np.flatnonzero(usable):
            joints[index] = np.asarray(joint_payload[int(index)], dtype=np.float32)
        return cls(
            frame_number=int(payload["frame_number"]),
            timestamp_ns=int(payload["timestamp_ns"]),
            source_id=str(payload["source_id"]),
            joints_g=joints,
            confidence=np.asarray(payload["confidence"], dtype=np.float32),
            usable=usable,
            mapping_mode=str(payload["mapping_mode"]),
        )

    @classmethod
    def from_json_bytes(cls, payload: bytes) -> "LivePosePacket":
        decoded = json.loads(payload.decode("utf-8"))
        if not isinstance(decoded, dict):
            raise ValueError("Live pose JSON must contain an object.")
        return cls.from_mapping(decoded)
