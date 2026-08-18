"""Run the multi-person RGB-D pipeline with isolated TensorRT FP16 engines."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

from rgbd_avatar.pose import (
    TensorRTHalpe26Backend,
    TensorRTHalpe26BackendConfig,
)

from . import live_multi_person


LOGGER = logging.getLogger("view_live_multi_person_tensorrt")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ENGINE_ROOT = PROJECT_ROOT / "outputs/tensorrt_fp16/engines"


def _environment_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _environment_positive_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _engine_path(environment_name: str, default_name: str) -> Path:
    configured = os.environ.get(environment_name)
    return (
        Path(configured).expanduser().resolve()
        if configured
        else (DEFAULT_ENGINE_ROOT / default_name).resolve()
    )


def _build_tensorrt_backend(
    pose_config: dict[str, Any],
    args: argparse.Namespace,
) -> TensorRTHalpe26Backend:
    detector = args.detector or pose_config.get("detector", "auto")
    if detector != "auto":
        raise ValueError(
            "The TensorRT multi-person entry requires --detector auto; "
            "whole_image cannot detect multiple people."
        )
    requested_device = args.device or pose_config.get("device", "auto")
    if requested_device not in ("auto", "cuda", "cuda:0"):
        raise ValueError(
            "The TensorRT entry only supports CUDA device 0, got "
            f"{requested_device!r}."
        )

    detector_engine = _engine_path(
        "HUMANPOSE_TRT_DETECTOR_ENGINE",
        "rtmdet_m_person_640_fp16.engine",
    )
    pose_engine = _engine_path(
        "HUMANPOSE_TRT_POSE_ENGINE",
        "rtmpose_m_halpe26_256x192_fp16.engine",
    )
    LOGGER.info("TensorRT detector engine: %s", detector_engine)
    LOGGER.info("TensorRT pose engine: %s", pose_engine)
    return TensorRTHalpe26Backend(
        TensorRTHalpe26BackendConfig(
            detector_engine=detector_engine,
            pose_engine=pose_engine,
            max_persons=int(args.max_persons),
            bbox_threshold=float(pose_config["bbox_threshold"]),
            keypoint_threshold=float(pose_config["keypoint_threshold"]),
            min_valid_keypoints=int(pose_config["min_valid_keypoints"]),
            min_mean_keypoint_score=float(
                pose_config["min_mean_keypoint_score"]
            ),
            detector_interval=_environment_positive_int(
                "HUMANPOSE_TRT_DETECTOR_INTERVAL",
                1,
            ),
            profile_timings=_environment_flag("HUMANPOSE_TRT_PROFILE"),
        )
    )


def main() -> int:
    detector_interval = _environment_positive_int(
        "HUMANPOSE_TRT_DETECTOR_INTERVAL",
        1,
    )
    depth_connected_interval = _environment_positive_int(
        "HUMANPOSE_TRT_DEPTH_CONNECTED_INTERVAL",
        1,
    )
    LOGGER.info(
        "TensorRT cadence: detector_interval=%d "
        "depth_connected_interval=%d",
        detector_interval,
        depth_connected_interval,
    )
    return live_multi_person.main(
        rtmpose_backend_builder=_build_tensorrt_backend,
        rtmpose_backend_description="TensorRT FP16 RTMDet + RTMPose",
        depth_connected_refresh_interval=depth_connected_interval,
    )
