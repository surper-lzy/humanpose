"""MMPose-backed RTMPose-M inference."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Literal

import numpy as np

from .models import Pose2D

Device = Literal["auto", "cpu"] | str


@dataclass(frozen=True)
class RTMPoseBackendConfig:
    model_config: Path
    model_checkpoint: Path
    detector: str = "auto"
    model_cache_dir: Path | None = None
    device: Device = "auto"
    bbox_threshold: float = 0.3
    keypoint_threshold: float = 0.3
    min_valid_keypoints: int = 10
    min_mean_keypoint_score: float = 0.3


def resolve_device(device: Device) -> str:
    if device != "auto":
        return device

    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


class RTMPoseBackend:
    """Load RTMPose once and infer Halpe26 poses from RGB images."""

    def __init__(self, config: RTMPoseBackendConfig) -> None:
        config_path = Path(config.model_config).expanduser().resolve()
        checkpoint_path = Path(config.model_checkpoint).expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"RTMPose config not found: {config_path}")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"RTMPose checkpoint not found: {checkpoint_path}"
            )

        if config.model_cache_dir is not None:
            cache_dir = Path(config.model_cache_dir).expanduser().resolve()
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("TORCH_HOME", str(cache_dir))

        from mmpose.apis import MMPoseInferencer
        from mmpose.utils import register_all_modules

        register_all_modules()
        self.device = resolve_device(config.device)
        self.detector = config.detector
        self.bbox_threshold = float(config.bbox_threshold)
        self.keypoint_threshold = float(config.keypoint_threshold)
        self.min_valid_keypoints = int(config.min_valid_keypoints)
        self.min_mean_keypoint_score = float(
            config.min_mean_keypoint_score
        )
        detector_model = None if config.detector == "auto" else config.detector
        self._inferencer = MMPoseInferencer(
            pose2d=str(config_path),
            pose2d_weights=str(checkpoint_path),
            det_model=detector_model,
            device=self.device,
        )

    def infer(self, image: np.ndarray | str | Path) -> list[Pose2D]:
        image_input = str(image) if isinstance(image, Path) else image
        result = next(
            self._inferencer(
                image_input,
                return_vis=False,
                draw_bbox=False,
                bbox_thr=self.bbox_threshold,
            )
        )

        prediction_batches = result.get("predictions", [])
        if not prediction_batches:
            return []

        poses: list[Pose2D] = []
        for prediction in prediction_batches[0]:
            keypoints = np.asarray(prediction["keypoints"], dtype=np.float32)
            scores = np.asarray(
                prediction["keypoint_scores"], dtype=np.float32
            )
            bbox = np.asarray(prediction["bbox"], dtype=np.float32).reshape(
                -1, 4
            )[0]
            bbox_score = float(
                np.asarray(prediction["bbox_score"]).reshape(-1)[0]
            )
            pose = Pose2D(
                keypoints=keypoints,
                scores=scores,
                bbox_xyxy=bbox,
                bbox_score=bbox_score,
            )

            # Pose2DInferencer falls back to a full-image box with score 1.0
            # when its detector finds no person. That fallback is useful for
            # cropped-image demos but creates false poses on empty RGB frames.
            is_full_image_fallback = (
                self.detector == "auto"
                and bbox_score == 1.0
                and np.allclose(bbox[:2], 0.0)
            )
            valid_count = int(
                np.count_nonzero(scores >= self.keypoint_threshold)
            )
            passes_quality_gate = (
                valid_count >= self.min_valid_keypoints
                and pose.mean_score >= self.min_mean_keypoint_score
            )
            if not is_full_image_fallback and passes_quality_gate:
                poses.append(pose)

        poses.sort(key=lambda pose: pose.bbox_score, reverse=True)
        return poses
