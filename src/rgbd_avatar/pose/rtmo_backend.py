"""Experimental one-stage RTMO multi-person pose backend."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np

from .models import Pose2D
from .rtmpose_backend import Device, resolve_device


RTMO_TINY_MODEL = "rtmo-t_8xb32-600e_body7-416x416"
_COCO17_COUNT = 17
_HALPE26_COUNT = 26


@dataclass(frozen=True)
class RTMOBackendConfig:
    """Configuration for the opt-in MMPose RTMO experiment."""

    model: str = RTMO_TINY_MODEL
    model_checkpoint: Path | str | None = None
    model_cache_dir: Path | None = None
    device: Device = "auto"
    bbox_threshold: float = 0.3
    nms_threshold: float = 0.65
    keypoint_threshold: float = 0.3
    min_valid_keypoints: int = 10
    min_mean_keypoint_score: float = 0.3

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("RTMO model must not be empty.")
        if not 0 <= self.bbox_threshold <= 1:
            raise ValueError("bbox_threshold must be in [0, 1].")
        if not 0 <= self.nms_threshold <= 1:
            raise ValueError("nms_threshold must be in [0, 1].")
        if not 0 <= self.keypoint_threshold <= 1:
            raise ValueError("keypoint_threshold must be in [0, 1].")
        if not 1 <= self.min_valid_keypoints <= _COCO17_COUNT:
            raise ValueError("min_valid_keypoints must be in [1, 17].")
        if not 0 <= self.min_mean_keypoint_score <= 1:
            raise ValueError("min_mean_keypoint_score must be in [0, 1].")


def coco17_to_halpe26(
    keypoints: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map COCO17 into the fixed Halpe26 transport representation.

    COCO and Halpe use the same order for their first 17 body points. RTMO
    does not predict Halpe's head/neck/hip anchors or detailed foot points, so
    the three body anchors are derived conservatively and the six foot points
    remain unavailable with score zero.
    """

    source_points = np.asarray(keypoints, dtype=np.float32)
    source_scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if source_points.shape != (_COCO17_COUNT, 2):
        raise ValueError(
            "Expected COCO17 keypoints with shape (17, 2), got "
            f"{source_points.shape}."
        )
    if source_scores.shape != (_COCO17_COUNT,):
        raise ValueError(
            "Expected COCO17 scores with shape (17,), got "
            f"{source_scores.shape}."
        )

    target_points = np.zeros((_HALPE26_COUNT, 2), dtype=np.float32)
    target_scores = np.zeros(_HALPE26_COUNT, dtype=np.float32)
    target_points[:_COCO17_COUNT] = source_points
    target_scores[:_COCO17_COUNT] = source_scores

    # A duplicated nose is a more stable depth sample than extrapolating the
    # top of the head when a person is small or partly outside the image.
    target_points[17] = source_points[0]
    target_scores[17] = source_scores[0]

    target_points[18] = 0.5 * (source_points[5] + source_points[6])
    target_scores[18] = min(source_scores[5], source_scores[6])
    target_points[19] = 0.5 * (source_points[11] + source_points[12])
    target_scores[19] = min(source_scores[11], source_scores[12])
    return target_points, target_scores


def _bbox_from_keypoints(
    keypoints: np.ndarray,
    scores: np.ndarray,
    *,
    image_shape: tuple[int, int],
    score_threshold: float,
) -> np.ndarray | None:
    visible = (
        np.isfinite(keypoints).all(axis=1)
        & np.isfinite(scores)
        & (scores >= score_threshold)
    )
    if not np.any(visible):
        return None
    points = keypoints[visible]
    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    size = np.maximum(maximum - minimum, 1.0)
    padding = np.maximum(0.10 * size, 2.0)
    height, width = image_shape
    return np.asarray(
        [
            np.clip(minimum[0] - padding[0], 0.0, width - 1.0),
            np.clip(minimum[1] - padding[1], 0.0, height - 1.0),
            np.clip(maximum[0] + padding[0], 0.0, width - 1.0),
            np.clip(maximum[1] + padding[1], 0.0, height - 1.0),
        ],
        dtype=np.float32,
    )


def _prediction_bbox(
    prediction: dict,
    keypoints: np.ndarray,
    scores: np.ndarray,
    *,
    image_shape: tuple[int, int],
    score_threshold: float,
) -> np.ndarray | None:
    raw_bbox = prediction.get("bbox")
    if raw_bbox is not None:
        bbox = np.asarray(raw_bbox, dtype=np.float32).reshape(-1, 4)[0]
        if np.isfinite(bbox).all() and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
            return bbox
    return _bbox_from_keypoints(
        keypoints,
        scores,
        image_shape=image_shape,
        score_threshold=score_threshold,
    )


class RTMOBackend:
    """Load RTMO once and expose one-stage multi-person Halpe26 poses."""

    def __init__(self, config: RTMOBackendConfig) -> None:
        if config.model_cache_dir is not None:
            cache_dir = Path(config.model_cache_dir).expanduser().resolve()
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("TORCH_HOME", str(cache_dir))

        model = str(config.model)
        model_path = Path(model).expanduser()
        if model_path.suffix == ".py":
            model_path = model_path.resolve()
            if not model_path.is_file():
                raise FileNotFoundError(f"RTMO config not found: {model_path}")
            model = str(model_path)

        checkpoint: str | None = None
        if config.model_checkpoint is not None:
            checkpoint = str(config.model_checkpoint)
            if not checkpoint.startswith(("http://", "https://")):
                checkpoint_path = Path(checkpoint).expanduser().resolve()
                if not checkpoint_path.is_file():
                    raise FileNotFoundError(
                        f"RTMO checkpoint not found: {checkpoint_path}"
                    )
                checkpoint = str(checkpoint_path)

        from mmpose.apis import MMPoseInferencer
        from mmpose.utils import register_all_modules

        register_all_modules()
        self.device = resolve_device(config.device)
        self.model_name = model
        self.bbox_threshold = float(config.bbox_threshold)
        self.nms_threshold = float(config.nms_threshold)
        self.keypoint_threshold = float(config.keypoint_threshold)
        self.min_valid_keypoints = int(config.min_valid_keypoints)
        self.min_mean_keypoint_score = float(
            config.min_mean_keypoint_score
        )
        # A bottom-up RTMO config bypasses MMPose's detector initialization.
        # If checkpoint is None, the official weight URL is resolved from the
        # installed MMPose model index and cached under TORCH_HOME.
        self._inferencer = MMPoseInferencer(
            pose2d=model,
            pose2d_weights=checkpoint,
            det_model=None,
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
                nms_thr=self.nms_threshold,
            )
        )
        prediction_batches = result.get("predictions", [])
        if not prediction_batches:
            return []

        if isinstance(image, np.ndarray):
            image_shape = (int(image.shape[0]), int(image.shape[1]))
        else:
            # Official RTMO predictions normally include a bbox. This fallback
            # only affects direct path-based use with malformed predictions.
            image_shape = (2**30, 2**30)

        poses: list[Pose2D] = []
        for prediction in prediction_batches[0]:
            keypoints = np.asarray(
                prediction["keypoints"], dtype=np.float32
            ).reshape(-1, 2)
            scores = np.asarray(
                prediction["keypoint_scores"], dtype=np.float32
            ).reshape(-1)
            if keypoints.shape != (_COCO17_COUNT, 2) or scores.shape != (
                _COCO17_COUNT,
            ):
                raise ValueError(
                    "The experimental RTMO backend requires a COCO17 "
                    f"checkpoint, got {len(scores)} keypoints."
                )

            finite_scores = scores[np.isfinite(scores)]
            valid_count = int(
                np.count_nonzero(scores >= self.keypoint_threshold)
            )
            mean_score = (
                float(np.mean(finite_scores)) if finite_scores.size else 0.0
            )
            if (
                valid_count < self.min_valid_keypoints
                or mean_score < self.min_mean_keypoint_score
            ):
                continue

            bbox = _prediction_bbox(
                prediction,
                keypoints,
                scores,
                image_shape=image_shape,
                score_threshold=self.keypoint_threshold,
            )
            if bbox is None:
                continue
            raw_bbox_score = prediction.get("bbox_score")
            bbox_score = (
                mean_score
                if raw_bbox_score is None
                else float(np.asarray(raw_bbox_score).reshape(-1)[0])
            )
            if not np.isfinite(bbox_score) or bbox_score < self.bbox_threshold:
                continue

            halpe_keypoints, halpe_scores = coco17_to_halpe26(
                keypoints,
                scores,
            )
            poses.append(
                Pose2D(
                    keypoints=halpe_keypoints,
                    scores=halpe_scores,
                    bbox_xyxy=bbox,
                    bbox_score=bbox_score,
                )
            )

        poses.sort(key=lambda pose: pose.bbox_score, reverse=True)
        return poses
