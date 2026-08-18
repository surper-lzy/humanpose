"""TensorRT 10 FP16 RTMDet + RTMPose Halpe26 inference.

This module is deliberately independent from the MMPose runtime backend.  It
uses the engines produced by ``scripts/tensorrt_fp16`` and only imports
TensorRT/PyTorch when an engine is opened, so the established PyTorch path is
unchanged on machines without TensorRT.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np

from .models import Pose2D


LOGGER = logging.getLogger(__name__)


_DETECTOR_INPUT_WH = (640, 640)
_DETECTOR_OUTPUT_REFERENCE_WH = _DETECTOR_INPUT_WH
_POSE_INPUT_WH = (192, 256)
_SIMCC_SPLIT_RATIO = 2.0
_POSE_BBOX_PADDING = 1.25

_DETECTOR_MEAN_BGR = np.asarray(
    [103.53, 116.28, 123.675], dtype=np.float32
)
_DETECTOR_STD_BGR = np.asarray(
    [57.375, 57.12, 58.395], dtype=np.float32
)
_POSE_MEAN_RGB = np.asarray(
    [123.675, 116.28, 103.53], dtype=np.float32
)
_POSE_STD_RGB = np.asarray(
    [58.395, 57.12, 57.375], dtype=np.float32
)


@dataclass(frozen=True)
class TensorRTHalpe26BackendConfig:
    """Runtime settings for the two TensorRT engines."""

    detector_engine: Path
    pose_engine: Path
    max_persons: int = 2
    bbox_threshold: float = 0.3
    keypoint_threshold: float = 0.3
    min_valid_keypoints: int = 10
    min_mean_keypoint_score: float = 0.3
    detector_output_reference_wh: tuple[int, int] = (
        _DETECTOR_OUTPUT_REFERENCE_WH
    )
    detector_interval: int = 1
    profile_timings: bool = False

    def __post_init__(self) -> None:
        if not 1 <= self.max_persons <= 4:
            raise ValueError("TensorRT max_persons must be in [1, 4].")
        if not 0.0 <= self.bbox_threshold <= 1.0:
            raise ValueError("bbox_threshold must be in [0, 1].")
        if not 0.0 <= self.keypoint_threshold <= 1.0:
            raise ValueError("keypoint_threshold must be in [0, 1].")
        if not 1 <= self.min_valid_keypoints <= 26:
            raise ValueError("min_valid_keypoints must be in [1, 26].")
        if not 0.0 <= self.min_mean_keypoint_score <= 1.0:
            raise ValueError("min_mean_keypoint_score must be in [0, 1].")
        if any(value <= 0 for value in self.detector_output_reference_wh):
            raise ValueError("detector_output_reference_wh must be positive.")
        if self.detector_interval <= 0:
            raise ValueError("detector_interval must be positive.")


def preprocess_detector(image_bgr: np.ndarray) -> np.ndarray:
    """Reproduce the MMDeploy RTMDet export preprocessing exactly."""

    image = _validate_bgr_image(image_bgr)
    resized = cv2.resize(
        image,
        _DETECTOR_INPUT_WH,
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    normalized = (resized - _DETECTOR_MEAN_BGR) / _DETECTOR_STD_BGR
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None])


def select_person_detections(
    dets: np.ndarray,
    labels: np.ndarray,
    *,
    image_wh: tuple[int, int],
    output_reference_wh: tuple[int, int],
    score_threshold: float,
    max_persons: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter NMS output and map detector-input boxes to this image."""

    boxes_with_scores = np.asarray(dets, dtype=np.float32)
    class_ids = np.asarray(labels)
    if boxes_with_scores.ndim == 3:
        if boxes_with_scores.shape[0] != 1:
            raise ValueError(
                "RTMDet engine must use batch 1, got "
                f"{boxes_with_scores.shape}."
            )
        boxes_with_scores = boxes_with_scores[0]
    if class_ids.ndim == 2:
        if class_ids.shape[0] != 1:
            raise ValueError(
                "RTMDet labels must use batch 1, got "
                f"{class_ids.shape}."
            )
        class_ids = class_ids[0]
    if boxes_with_scores.ndim != 2 or boxes_with_scores.shape[1] != 5:
        raise ValueError(
            f"Expected detector output (N, 5), got {boxes_with_scores.shape}."
        )
    if class_ids.shape != (boxes_with_scores.shape[0],):
        raise ValueError(
            "Detector boxes and labels have incompatible shapes: "
            f"{boxes_with_scores.shape}, {class_ids.shape}."
        )

    boxes = boxes_with_scores[:, :4].copy()
    scores = boxes_with_scores[:, 4].copy()
    finite = np.isfinite(boxes_with_scores).all(axis=1)
    is_person = np.rint(class_ids).astype(np.int64) == 0
    positive_area = (boxes[:, 2] > boxes[:, 0]) & (
        boxes[:, 3] > boxes[:, 1]
    )
    keep = finite & is_person & positive_area & (scores >= score_threshold)
    boxes = boxes[keep]
    scores = scores[keep]
    if not len(boxes):
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    reference_w, reference_h = output_reference_wh
    image_w, image_h = image_wh
    boxes[:, [0, 2]] *= float(image_w) / float(reference_w)
    boxes[:, [1, 3]] *= float(image_h) / float(reference_h)
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0.0, float(image_w - 1))
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0.0, float(image_h - 1))

    order = np.argsort(-scores, kind="stable")[:max_persons]
    return boxes[order], scores[order]


def bbox_center_scale(
    bbox_xyxy: np.ndarray,
    *,
    padding: float = _POSE_BBOX_PADDING,
    pose_input_wh: tuple[int, int] = _POSE_INPUT_WH,
) -> tuple[np.ndarray, np.ndarray]:
    """Match GetBBoxCenterScale + TopdownAffine aspect correction."""

    bbox = np.asarray(bbox_xyxy, dtype=np.float32)
    if bbox.shape != (4,):
        raise ValueError(f"Expected bbox shape (4,), got {bbox.shape}.")
    center = (bbox[2:] + bbox[:2]) * 0.5
    scale = (bbox[2:] - bbox[:2]) * float(padding)
    input_w, input_h = pose_input_wh
    aspect_ratio = float(input_w) / float(input_h)
    width, height = map(float, scale)
    if width > height * aspect_ratio:
        height = width / aspect_ratio
    else:
        width = height * aspect_ratio
    return center.astype(np.float32), np.asarray(
        [width, height], dtype=np.float32
    )


def pose_warp_matrix(
    center: np.ndarray,
    scale: np.ndarray,
    *,
    pose_input_wh: tuple[int, int] = _POSE_INPUT_WH,
) -> np.ndarray:
    """Return MMPose's zero-rotation affine matrix without importing MMPose."""

    center_value = np.asarray(center, dtype=np.float32)
    scale_value = np.asarray(scale, dtype=np.float32)
    if center_value.shape != (2,) or scale_value.shape != (2,):
        raise ValueError("Pose center and scale must both have shape (2,).")
    input_w, input_h = pose_input_wh
    source = np.asarray(
        [
            center_value,
            [center_value[0] - scale_value[0] * 0.5, center_value[1]],
            [center_value[0], center_value[1] - scale_value[0] * 0.5],
        ],
        dtype=np.float32,
    )
    destination = np.asarray(
        [
            [input_w * 0.5, input_h * 0.5],
            [0.0, input_h * 0.5],
            [input_w * 0.5, input_h * 0.5 - input_w * 0.5],
        ],
        dtype=np.float32,
    )
    return cv2.getAffineTransform(source, destination)


def preprocess_pose_crops(
    image_bgr: np.ndarray,
    boxes_xyxy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a batched normalized RTMPose tensor and mapping metadata."""

    image = _validate_bgr_image(image_bgr)
    boxes = np.asarray(boxes_xyxy, dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1:] != (4,):
        raise ValueError(f"Expected boxes shape (N, 4), got {boxes.shape}.")
    if not len(boxes):
        return (
            np.empty((0, 3, _POSE_INPUT_WH[1], _POSE_INPUT_WH[0]), np.float32),
            np.empty((0, 2), np.float32),
            np.empty((0, 2), np.float32),
        )

    input_w, input_h = _POSE_INPUT_WH
    tensors: list[np.ndarray] = []
    centers: list[np.ndarray] = []
    scales: list[np.ndarray] = []
    for bbox in boxes:
        center, scale = bbox_center_scale(bbox)
        warp = pose_warp_matrix(center, scale)
        crop_bgr = cv2.warpAffine(
            image,
            warp,
            (input_w, input_h),
            flags=cv2.INTER_LINEAR,
        )
        crop_rgb = crop_bgr[..., ::-1].astype(np.float32)
        normalized = (crop_rgb - _POSE_MEAN_RGB) / _POSE_STD_RGB
        tensors.append(normalized.transpose(2, 0, 1))
        centers.append(center)
        scales.append(scale)
    return (
        np.ascontiguousarray(np.stack(tensors).astype(np.float32)),
        np.stack(centers).astype(np.float32),
        np.stack(scales).astype(np.float32),
    )


def decode_simcc(
    simcc_x: np.ndarray,
    simcc_y: np.ndarray,
    centers: np.ndarray,
    scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode raw SimCC vectors and map keypoints back to RGB pixels."""

    x_values = np.asarray(simcc_x, dtype=np.float32)
    y_values = np.asarray(simcc_y, dtype=np.float32)
    center_values = np.asarray(centers, dtype=np.float32)
    scale_values = np.asarray(scales, dtype=np.float32)
    if x_values.ndim != 3 or x_values.shape[1:] != (26, 384):
        raise ValueError(f"Expected simcc_x (N, 26, 384), got {x_values.shape}.")
    if y_values.shape != (x_values.shape[0], 26, 512):
        raise ValueError(f"Expected simcc_y (N, 26, 512), got {y_values.shape}.")
    batch = x_values.shape[0]
    if center_values.shape != (batch, 2) or scale_values.shape != (batch, 2):
        raise ValueError("SimCC batch and center/scale metadata differ.")

    x_indices = np.argmax(x_values, axis=2).astype(np.float32)
    y_indices = np.argmax(y_values, axis=2).astype(np.float32)
    max_x = np.max(x_values, axis=2)
    max_y = np.max(y_values, axis=2)
    scores = np.minimum(max_x, max_y).astype(np.float32)
    keypoints_crop = np.stack([x_indices, y_indices], axis=2)
    keypoints_crop /= _SIMCC_SPLIT_RATIO
    invalid = scores <= 0.0
    keypoints_crop[invalid] = -1.0

    input_size = np.asarray(_POSE_INPUT_WH, dtype=np.float32)
    keypoints_image = (
        keypoints_crop / input_size[None, None, :] * scale_values[:, None, :]
        + center_values[:, None, :]
        - 0.5 * scale_values[:, None, :]
    )
    keypoints_image[invalid] = -1.0
    return keypoints_image.astype(np.float32), scores


class TensorRTHalpe26Backend:
    """Pose backend compatible with ``LocalMultiPersonPoseProcessor``."""

    def __init__(self, config: TensorRTHalpe26BackendConfig) -> None:
        detector_path = Path(config.detector_engine).expanduser().resolve()
        pose_path = Path(config.pose_engine).expanduser().resolve()
        for label, path in (
            ("RTMDet TensorRT engine", detector_path),
            ("RTMPose TensorRT engine", pose_path),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"{label} not found: {path}")

        self.device = "cuda:0 (TensorRT FP16)"
        self.detector = "tensorrt_rtmdet_m"
        self.max_persons = int(config.max_persons)
        self.bbox_threshold = float(config.bbox_threshold)
        self.keypoint_threshold = float(config.keypoint_threshold)
        self.min_valid_keypoints = int(config.min_valid_keypoints)
        self.min_mean_keypoint_score = float(
            config.min_mean_keypoint_score
        )
        self.detector_output_reference_wh = tuple(
            config.detector_output_reference_wh
        )
        self.detector_interval = int(config.detector_interval)
        self.profile_timings = bool(config.profile_timings)
        self.last_timing_ms: dict[str, float] = {}
        self._inference_index = 0
        self._cached_boxes = np.empty((0, 4), dtype=np.float32)
        self._cached_bbox_scores = np.empty((0,), dtype=np.float32)
        self._cached_image_wh: tuple[int, int] | None = None
        self._detector_engine = _TensorRTEngine(detector_path)
        self._pose_engine = _TensorRTEngine(pose_path)
        self._detector_engine.require_io(
            input_names=("input",), output_names=("dets", "labels")
        )
        self._pose_engine.require_io(
            input_names=("input",), output_names=("simcc_x", "simcc_y")
        )

    def infer(self, image: np.ndarray | str | Path) -> list[Pose2D]:
        started_at = time.perf_counter()
        if isinstance(image, (str, Path)):
            image_array = cv2.imread(str(image), cv2.IMREAD_COLOR)
            if image_array is None:
                raise FileNotFoundError(f"Could not read image: {image}")
        else:
            image_array = _validate_bgr_image(image)
        height, width = image_array.shape[:2]
        image_wh = (width, height)
        run_detector = (
            self.detector_interval == 1
            or not len(self._cached_boxes)
            or self._cached_image_wh != image_wh
            or self._inference_index % self.detector_interval == 0
        )
        self._inference_index += 1
        if run_detector:
            detector_input = preprocess_detector(image_array)
            detector_preprocessed_at = time.perf_counter()
            detector_outputs = self._detector_engine.infer(
                {"input": detector_input}
            )
            detector_finished_at = time.perf_counter()
            boxes, bbox_scores = select_person_detections(
                detector_outputs["dets"],
                detector_outputs["labels"],
                image_wh=image_wh,
                output_reference_wh=self.detector_output_reference_wh,
                score_threshold=self.bbox_threshold,
                max_persons=self.max_persons,
            )
            detector_postprocessed_at = time.perf_counter()
            self._cached_boxes = boxes.copy()
            self._cached_bbox_scores = bbox_scores.copy()
            self._cached_image_wh = image_wh
        else:
            detector_preprocessed_at = started_at
            detector_finished_at = started_at
            detector_postprocessed_at = started_at
            boxes = self._cached_boxes.copy()
            bbox_scores = self._cached_bbox_scores.copy()
        if not len(boxes):
            self._record_timings(
                started_at=started_at,
                detector_preprocessed_at=detector_preprocessed_at,
                detector_finished_at=detector_finished_at,
                detector_postprocessed_at=detector_postprocessed_at,
                detector_reused=not run_detector,
            )
            return []

        pose_inputs, centers, scales = preprocess_pose_crops(
            image_array, boxes
        )
        pose_preprocessed_at = time.perf_counter()
        pose_outputs = self._pose_engine.infer({"input": pose_inputs})
        pose_finished_at = time.perf_counter()
        keypoints, keypoint_scores = decode_simcc(
            pose_outputs["simcc_x"],
            pose_outputs["simcc_y"],
            centers,
            scales,
        )
        pose_decoded_at = time.perf_counter()

        poses: list[Pose2D] = []
        for index in range(len(boxes)):
            scores = keypoint_scores[index]
            valid_count = int(
                np.count_nonzero(scores >= self.keypoint_threshold)
            )
            mean_score = float(np.mean(scores))
            if (
                valid_count < self.min_valid_keypoints
                or mean_score < self.min_mean_keypoint_score
            ):
                continue
            poses.append(
                Pose2D(
                    keypoints=keypoints[index],
                    scores=scores,
                    bbox_xyxy=boxes[index],
                    bbox_score=float(bbox_scores[index]),
                )
            )
        poses.sort(key=lambda pose: pose.bbox_score, reverse=True)
        self._record_timings(
            started_at=started_at,
            detector_preprocessed_at=detector_preprocessed_at,
            detector_finished_at=detector_finished_at,
            detector_postprocessed_at=detector_postprocessed_at,
            pose_preprocessed_at=pose_preprocessed_at,
            pose_finished_at=pose_finished_at,
            pose_decoded_at=pose_decoded_at,
            finished_at=time.perf_counter(),
            detector_reused=not run_detector,
        )
        return poses

    def _record_timings(
        self,
        *,
        started_at: float,
        detector_preprocessed_at: float,
        detector_finished_at: float,
        detector_postprocessed_at: float,
        pose_preprocessed_at: float | None = None,
        pose_finished_at: float | None = None,
        pose_decoded_at: float | None = None,
        finished_at: float | None = None,
        detector_reused: bool = False,
    ) -> None:
        """Record one-frame stage timings for Nano-only diagnostics."""

        pose_preprocessed_at = pose_preprocessed_at or detector_postprocessed_at
        pose_finished_at = pose_finished_at or pose_preprocessed_at
        pose_decoded_at = pose_decoded_at or pose_finished_at
        finished_at = finished_at or pose_decoded_at
        milliseconds = 1000.0
        self.last_timing_ms = {
            "detector_preprocess": (
                detector_preprocessed_at - started_at
            ) * milliseconds,
            "detector_engine": (
                detector_finished_at - detector_preprocessed_at
            ) * milliseconds,
            "detector_postprocess": (
                detector_postprocessed_at - detector_finished_at
            ) * milliseconds,
            "pose_preprocess": (
                pose_preprocessed_at - detector_postprocessed_at
            ) * milliseconds,
            "pose_engine": (
                pose_finished_at - pose_preprocessed_at
            ) * milliseconds,
            "pose_decode": (
                pose_decoded_at - pose_finished_at
            ) * milliseconds,
            "pose_filter": (finished_at - pose_decoded_at) * milliseconds,
            "detector_reused": float(detector_reused),
            "total": (finished_at - started_at) * milliseconds,
        }
        if self.profile_timings:
            LOGGER.info(
                "TensorRT stages detector_reused=%d detector_preprocess=%.2f ms "
                "detector_engine=%.2f ms detector_postprocess=%.2f ms "
                "pose_preprocess=%.2f ms pose_engine=%.2f ms "
                "pose_decode=%.2f ms pose_filter=%.2f ms total=%.2f ms",
                int(self.last_timing_ms["detector_reused"]),
                self.last_timing_ms["detector_preprocess"],
                self.last_timing_ms["detector_engine"],
                self.last_timing_ms["detector_postprocess"],
                self.last_timing_ms["pose_preprocess"],
                self.last_timing_ms["pose_engine"],
                self.last_timing_ms["pose_decode"],
                self.last_timing_ms["pose_filter"],
                self.last_timing_ms["total"],
            )


class _TensorRTEngine:
    """Small TensorRT 10 runner backed by PyTorch CUDA allocations."""

    def __init__(self, engine_path: Path) -> None:
        try:
            import tensorrt as trt
            import torch
        except ImportError as error:
            raise RuntimeError(
                "TensorRT backend needs the Nano system tensorrt package and "
                "the CUDA-enabled humanpose PyTorch environment."
            ) from error
        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT backend requires CUDA.")

        self._trt = trt
        self._torch = torch
        self.path = engine_path
        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)
        engine_bytes = engine_path.read_bytes()
        self._engine = self._runtime.deserialize_cuda_engine(engine_bytes)
        if self._engine is None:
            raise RuntimeError(f"Could not deserialize TensorRT engine: {engine_path}")
        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError(
                f"Could not create TensorRT execution context: {engine_path}"
            )
        self._stream = torch.cuda.Stream()
        self.input_names: tuple[str, ...] = tuple(
            name
            for name in self._io_names()
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        )
        self.output_names: tuple[str, ...] = tuple(
            name
            for name in self._io_names()
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
        )
        self._output_allocator = self._make_output_allocator()
        self._static_output_buffers: dict[tuple[str, tuple[int, ...]], Any] = {}

    def _io_names(self) -> tuple[str, ...]:
        return tuple(
            self._engine.get_tensor_name(index)
            for index in range(self._engine.num_io_tensors)
        )

    def require_io(
        self,
        *,
        input_names: tuple[str, ...],
        output_names: tuple[str, ...],
    ) -> None:
        if self.input_names != input_names or self.output_names != output_names:
            raise RuntimeError(
                f"Unexpected TensorRT bindings in {self.path}: "
                f"inputs={self.input_names}, outputs={self.output_names}."
            )

    def _torch_dtype(self, name: str) -> Any:
        np_dtype = np.dtype(
            self._trt.nptype(self._engine.get_tensor_dtype(name))
        )
        mapping = {
            np.dtype(np.float32): self._torch.float32,
            np.dtype(np.float16): self._torch.float16,
            np.dtype(np.int64): self._torch.int64,
            np.dtype(np.int32): self._torch.int32,
            np.dtype(np.int8): self._torch.int8,
            np.dtype(np.uint8): self._torch.uint8,
            np.dtype(np.bool_): self._torch.bool,
        }
        try:
            return mapping[np_dtype]
        except KeyError as error:
            raise TypeError(
                f"Unsupported TensorRT dtype for {name}: {np_dtype}."
            ) from error

    def _make_output_allocator(self) -> Any:
        trt = self._trt
        torch = self._torch
        dtype_for = self._torch_dtype

        class TorchOutputAllocator(trt.IOutputAllocator):
            def __init__(self) -> None:
                trt.IOutputAllocator.__init__(self)
                self.buffers: dict[str, Any] = {}
                self.capacity_bytes: dict[str, int] = {}
                self.shapes: dict[str, tuple[int, ...]] = {}

            def _allocate(
                self,
                tensor_name: str,
                size: int,
                alignment: int,
            ) -> int:
                dtype = dtype_for(tensor_name)
                item_size = torch.empty((), dtype=dtype).element_size()
                if size % item_size:
                    raise RuntimeError(
                        f"TensorRT requested {size} bytes for {tensor_name}, "
                        f"not divisible by dtype size {item_size}."
                    )
                if size > self.capacity_bytes.get(tensor_name, 0):
                    self.buffers[tensor_name] = torch.empty(
                        max(1, size // item_size),
                        dtype=dtype,
                        device="cuda",
                    )
                    self.capacity_bytes[tensor_name] = size
                pointer = int(self.buffers[tensor_name].data_ptr())
                if alignment > 0 and pointer % alignment:
                    raise RuntimeError(
                        f"CUDA allocation for {tensor_name} is not aligned to "
                        f"{alignment} bytes."
                    )
                return pointer

            def reallocate_output(
                self,
                tensor_name: str,
                memory: int,
                size: int,
                alignment: int,
            ) -> int:
                del memory
                return self._allocate(tensor_name, int(size), int(alignment))

            def reallocate_output_async(
                self,
                tensor_name: str,
                memory: int,
                size: int,
                alignment: int,
                stream: int,
            ) -> int:
                del memory, stream
                return self._allocate(tensor_name, int(size), int(alignment))

            def notify_shape(self, tensor_name: str, shape: Any) -> None:
                self.shapes[tensor_name] = tuple(int(value) for value in shape)

            def begin(self) -> None:
                self.shapes.clear()

            def result(self, tensor_name: str) -> Any:
                if tensor_name not in self.buffers or tensor_name not in self.shapes:
                    raise RuntimeError(
                        f"TensorRT did not allocate/resolve output {tensor_name}."
                    )
                shape = self.shapes[tensor_name]
                if any(value < 0 for value in shape):
                    raise RuntimeError(
                        f"TensorRT output {tensor_name} is unresolved: {shape}."
                    )
                count = math.prod(shape)
                return self.buffers[tensor_name][:count].reshape(shape)

        return TorchOutputAllocator()

    def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if tuple(inputs) != self.input_names:
            raise ValueError(
                f"Expected TensorRT inputs {self.input_names}, got {tuple(inputs)}."
            )
        torch = self._torch
        context = self._context
        device_inputs: dict[str, Any] = {}
        static_outputs: dict[str, Any] = {}
        dynamic_outputs: set[str] = set()
        self._output_allocator.begin()

        with torch.cuda.stream(self._stream):
            for name, value in inputs.items():
                array = np.ascontiguousarray(value)
                expected_np_dtype = np.dtype(
                    self._trt.nptype(self._engine.get_tensor_dtype(name))
                )
                if array.dtype != expected_np_dtype:
                    array = array.astype(expected_np_dtype)
                device_value = torch.from_numpy(array).to(
                    device="cuda", non_blocking=False
                )
                device_inputs[name] = device_value
                if not context.set_input_shape(name, tuple(array.shape)):
                    raise RuntimeError(
                        f"TensorRT rejected input shape {array.shape} for {name}."
                    )
                if not context.set_tensor_address(name, int(device_value.data_ptr())):
                    raise RuntimeError(f"Could not bind TensorRT input {name}.")

            missing = tuple(context.infer_shapes())
            if missing:
                raise RuntimeError(
                    f"TensorRT shape inference is missing tensors: {missing}."
                )

            for name in self.output_names:
                shape = tuple(int(value) for value in context.get_tensor_shape(name))
                if any(value < 0 for value in shape):
                    dynamic_outputs.add(name)
                    if not context.set_output_allocator(
                        name, self._output_allocator
                    ):
                        raise RuntimeError(
                            f"Could not set TensorRT output allocator for {name}."
                        )
                    continue
                key = (name, shape)
                buffer = self._static_output_buffers.get(key)
                if buffer is None:
                    buffer = torch.empty(
                        shape,
                        dtype=self._torch_dtype(name),
                        device="cuda",
                    )
                    self._static_output_buffers[key] = buffer
                static_outputs[name] = buffer
                if not context.set_tensor_address(name, int(buffer.data_ptr())):
                    raise RuntimeError(f"Could not bind TensorRT output {name}.")

            success = context.execute_async_v3(
                stream_handle=int(self._stream.cuda_stream)
            )
            if not success:
                raise RuntimeError(f"TensorRT execution failed for {self.path}.")
        self._stream.synchronize()

        outputs: dict[str, np.ndarray] = {}
        for name in self.output_names:
            tensor = (
                self._output_allocator.result(name)
                if name in dynamic_outputs
                else static_outputs[name]
            )
            outputs[name] = tensor.detach().cpu().numpy().copy()
        return outputs


def _validate_bgr_image(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image)
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError(f"Expected an HxWx3 BGR image, got {value.shape}.")
    if value.dtype != np.uint8:
        raise ValueError(f"Expected uint8 BGR image, got {value.dtype}.")
    return value
