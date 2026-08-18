#!/usr/bin/env python3
"""Run both TensorRT engines on one image before opening the live camera."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from rgbd_avatar.pose import (
    TensorRTHalpe26Backend,
    TensorRTHalpe26BackendConfig,
)
from rgbd_avatar.pose.visualization import draw_pose


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--detector-engine",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs/tensorrt_fp16/engines/rtmdet_m_person_640_fp16.engine"
        ),
    )
    parser.add_argument(
        "--pose-engine",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs/tensorrt_fp16/engines/rtmpose_m_halpe26_256x192_fp16.engine"
        ),
    )
    parser.add_argument("--max-persons", type=int, default=2)
    parser.add_argument("--bbox-threshold", type=float, default=0.3)
    parser.add_argument("--keypoint-threshold", type=float, default=0.3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/tensorrt_fp16/runtime-smoke",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_path = args.image.expanduser().resolve()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    backend = TensorRTHalpe26Backend(
        TensorRTHalpe26BackendConfig(
            detector_engine=args.detector_engine,
            pose_engine=args.pose_engine,
            max_persons=args.max_persons,
            bbox_threshold=args.bbox_threshold,
            keypoint_threshold=args.keypoint_threshold,
        )
    )
    poses = backend.infer(image)
    payload = {
        "image": str(image_path),
        "pose_count": len(poses),
        "poses": [
            pose.to_dict(score_threshold=args.keypoint_threshold)
            for pose in poses
        ],
    }

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{image_path.stem}_tensorrt.json"
    overlay_path = output_dir / f"{image_path.stem}_tensorrt_overlay.png"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    overlay = image
    for pose in poses:
        overlay = draw_pose(
            overlay,
            pose,
            score_threshold=args.keypoint_threshold,
        )
    if not cv2.imwrite(str(overlay_path), overlay):
        raise RuntimeError(f"Could not write overlay: {overlay_path}")

    print(f"TensorRT poses: {len(poses)}")
    for index, pose in enumerate(poses):
        valid = int(np.count_nonzero(pose.scores >= args.keypoint_threshold))
        print(
            f"pose[{index}] bbox_score={pose.bbox_score:.3f} "
            f"mean_score={pose.mean_score:.3f} valid={valid}/26 "
            f"bbox={pose.bbox_xyxy.tolist()}"
        )
    print(f"JSON: {json_path}")
    print(f"Overlay: {overlay_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
