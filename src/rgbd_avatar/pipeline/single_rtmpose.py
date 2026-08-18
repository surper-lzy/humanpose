#!/usr/bin/env python3
"""Run RTMPose-M Halpe26 inference on one RGB image."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import yaml

from rgbd_avatar.pose import RTMPoseBackend, RTMPoseBackendConfig
from rgbd_avatar.pose.visualization import draw_pose

LOGGER = logging.getLogger("test_rtmpose_single")
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        type=Path,
        default=PROJECT_ROOT.parent
        / "data/1/20260730_145911656_r.png",
        help="Input RGB image.",
    )
    parser.add_argument(
        "--pose-config",
        type=Path,
        default=PROJECT_ROOT / "configs/pose.yaml",
        help="Project pose YAML.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/pose2d",
        help="Directory for JSON and overlay outputs.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override device from YAML, e.g. cpu or cuda:0.",
    )
    parser.add_argument(
        "--detector",
        default=None,
        choices=("auto", "whole_image"),
        help="Override detector from YAML.",
    )
    return parser.parse_args()


def resolve_project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    image_path = args.image.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"RGB image not found: {image_path}")

    with args.pose_config.open("r", encoding="utf-8") as file:
        pose_config = yaml.safe_load(file)["pose"]

    backend_config = RTMPoseBackendConfig(
        model_config=resolve_project_path(pose_config["model_config"]),
        model_checkpoint=resolve_project_path(
            pose_config["model_checkpoint"]
        ),
        detector=args.detector or pose_config["detector"],
        model_cache_dir=resolve_project_path(pose_config["model_cache_dir"]),
        device=args.device or pose_config["device"],
        bbox_threshold=float(pose_config["bbox_threshold"]),
        keypoint_threshold=float(pose_config["keypoint_threshold"]),
        min_valid_keypoints=int(pose_config["min_valid_keypoints"]),
        min_mean_keypoint_score=float(
            pose_config["min_mean_keypoint_score"]
        ),
    )
    threshold = float(pose_config["keypoint_threshold"])

    LOGGER.info(
        "Initializing RTMPose on %s with detector=%s",
        backend_config.device,
        backend_config.detector,
    )
    backend = RTMPoseBackend(backend_config)
    LOGGER.info("Resolved inference device: %s", backend.device)

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"OpenCV failed to read image: {image_path}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = image_path.stem.removesuffix("_r")
    json_path = output_dir / f"{output_stem}_keypoints.json"
    overlay_path = output_dir / f"{output_stem}_overlay.png"

    poses = backend.infer(image_path)
    if not poses:
        output = {
            "source_image": str(image_path),
            "device": backend.device,
            "detector": backend.detector,
            "selected_person": None,
            "detected_person_count": 0,
        }
        with json_path.open("w", encoding="utf-8") as file:
            json.dump(output, file, indent=2, ensure_ascii=False)
            file.write("\n")
        cv2.putText(
            image_bgr,
            "No person",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        if not cv2.imwrite(str(overlay_path), image_bgr):
            raise RuntimeError(f"Failed to save overlay: {overlay_path}")
        LOGGER.info("No person passed the detection and pose quality gates")
        LOGGER.info("Saved JSON: %s", json_path)
        LOGGER.info("Saved overlay: %s", overlay_path)
        return 0

    pose = poses[0]
    overlay = draw_pose(image_bgr, pose, score_threshold=threshold)
    output = {
        "source_image": str(image_path),
        "device": backend.device,
        "detector": backend.detector,
        "selected_person": pose.to_dict(score_threshold=threshold),
        "detected_person_count": len(poses),
    }
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)
        file.write("\n")

    if not cv2.imwrite(str(overlay_path), overlay):
        raise RuntimeError(f"Failed to save overlay: {overlay_path}")

    valid_count = int((pose.scores >= threshold).sum())
    LOGGER.info(
        "Detected %d person(s); selected pose has %d/26 valid keypoints, "
        "mean score %.3f",
        len(poses),
        valid_count,
        pose.mean_score,
    )
    LOGGER.info("Saved JSON: %s", json_path)
    LOGGER.info("Saved overlay: %s", overlay_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
