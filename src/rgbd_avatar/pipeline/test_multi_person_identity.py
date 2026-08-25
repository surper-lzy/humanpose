#!/usr/bin/env python3
"""Compare current multi-person IDs with an isolated RGB-D shadow tracker."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import time

import cv2
import numpy as np

from rgbd_avatar.io import load_yaml_mapping
from rgbd_avatar.live import LocalPersonPoseResult
from rgbd_avatar.pose import HALPE26_LINKS
from rgbd_avatar.tracking.shadow_identity import (
    ShadowIdentityConfig,
    ShadowIdentityObservation,
    ShadowRGBDIdentityTracker,
    extract_upper_body_hsv_descriptor,
)

from .live_multi_person import _build_processor, _build_source


LOGGER = logging.getLogger("test_multi_person_identity")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
_TORSO_INDICES = np.asarray((18, 19, 5, 6, 11, 12), dtype=np.int64)
_COLORS = (
    (80, 220, 80),
    (80, 160, 255),
    (240, 120, 80),
    (220, 80, 220),
    (80, 220, 220),
    (220, 220, 80),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live-config",
        type=Path,
        default=PROJECT_ROOT / "configs/live.yaml",
    )
    parser.add_argument(
        "--camera-config",
        type=Path,
        default=PROJECT_ROOT / "configs/camera.yaml",
    )
    parser.add_argument(
        "--pose-config",
        type=Path,
        default=PROJECT_ROOT / "configs/pose.yaml",
    )
    parser.add_argument(
        "--tracking-config",
        type=Path,
        default=PROJECT_ROOT / "configs/tracking.yaml",
    )
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument(
        "--source",
        choices=("directory", "sdk"),
        default=None,
    )
    parser.add_argument(
        "--start-at",
        choices=("latest", "new", "oldest"),
        default=None,
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--detector",
        choices=("auto", "whole_image"),
        default=None,
    )
    parser.add_argument(
        "--recovery-method",
        choices=(
            "window_median",
            "guided_window",
            "depth_connected",
            "pointcloud_cluster",
            "hybrid",
            "adaptive_hybrid",
        ),
        default="hybrid",
    )
    parser.add_argument("--max-persons", type=int, default=4)
    parser.add_argument("--max-missing-s", type=float, default=0.35)
    parser.add_argument("--shadow-normal-missing-s", type=float, default=0.35)
    parser.add_argument("--shadow-occluded-missing-s", type=float, default=1.0)
    parser.add_argument("--overlap-iou", type=float, default=0.25)
    parser.add_argument("--ambiguity-margin", type=float, default=0.08)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Log the comparison without opening an OpenCV window.",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--save-video",
        type=Path,
        default=None,
        help="Optionally save the annotated comparison as an MP4 file.",
    )
    return parser.parse_args()


def _root_camera_m(person: LocalPersonPoseResult) -> np.ndarray | None:
    pose3d = person.pose3d_raw
    if pose3d is None:
        return None
    if pose3d.valid[19] and np.isfinite(pose3d.joints_m[19]).all():
        return pose3d.joints_m[19].astype(np.float64, copy=True)
    valid = _TORSO_INDICES[
        pose3d.valid[_TORSO_INDICES]
        & np.isfinite(pose3d.joints_m[_TORSO_INDICES]).all(axis=1)
    ]
    if not len(valid):
        return None
    return np.median(pose3d.joints_m[valid], axis=0).astype(np.float64)


def _observations(result) -> list[ShadowIdentityObservation]:
    observations: list[ShadowIdentityObservation] = []
    for person in result.persons:
        if not person.observed_in_frame or person.pose2d is None:
            continue
        observations.append(
            ShadowIdentityObservation(
                # This is only a token that lets the overlay compare the
                # production ID with the independent shadow ID.
                observation_id=person.track_id,
                pose2d=person.pose2d,
                root_camera_m=_root_camera_m(person),
                appearance=extract_upper_body_hsv_descriptor(
                    result.rgb_bgr,
                    person.pose2d,
                ),
            )
        )
    return observations


def _color(shadow_id: int) -> tuple[int, int, int]:
    return _COLORS[(shadow_id - 1) % len(_COLORS)]


def _draw_comparison(result, shadow_frame) -> np.ndarray:
    image = result.rgb_bgr.copy()
    assignment_by_current = {
        assignment.observation_id: assignment
        for assignment in shadow_frame.assignments
    }
    ambiguous = set(shadow_frame.ambiguous_observation_ids)
    for person in result.persons:
        if not person.observed_in_frame or person.pose2d is None:
            continue
        pose = person.pose2d
        assignment = assignment_by_current.get(person.track_id)
        if assignment is None:
            color = (0, 0, 255) if person.track_id in ambiguous else (160, 160, 160)
            label = f"current:{person.track_id} shadow:?"
        else:
            color = _color(assignment.shadow_id)
            suffix = " frozen" if assignment.appearance_frozen else ""
            label = (
                f"current:{person.track_id} shadow:{assignment.shadow_id}"
                f"{suffix}"
            )
        x1, y1, x2, y2 = np.rint(pose.bbox_xyxy).astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            label,
            (max(0, x1), max(18, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
        visible = pose.scores >= 0.3
        for first, second in HALPE26_LINKS:
            if not visible[first] or not visible[second]:
                continue
            first_xy = tuple(np.rint(pose.keypoints[first]).astype(int))
            second_xy = tuple(np.rint(pose.keypoints[second]).astype(int))
            cv2.line(image, first_xy, second_xy, color, 1, cv2.LINE_AA)

    summary = (
        f"frame={result.frame_number} detected={result.detected_person_count} "
        f"shadow={len(assignment_by_current)} "
        f"pose={result.timing_ms['total']:.1f}ms "
        f"shadow_assoc={shadow_frame.timing_ms:.2f}ms"
    )
    cv2.rectangle(image, (0, 0), (image.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(
        image,
        summary,
        (8, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return image


def _open_video_writer(
    output_path: Path,
    image: np.ndarray,
) -> cv2.VideoWriter:
    path = output_path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = image.shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        15.0,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open comparison video: {path}")
    return writer


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    if args.max_persons <= 0:
        raise ValueError("--max-persons must be positive.")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive.")

    live_config = load_yaml_mapping(args.live_config)["live"]
    camera_config = load_yaml_mapping(args.camera_config)["camera"]
    pose_config = load_yaml_mapping(args.pose_config)["pose"]
    tracking_config = load_yaml_mapping(args.tracking_config)["tracking"]
    source = _build_source(args, live_config, camera_config)
    processor = _build_processor(
        args,
        live_config,
        camera_config,
        pose_config,
        tracking_config,
    )
    shadow_tracker = ShadowRGBDIdentityTracker(
        ShadowIdentityConfig(
            normal_missing_s=args.shadow_normal_missing_s,
            occluded_missing_s=args.shadow_occluded_missing_s,
            overlap_iou=args.overlap_iou,
            ambiguity_margin=args.ambiguity_margin,
        )
    )
    read_timeout_ms = int(live_config["source"]["read_timeout_ms"])
    writer: cv2.VideoWriter | None = None
    processed = 0
    current_id_by_shadow: dict[int, int] = {}
    mapping_change_count = 0

    LOGGER.info("Identity test input: %s", source.source_id)
    LOGGER.info(
        "Shadow-only test: existing pose IDs and WebSocket path are unchanged."
    )
    LOGGER.info(
        "current tracker = geometry Hungarian; shadow tracker = motion + "
        "HSV appearance + RGB-D root + occlusion ambiguity gate"
    )
    source.start()
    try:
        while args.max_frames is None or processed < args.max_frames:
            try:
                frame = source.read(timeout_ms=read_timeout_ms)
            except TimeoutError:
                continue
            result = processor.process(frame)
            shadow_frame = shadow_tracker.update(
                _observations(result),
                frame.timestamp_ns * 1e-9,
            )
            pairs = [
                {
                    "current": item.observation_id,
                    "shadow": item.shadow_id,
                    "cost": (
                        round(item.match_cost, 3)
                        if item.match_cost is not None
                        else None
                    ),
                    "frozen": item.appearance_frozen,
                }
                for item in shadow_frame.assignments
            ]
            for item in shadow_frame.assignments:
                previous_current_id = current_id_by_shadow.get(item.shadow_id)
                if (
                    previous_current_id is not None
                    and previous_current_id != item.observation_id
                ):
                    mapping_change_count += 1
                current_id_by_shadow[item.shadow_id] = item.observation_id
            LOGGER.info(
                "frame=%d detected=%d pairs=%s predicted=%s ambiguous=%s "
                "removed=%s remaps=%d pose=%.1f ms shadow=%.2f ms",
                result.frame_number,
                result.detected_person_count,
                pairs,
                shadow_frame.predicted_shadow_ids,
                shadow_frame.ambiguous_observation_ids,
                shadow_frame.removed_shadow_ids,
                mapping_change_count,
                result.timing_ms["total"],
                shadow_frame.timing_ms,
            )
            annotated = _draw_comparison(result, shadow_frame)
            if args.save_video is not None:
                if writer is None:
                    writer = _open_video_writer(args.save_video, annotated)
                writer.write(annotated)
            processed += 1
            if args.validate_only:
                continue
            cv2.imshow("RGB-D identity shadow test", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            time.sleep(0.001)
    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user.")
    finally:
        if writer is not None:
            writer.release()
        if not args.validate_only:
            cv2.destroyAllWindows()
        source.close()
    LOGGER.info(
        "Identity shadow test complete: frames=%d mapping_changes=%d",
        processed,
        mapping_change_count,
    )
    LOGGER.info("Identity test source statistics: %s", source.stats)
    return 0
