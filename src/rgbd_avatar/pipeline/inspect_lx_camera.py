#!/usr/bin/env python3
"""Probe direct MRDVS SDK capture without loading the pose model or a GUI."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import statistics
import time

import numpy as np

from rgbd_avatar.io import load_yaml_mapping
from rgbd_avatar.live import LxCameraRGBDSource


LOGGER = logging.getLogger("inspect_lx_camera")
PROJECT_ROOT = Path(__file__).resolve().parents[3]


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
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--timeout-ms", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    if args.frames < 2:
        raise ValueError("--frames must be at least 2.")
    live_config = load_yaml_mapping(args.live_config)["live"]
    camera_config = load_yaml_mapping(args.camera_config)["camera"]
    source_config = live_config["source"]
    timeout_ms = args.timeout_ms or int(source_config["read_timeout_ms"])
    source = LxCameraRGBDSource.from_mapping(
        source_config["sdk"],
        depth_scale=float(camera_config["depth_scale"]),
    )

    host_arrivals_ns: list[int] = []
    frame_timestamps_ns: list[int] = []
    valid_ratios: list[float] = []
    median_depths_m: list[float] = []
    rgb_depth_skews_ms: list[float] = []
    try:
        LOGGER.info("Opening %s", source.source_id)
        source.start()
        LOGGER.info("Runtime aligned intrinsics: %s", source.intrinsics)
        for index in range(args.frames):
            frame = source.read(timeout_ms=timeout_ms)
            host_arrivals_ns.append(time.monotonic_ns())
            frame_timestamps_ns.append(frame.timestamp_ns)
            valid = np.isfinite(frame.depth_m) & (frame.depth_m > 0.0)
            valid_ratios.append(float(np.mean(valid)))
            median_depths_m.append(
                float(np.median(frame.depth_m[valid]))
                if np.any(valid)
                else float("nan")
            )
            stats = source.stats
            if (
                stats.last_depth_sensor_timestamp is not None
                and stats.last_rgb_sensor_timestamp is not None
            ):
                rgb_depth_skews_ms.append(
                    (
                        stats.last_rgb_sensor_timestamp
                        - stats.last_depth_sensor_timestamp
                    )
                    * 1e-3
                )
            if index == 0 or (index + 1) % 10 == 0:
                LOGGER.info(
                    "capture [%d/%d] shape=%s valid_depth=%.1f%% "
                    "median_depth=%.3f m",
                    index + 1,
                    args.frames,
                    frame.rgb_bgr.shape,
                    valid_ratios[-1] * 100.0,
                    median_depths_m[-1],
                )
    finally:
        source.close()

    host_intervals_ms = np.diff(host_arrivals_ns) * 1e-6
    sensor_intervals_ms = np.diff(frame_timestamps_ns) * 1e-6
    host_interval_ms = float(np.median(host_intervals_ms))
    sensor_interval_ms = float(np.median(sensor_intervals_ms))
    host_fps = 1000.0 / host_interval_ms if host_interval_ms > 0.0 else 0.0
    sensor_fps = (
        1000.0 / sensor_interval_ms if sensor_interval_ms > 0.0 else 0.0
    )
    LOGGER.info(
        "Probe complete: host median interval=%.2f ms (%.2f FPS), "
        "sensor median interval=%.2f ms (%.2f FPS)",
        host_interval_ms,
        host_fps,
        sensor_interval_ms,
        sensor_fps,
    )
    if rgb_depth_skews_ms:
        LOGGER.info(
            "RGB minus depth sensor timestamp: median=%.2f ms, "
            "range=[%.2f, %.2f] ms",
            float(np.median(rgb_depth_skews_ms)),
            float(np.min(rgb_depth_skews_ms)),
            float(np.max(rgb_depth_skews_ms)),
        )
    LOGGER.info(
        "Depth valid median=%.1f%%, scene depth median=%.3f m",
        statistics.median(valid_ratios) * 100.0,
        statistics.median(median_depths_m),
    )
    LOGGER.info("SDK source statistics: %s", source.stats)
    return 0
