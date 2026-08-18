from pathlib import Path

import cv2
import numpy as np
import pytest

from rgbd_avatar.camera import CameraIntrinsics
from rgbd_avatar.live import DirectoryRGBDSource, capture_timestamp_ns


def _write_pair(directory: Path, timestamp: str, depth_mm: int) -> None:
    rgb = np.zeros((3, 4, 3), dtype=np.uint8)
    rgb[..., 1] = 120
    depth = np.full((3, 4), depth_mm, dtype=np.uint16)
    assert cv2.imwrite(str(directory / f"{timestamp}_r.png"), rgb)
    assert cv2.imwrite(str(directory / f"{timestamp}_d.pgm"), depth)


def _source(directory: Path, *, start_at: str) -> DirectoryRGBDSource:
    return DirectoryRGBDSource(
        directory,
        intrinsics=CameraIntrinsics(100.0, 100.0, 1.5, 1.0, 4, 3),
        depth_scale=0.001,
        start_at=start_at,
        poll_interval_s=0.001,
        stable_interval_s=0.0,
    )


def test_capture_timestamp_ns_preserves_millisecond_intervals() -> None:
    first = capture_timestamp_ns("20260805_165548996")
    second = capture_timestamp_ns("20260805_165549496")
    assert second - first == 500_000_000


def test_directory_source_latest_reads_newest_complete_pair(tmp_path: Path) -> None:
    _write_pair(tmp_path, "20260805_165548996", 1000)
    _write_pair(tmp_path, "20260805_165549496", 1250)
    source = _source(tmp_path, start_at="latest")
    source.start()

    frame = source.read(timeout_ms=100)

    assert frame.frame_number == 0
    assert frame.rgb_bgr.shape == (3, 4, 3)
    np.testing.assert_allclose(frame.depth_m, 1.25)
    assert source.stats.last_timestamp_raw == "20260805_165549496"
    source.close()


def test_directory_source_new_ignores_existing_and_waits_for_next_pair(
    tmp_path: Path,
) -> None:
    _write_pair(tmp_path, "20260805_165548996", 1000)
    source = _source(tmp_path, start_at="new")
    source.start()
    with pytest.raises(TimeoutError):
        source.read(timeout_ms=5)

    _write_pair(tmp_path, "20260805_165549496", 1500)
    frame = source.read(timeout_ms=100)

    np.testing.assert_allclose(frame.depth_m, 1.5)
    source.close()


def test_directory_source_skips_incomplete_and_counts_superseded_pairs(
    tmp_path: Path,
) -> None:
    _write_pair(tmp_path, "20260805_165548996", 1000)
    source = _source(tmp_path, start_at="latest")
    source.start()
    source.read(timeout_ms=100)
    cv2.imwrite(
        str(tmp_path / "20260805_165549196_r.png"),
        np.zeros((3, 4, 3), dtype=np.uint8),
    )
    _write_pair(tmp_path, "20260805_165549496", 1250)
    _write_pair(tmp_path, "20260805_165549996", 1500)

    frame = source.read(timeout_ms=100)

    np.testing.assert_allclose(frame.depth_m, 1.5)
    assert source.stats.dropped_complete_frame_count == 1
    source.close()
