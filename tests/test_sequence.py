from pathlib import Path

import pytest

from rgbd_avatar.data import discover_rgbd_sequence, split_at_time_gaps


def _touch_frame(
    directory: Path,
    timestamp: str,
    *,
    amplitude: bool = True,
    point_cloud: bool = True,
) -> None:
    (directory / f"{timestamp}_r.png").touch()
    (directory / f"{timestamp}_d.pgm").touch()
    if amplitude:
        (directory / f"{timestamp}_a.pgm").touch()
    if point_cloud:
        (directory / f"{timestamp}_t.pcd").touch()


def test_discover_sequence_pairs_and_sorts_exact_timestamps(
    tmp_path: Path,
) -> None:
    _touch_frame(tmp_path, "20260730_150451282", amplitude=False)
    _touch_frame(tmp_path, "20260730_150448663")
    _touch_frame(tmp_path, "20260730_150448159")
    (tmp_path / "notes.txt").touch()

    frames = discover_rgbd_sequence(tmp_path)

    assert [frame.timestamp_raw for frame in frames] == [
        "20260730_150448159",
        "20260730_150448663",
        "20260730_150451282",
    ]
    assert frames[0].relative_time_s == 0.0
    assert frames[1].relative_time_s == pytest.approx(0.504)
    assert frames[2].relative_time_s == pytest.approx(3.123)
    assert frames[2].amplitude_path is None
    assert frames[2].point_cloud_path is not None

    segments = split_at_time_gaps(frames, max_gap_s=2.0)
    assert [len(segment) for segment in segments] == [2, 1]


def test_discover_sequence_rejects_unpaired_required_stream(
    tmp_path: Path,
) -> None:
    (tmp_path / "20260730_145911656_r.png").touch()

    with pytest.raises(FileNotFoundError, match="missing d"):
        discover_rgbd_sequence(tmp_path)


def test_split_rejects_non_positive_threshold(tmp_path: Path) -> None:
    _touch_frame(tmp_path, "20260730_145911656")
    frames = discover_rgbd_sequence(tmp_path)

    with pytest.raises(ValueError, match="positive"):
        split_at_time_gaps(frames, max_gap_s=0.0)
