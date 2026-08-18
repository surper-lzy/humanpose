from types import SimpleNamespace

import pytest

from rgbd_avatar.pipeline.metrics import (
    duration_statistics,
    segment_metadata,
    value_statistics,
)


def test_duration_statistics_handles_empty_and_nonempty_inputs() -> None:
    assert duration_statistics([])["count"] == 0
    result = duration_statistics([10.0, 20.0, 30.0])
    assert result["count"] == 3
    assert result["mean_ms"] == pytest.approx(20.0)
    assert result["throughput_fps_from_mean"] == pytest.approx(50.0)


def test_value_statistics_reports_distribution() -> None:
    result = value_statistics([1.0, 2.0, 7.0])
    assert result["median"] == pytest.approx(2.0)
    assert result["max"] == pytest.approx(7.0)


def test_segment_metadata_keeps_discontinuities_separate() -> None:
    first = [
        SimpleNamespace(
            frame_index=0, timestamp_raw="a", relative_time_s=0.0
        ),
        SimpleNamespace(
            frame_index=1, timestamp_raw="b", relative_time_s=0.5
        ),
    ]
    second = [
        SimpleNamespace(
            frame_index=2, timestamp_raw="c", relative_time_s=3.0
        )
    ]
    metadata, intervals = segment_metadata([first, second])
    assert intervals == [pytest.approx(0.5)]
    assert metadata[0]["nominal_fps"] == pytest.approx(2.0)
    assert metadata[1]["median_interval_s"] is None
