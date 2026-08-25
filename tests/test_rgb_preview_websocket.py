from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np

from rgbd_avatar.live.rgb_preview_websocket import (
    RgbPreviewPublishConfig,
    RgbPreviewWebSocketPublisher,
    encode_rgb_preview_jpeg,
)


def test_preview_connection_url_adds_binary_role_and_source() -> None:
    config = RgbPreviewPublishConfig(
        enabled=True,
        url="ws://127.0.0.1:8000/api/realtime/ws",
        source_id="FF6690772788",
    )
    assert config.connection_url.endswith(
        "?preview_role=producer&preview_source_id=FF6690772788"
    )


def test_preview_encoder_returns_scaled_jpeg() -> None:
    result = SimpleNamespace(
        rgb_bgr=np.full((120, 160, 3), 96, dtype=np.uint8),
        persons=(),
        frame_number=7,
        detected_person_count=0,
        status="no_person",
        recovery_stats={},
        timing_ms={
            "inference": 1.0,
            "recovery": 2.0,
            "matching": 0.5,
            "quality": 0.25,
            "total": 4.0,
        },
    )
    config = RgbPreviewPublishConfig(
        enabled=True,
        scale=0.5,
        jpeg_quality=70,
    )

    jpeg = encode_rgb_preview_jpeg(result, config)
    decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)

    assert jpeg.startswith(b"\xff\xd8") and jpeg.endswith(b"\xff\xd9")
    assert decoded.shape == (60, 80, 3)


def test_preview_submit_does_not_encode_on_pose_thread(monkeypatch) -> None:
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("JPEG encoding ran synchronously in submit()")

    monkeypatch.setattr(
        "rgbd_avatar.live.rgb_preview_websocket.encode_rgb_preview_jpeg",
        fail_if_called,
    )
    publisher = RgbPreviewWebSocketPublisher(
        RgbPreviewPublishConfig(enabled=True, fps=5.0)
    )

    publisher.submit(SimpleNamespace(frame_number=12))

    assert publisher.stats.submitted_frame_count == 1
    assert publisher.stats.last_jpeg_bytes is None
