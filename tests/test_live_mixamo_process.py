"""Tests for the process-isolated live Mixamo transport."""

from __future__ import annotations

import pickle
from queue import Empty, Full
from types import SimpleNamespace

import numpy as np

from rgbd_avatar.visualization.live_mixamo_process import (
    LiveMixamoPosePacket,
    LiveMixamoViewerProcess,
)


def _result(frame_number: int = 4):
    temporal = SimpleNamespace(
        confidence=np.linspace(0.0, 1.0, 26, dtype=np.float32),
        usable=np.ones(26, dtype=bool),
        predicted=np.zeros(26, dtype=bool),
    )
    return SimpleNamespace(
        source_id="camera",
        frame_number=frame_number,
        timestamp_ns=1_000_000_000 + frame_number,
        joints_application_m=np.arange(78, dtype=np.float32).reshape(26, 3),
        pose3d_output=temporal,
        presence=SimpleNamespace(
            track_reset_required=True,
            reacquired_after_exit=False,
        ),
    )


def test_pose_packet_is_compact_picklable_structural_input() -> None:
    result = _result()
    packet = LiveMixamoPosePacket.from_result(result)
    result.joints_application_m[:] = -1.0

    restored = pickle.loads(pickle.dumps(packet))
    assert restored.frame_number == 4
    assert restored.joints_application_m.shape == (26, 3)
    assert restored.joints_application_m[0, 0] == 0.0
    assert restored.track_reset_required
    assert restored.pose3d_output is restored
    assert restored.presence is restored


def test_submit_replaces_queued_pose_without_blocking() -> None:
    class _AliveProcess:
        exitcode = None

        @staticmethod
        def is_alive() -> bool:
            return True

    class _OneItemQueue:
        def __init__(self) -> None:
            self.item = LiveMixamoPosePacket.from_result(_result(1))

        def put_nowait(self, value) -> None:
            if self.item is not None:
                raise Full
            self.item = value

        def get_nowait(self):
            if self.item is None:
                raise Empty
            value = self.item
            self.item = None
            return value

    controller = object.__new__(LiveMixamoViewerProcess)
    controller._started = True
    controller._closed = False
    controller._process = _AliveProcess()
    controller._pose_queue = _OneItemQueue()

    assert controller.submit(_result(2))
    assert controller._pose_queue.item.frame_number == 2

