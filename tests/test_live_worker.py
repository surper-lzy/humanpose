from types import SimpleNamespace
import threading

import numpy as np

from rgbd_avatar.pipeline.live_mannequin import LatestPoseWorker


class _OneFrameSource:
    def __init__(self) -> None:
        self.started = False
        self.closed = False
        self.start_thread = None
        self.read_thread = None
        self.close_thread = None

    @property
    def source_id(self) -> str:
        return "test:one-frame"

    def start(self) -> None:
        self.started = True
        self.start_thread = threading.get_ident()

    def read(self, timeout_ms: int = 1000):
        assert timeout_ms == 50
        self.read_thread = threading.get_ident()
        return SimpleNamespace(frame_number=4)

    def close(self) -> None:
        self.closed = True
        self.close_thread = threading.get_ident()


class _FixedProcessor:
    def process(self, frame):
        return SimpleNamespace(
            frame_number=frame.frame_number,
            status="ok",
            pose3d_output=SimpleNamespace(usable=np.ones(26, dtype=bool)),
            timing_ms={"inference": 1.0, "recovery": 2.0, "total": 3.0},
        )


def test_latest_pose_worker_sends_each_result_to_nonblocking_sink() -> None:
    caller_thread = threading.get_ident()
    source = _OneFrameSource()
    received = []
    worker = LatestPoseWorker(
        source,
        _FixedProcessor(),
        read_timeout_ms=50,
        max_frames=1,
        result_sink=received.append,
    )

    worker.start()
    assert worker.wait(timeout_s=1.0)
    worker.stop()

    assert source.started
    assert source.closed
    assert source.start_thread == source.read_thread == source.close_thread
    assert source.start_thread != caller_thread
    assert [result.frame_number for result in received] == [4]
    assert worker.error is None
