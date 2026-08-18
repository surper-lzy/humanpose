from ctypes import Structure, c_uint, c_void_p, cast, pointer
from enum import Enum, auto
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rgbd_avatar.live import LxCameraRGBDSource, LxCameraSDKError


class _State(Enum):
    LX_SUCCESS = 0
    LX_E_TIME_OUT = -18
    LX_E_FRAME_ID_NOT_MATCH = -28
    LX_E_FRAME_MULTI_MACHINE = -29
    LX_E_CTRL_PERMISS_ERROR = -9


class _OpenMode(Enum):
    OPEN_BY_INDEX = 0
    OPEN_BY_IP = 1
    OPEN_BY_SN = 2
    OPEN_BY_ID = 3


class _AlignMode(Enum):
    DEPTH_TO_RGB = 1


class _AlgorithmMode(Enum):
    MODE_ALL_OFF = 0


class _Feature(Enum):
    LX_BOOL_ENABLE_2D_STREAM = auto()
    LX_BOOL_ENABLE_3D_DEPTH_STREAM = auto()
    LX_BOOL_ENABLE_3D_AMP_STREAM = auto()
    LX_BOOL_ENABLE_SYNC_FRAME = auto()
    LX_INT_RGBD_ALIGN_MODE = auto()
    LX_INT_ALGORITHM_MODE = auto()
    LX_INT_2D_IMAGE_WIDTH = auto()
    LX_INT_2D_IMAGE_HEIGHT = auto()
    LX_INT_3D_IMAGE_WIDTH = auto()
    LX_INT_3D_IMAGE_HEIGHT = auto()
    LX_FLOAT_3D_DEPTH_FPS = auto()
    LX_FLOAT_2D_IMAGE_FPS = auto()


class _FrameExtendInfo(Structure):
    _fields_ = [
        ("depth_frame_id", c_uint),
        ("rgb_frame_id", c_uint),
    ]


class _FakeCamera:
    def __init__(self, _library_path: str, *, depth_width: int = 4) -> None:
        self.depth_width = depth_width
        self.bool_settings = []
        self.int_settings = []
        self.rgb = np.zeros((3, 4, 3), dtype=np.uint8)
        self.rgb[..., 0] = 10
        self.rgb[..., 1] = 20
        self.rgb[..., 2] = 30
        self.depth = np.full((3, depth_width), 1250, dtype=np.uint16)
        self.stopped = False
        self.closed = False

    def DcOpenDevice(self, mode, param):
        self.open_call = (mode, param)
        return _State.LX_SUCCESS, object(), object()

    def DcGetDeviceList(self):
        self.enumerated = True
        return _State.LX_SUCCESS, object(), 1

    def DcSetBoolValue(self, _handle, feature, value):
        self.bool_settings.append((feature, value))
        return _State.LX_SUCCESS

    def DcSetIntValue(self, _handle, feature, value):
        self.int_settings.append((feature, value))
        return _State.LX_SUCCESS

    def DcGetIntValue(self, _handle, feature):
        values = {
            _Feature.LX_INT_2D_IMAGE_WIDTH: 4,
            _Feature.LX_INT_2D_IMAGE_HEIGHT: 3,
            _Feature.LX_INT_3D_IMAGE_WIDTH: self.depth_width,
            _Feature.LX_INT_3D_IMAGE_HEIGHT: 3,
        }
        return _State.LX_SUCCESS, SimpleNamespace(cur_value=values[feature])

    def get2DIntricParam(self, _handle):
        return _State.LX_SUCCESS, [100.0, 101.0, 1.5, 1.0], [0.0] * 5

    def DcStartStream(self, _handle):
        return _State.LX_SUCCESS

    def getFrame(self, _handle):
        frame = SimpleNamespace(
            depth_data=SimpleNamespace(sensor_timestamp=123_456),
            rgb_data=SimpleNamespace(sensor_timestamp=123_455),
        )
        return _State.LX_SUCCESS, frame

    def getRGBImage(self, _frame):
        return _State.LX_SUCCESS, self.rgb

    def getDepthImage(self, _frame):
        return _State.LX_SUCCESS, self.depth

    def DcGetFloatValue(self, _handle, feature):
        value = 15.0 if feature == _Feature.LX_FLOAT_3D_DEPTH_FPS else 14.5
        return _State.LX_SUCCESS, SimpleNamespace(cur_value=value)

    def DcGetErrorString(self, state):
        return state.name

    def DcStopStream(self, _handle):
        self.stopped = True
        return _State.LX_SUCCESS

    def DcCloseDevice(self, _handle):
        self.closed = True
        return _State.LX_SUCCESS


def _sdk(camera: _FakeCamera):
    return SimpleNamespace(
        LxCamera=lambda _path: camera,
        LX_STATE=_State,
        LX_OPEN_MODE=_OpenMode,
        LX_RGBD_ALIGN_MODE=_AlignMode,
        LX_ALGORITHM_MODE=_AlgorithmMode,
        LX_CAMERA_FEATURE=_Feature,
        FrameExtendInfo=_FrameExtendInfo,
    )


def _source(tmp_path: Path, camera: _FakeCamera) -> LxCameraRGBDSource:
    library = tmp_path / "libLxCameraApi.so"
    library.touch()
    return LxCameraRGBDSource(
        library_path=library,
        open_mode="id",
        open_param="FF6690772788",
        depth_scale=0.001,
        sdk_module=_sdk(camera),
    )


def test_lx_camera_source_configures_aligned_sync_stream_and_copies_frames(
    tmp_path: Path,
) -> None:
    camera = _FakeCamera("unused")
    source = _source(tmp_path, camera)

    source.start()
    frame = source.read(timeout_ms=100)
    camera.rgb.fill(0)
    camera.depth.fill(0)

    assert camera.open_call == (_OpenMode.OPEN_BY_ID, "FF6690772788")
    assert camera.enumerated
    assert (_Feature.LX_BOOL_ENABLE_2D_STREAM, True) in camera.bool_settings
    assert (
        _Feature.LX_BOOL_ENABLE_3D_DEPTH_STREAM,
        True,
    ) in camera.bool_settings
    assert (
        _Feature.LX_BOOL_ENABLE_3D_AMP_STREAM,
        False,
    ) in camera.bool_settings
    assert (_Feature.LX_BOOL_ENABLE_SYNC_FRAME, True) in camera.bool_settings
    assert camera.int_settings == [
        (_Feature.LX_INT_ALGORITHM_MODE, _AlgorithmMode.MODE_ALL_OFF),
        (_Feature.LX_INT_RGBD_ALIGN_MODE, _AlignMode.DEPTH_TO_RGB)
    ]
    assert frame.intrinsics.fx == pytest.approx(100.0)
    assert frame.timestamp_ns == 123_456_000
    np.testing.assert_array_equal(frame.rgb_bgr[0, 0], [30, 20, 10])
    np.testing.assert_allclose(frame.depth_m, 1.25)
    assert source.stats.depth_fps == pytest.approx(15.0)
    assert source.stats.rgb_fps == pytest.approx(14.5)
    assert source.stats.frame_id_mismatch_count == 0
    assert source.stats.frame_id_unavailable_count == 1

    source.close()
    assert camera.stopped
    assert camera.closed


def test_lx_camera_source_skips_mismatched_extended_frame_ids(
    tmp_path: Path,
) -> None:
    camera = _FakeCamera("unused")
    extended_frames = [
        _FrameExtendInfo(depth_frame_id=40, rgb_frame_id=41),
        _FrameExtendInfo(depth_frame_id=41, rgb_frame_id=41),
    ]
    camera.frame_call_count = 0

    def get_frame(_handle):
        selected = extended_frames[camera.frame_call_count]
        camera.frame_call_count += 1
        frame = SimpleNamespace(
            depth_data=SimpleNamespace(
                sensor_timestamp=123_456 + camera.frame_call_count
            ),
            rgb_data=SimpleNamespace(
                sensor_timestamp=123_455 + camera.frame_call_count
            ),
            reserve_data=cast(pointer(selected), c_void_p).value,
        )
        return _State.LX_SUCCESS, frame

    camera.getFrame = get_frame
    source = _source(tmp_path, camera)

    source.start()
    frame = source.read(timeout_ms=100)

    assert camera.frame_call_count == 2
    assert frame.frame_number == 0
    assert frame.timestamp_ns == 123_458_000
    assert source.stats.delivered_frame_count == 1
    assert source.stats.transient_frame_error_count == 1
    assert source.stats.frame_id_mismatch_count == 1
    assert source.stats.frame_id_unavailable_count == 0
    assert source.stats.last_depth_frame_id == 41
    assert source.stats.last_rgb_frame_id == 41

    source.close()


def test_lx_camera_source_rejects_unaligned_runtime_shapes(
    tmp_path: Path,
) -> None:
    camera = _FakeCamera("unused", depth_width=2)
    source = _source(tmp_path, camera)

    with pytest.raises(LxCameraSDKError, match="not aligned"):
        source.start()

    assert camera.closed
