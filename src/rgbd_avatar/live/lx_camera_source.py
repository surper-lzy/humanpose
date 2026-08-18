"""Direct RGB-D capture from the MRDVS/Lanxin camera SDK."""

from __future__ import annotations

from ctypes import POINTER, cast
from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
import time
from types import ModuleType
from typing import Any, Literal, Mapping

import cv2
import numpy as np

from rgbd_avatar.camera import CameraIntrinsics

from .models import RGBDFrame


OpenMode = Literal["id", "ip", "sn", "index"]
RGBOrder = Literal["rgb", "bgr"]
TimestampSource = Literal["sensor_us", "host_monotonic"]


class LxCameraSDKError(RuntimeError):
    """Raised when an SDK operation cannot produce a valid RGB-D frame."""


@dataclass(frozen=True)
class LxCameraSourceStats:
    delivered_frame_count: int
    transient_frame_error_count: int
    frame_id_mismatch_count: int
    frame_id_unavailable_count: int
    last_depth_frame_id: int | None
    last_rgb_frame_id: int | None
    last_depth_sensor_timestamp: int | None
    last_rgb_sensor_timestamp: int | None
    depth_fps: float | None
    rgb_fps: float | None


def _load_sdk_module(
    python_wheel: Path | None,
) -> ModuleType:
    try:
        return importlib.import_module("LxCameraSDK")
    except ModuleNotFoundError as first_error:
        if python_wheel is None:
            raise LxCameraSDKError(
                "LxCameraSDK is not installed and no SDK Python wheel was "
                "configured."
            ) from first_error
        if not python_wheel.is_file():
            raise FileNotFoundError(
                f"LxCameraSDK Python wheel not found: {python_wheel}"
            ) from first_error
        wheel_string = str(python_wheel)
        if wheel_string not in sys.path:
            sys.path.insert(0, wheel_string)
        importlib.invalidate_caches()
        try:
            return importlib.import_module("LxCameraSDK")
        except ModuleNotFoundError as second_error:
            raise LxCameraSDKError(
                f"Cannot import LxCameraSDK from {python_wheel}."
            ) from second_error


class LxCameraRGBDSource:
    """Read synchronized depth-to-RGB frames directly from ``libLxCameraApi``.

    The vendor Python wrapper exposes NumPy views over SDK-owned memory. This
    adapter copies both images before returning so a following SDK call cannot
    overwrite a frame while pose inference is still using it.
    """

    def __init__(
        self,
        *,
        library_path: str | Path,
        open_param: str,
        depth_scale: float,
        python_wheel: str | Path | None = None,
        open_mode: OpenMode = "id",
        align_depth_to_rgb: bool = True,
        sync_frame: bool = True,
        require_matching_frame_ids: bool = True,
        enable_amplitude: bool = False,
        disable_builtin_algorithm: bool = True,
        rgb_order: RGBOrder = "rgb",
        timestamp_source: TimestampSource = "sensor_us",
        sdk_module: ModuleType | Any | None = None,
    ) -> None:
        if not open_param:
            raise ValueError("open_param must be non-empty.")
        if depth_scale <= 0:
            raise ValueError("depth_scale must be positive.")
        if open_mode not in ("id", "ip", "sn", "index"):
            raise ValueError("open_mode must be id, ip, sn, or index.")
        if rgb_order not in ("rgb", "bgr"):
            raise ValueError("rgb_order must be rgb or bgr.")
        if timestamp_source not in ("sensor_us", "host_monotonic"):
            raise ValueError(
                "timestamp_source must be sensor_us or host_monotonic."
            )
        self.library_path = Path(library_path).expanduser().resolve()
        self.python_wheel = (
            Path(python_wheel).expanduser().resolve()
            if python_wheel is not None
            else None
        )
        self.open_param = str(open_param)
        self.open_mode = open_mode
        self.depth_scale = float(depth_scale)
        self.align_depth_to_rgb = bool(align_depth_to_rgb)
        self.sync_frame = bool(sync_frame)
        self.require_matching_frame_ids = bool(require_matching_frame_ids)
        self.enable_amplitude = bool(enable_amplitude)
        self.disable_builtin_algorithm = bool(disable_builtin_algorithm)
        self.rgb_order = rgb_order
        self.timestamp_source = timestamp_source
        self._sdk = sdk_module
        self._camera: Any | None = None
        self._handle: Any | None = None
        self._intrinsics: CameraIntrinsics | None = None
        self._started = False
        self._closed = False
        self._stream_started = False
        self._delivered_frame_count = 0
        self._transient_frame_error_count = 0
        self._frame_id_mismatch_count = 0
        self._frame_id_unavailable_count = 0
        self._last_depth_frame_id: int | None = None
        self._last_rgb_frame_id: int | None = None
        self._last_depth_sensor_timestamp: int | None = None
        self._last_rgb_sensor_timestamp: int | None = None
        self._depth_fps: float | None = None
        self._rgb_fps: float | None = None
        self._last_fps_query_s = 0.0

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any],
        *,
        depth_scale: float,
    ) -> "LxCameraRGBDSource":
        return cls(
            library_path=mapping["library_path"],
            python_wheel=mapping.get("python_wheel"),
            open_mode=mapping.get("open_mode", "id"),
            open_param=str(mapping["open_param"]),
            depth_scale=depth_scale,
            align_depth_to_rgb=bool(mapping.get("align_depth_to_rgb", True)),
            sync_frame=bool(mapping.get("sync_frame", True)),
            require_matching_frame_ids=bool(
                mapping.get("require_matching_frame_ids", True)
            ),
            enable_amplitude=bool(mapping.get("enable_amplitude", False)),
            disable_builtin_algorithm=bool(
                mapping.get("disable_builtin_algorithm", True)
            ),
            rgb_order=mapping.get("rgb_order", "rgb"),
            timestamp_source=mapping.get("timestamp_source", "sensor_us"),
        )

    @property
    def source_id(self) -> str:
        return f"lx-sdk:{self.open_mode}:{self.open_param}"

    @property
    def intrinsics(self) -> CameraIntrinsics | None:
        return self._intrinsics

    @property
    def stats(self) -> LxCameraSourceStats:
        return LxCameraSourceStats(
            delivered_frame_count=self._delivered_frame_count,
            transient_frame_error_count=self._transient_frame_error_count,
            frame_id_mismatch_count=self._frame_id_mismatch_count,
            frame_id_unavailable_count=self._frame_id_unavailable_count,
            last_depth_frame_id=self._last_depth_frame_id,
            last_rgb_frame_id=self._last_rgb_frame_id,
            last_depth_sensor_timestamp=self._last_depth_sensor_timestamp,
            last_rgb_sensor_timestamp=self._last_rgb_sensor_timestamp,
            depth_fps=self._depth_fps,
            rgb_fps=self._rgb_fps,
        )

    def _state_name(self, state: Any) -> str:
        return getattr(state, "name", str(state))

    def _error_detail(self, state: Any) -> str:
        if self._camera is None:
            return self._state_name(state)
        try:
            return str(self._camera.DcGetErrorString(state))
        except Exception:
            return self._state_name(state)

    def _require_success(self, operation: str, state: Any) -> None:
        assert self._sdk is not None
        if state != self._sdk.LX_STATE.LX_SUCCESS:
            raise LxCameraSDKError(
                f"{operation} failed: {self._error_detail(state)} "
                f"({self._state_name(state)})."
            )

    def _set_bool(self, feature_name: str, value: bool) -> None:
        assert self._camera is not None and self._handle is not None
        feature = getattr(self._sdk.LX_CAMERA_FEATURE, feature_name)
        state = self._camera.DcSetBoolValue(self._handle, feature, value)
        self._require_success(f"set {feature_name}={value}", state)

    def _get_int(self, feature_name: str) -> int:
        assert self._camera is not None and self._handle is not None
        feature = getattr(self._sdk.LX_CAMERA_FEATURE, feature_name)
        state, value = self._camera.DcGetIntValue(self._handle, feature)
        self._require_success(f"get {feature_name}", state)
        return int(value.cur_value)

    def _query_intrinsics(self) -> CameraIntrinsics:
        assert self._camera is not None and self._handle is not None
        rgb_width = self._get_int("LX_INT_2D_IMAGE_WIDTH")
        rgb_height = self._get_int("LX_INT_2D_IMAGE_HEIGHT")
        depth_width = self._get_int("LX_INT_3D_IMAGE_WIDTH")
        depth_height = self._get_int("LX_INT_3D_IMAGE_HEIGHT")
        if (rgb_width, rgb_height) != (depth_width, depth_height):
            raise LxCameraSDKError(
                "SDK depth is not aligned to RGB: "
                f"RGB={rgb_width}x{rgb_height}, "
                f"depth={depth_width}x{depth_height}."
            )
        state, values, _distortion = self._camera.get2DIntricParam(
            self._handle
        )
        self._require_success("get aligned RGB intrinsics", state)
        if values is None or len(values) < 4:
            raise LxCameraSDKError("SDK returned invalid RGB intrinsics.")
        return CameraIntrinsics(
            fx=float(values[0]),
            fy=float(values[1]),
            cx=float(values[2]),
            cy=float(values[3]),
            width=rgb_width,
            height=rgb_height,
        )

    def start(self) -> None:
        if self._started and not self._closed:
            return
        if not self.library_path.is_file():
            raise FileNotFoundError(
                f"LxCamera API library not found: {self.library_path}"
            )
        self._sdk = self._sdk or _load_sdk_module(self.python_wheel)
        self._camera = self._sdk.LxCamera(str(self.library_path))
        open_modes = {
            "id": self._sdk.LX_OPEN_MODE.OPEN_BY_ID,
            "ip": self._sdk.LX_OPEN_MODE.OPEN_BY_IP,
            "sn": self._sdk.LX_OPEN_MODE.OPEN_BY_SN,
            "index": self._sdk.LX_OPEN_MODE.OPEN_BY_INDEX,
        }
        # The vendor samples enumerate before OPEN_BY_ID/SN/INDEX. Without
        # this discovery step a reachable GigE camera may be reported as
        # LX_E_DEVICE_NOT_FOUND even when its IP responds normally.
        if self.open_mode != "ip":
            state, _device_list, device_count = self._camera.DcGetDeviceList()
            self._require_success("enumerate cameras", state)
            if int(device_count) <= 0:
                raise LxCameraSDKError(
                    "SDK discovery found no camera. Check camera power, the "
                    "GigE cable, and the host 192.168.1.x interface."
                )
        state, handle, _device_info = self._camera.DcOpenDevice(
            open_modes[self.open_mode], self.open_param
        )
        if state != self._sdk.LX_STATE.LX_SUCCESS:
            detail = self._error_detail(state)
            self._camera = None
            raise LxCameraSDKError(
                "open camera failed: "
                f"{detail} ({self._state_name(state)}). Exit "
                "LxCameraViewer first because the camera is exclusive."
            )
        self._handle = handle
        self._closed = False
        try:
            self._set_bool("LX_BOOL_ENABLE_2D_STREAM", True)
            self._set_bool("LX_BOOL_ENABLE_3D_DEPTH_STREAM", True)
            self._set_bool(
                "LX_BOOL_ENABLE_3D_AMP_STREAM", self.enable_amplitude
            )
            if self.disable_builtin_algorithm:
                feature = self._sdk.LX_CAMERA_FEATURE.LX_INT_ALGORITHM_MODE
                mode = self._sdk.LX_ALGORITHM_MODE.MODE_ALL_OFF
                state = self._camera.DcSetIntValue(
                    self._handle, feature, mode
                )
                self._require_success("disable built-in algorithm stream", state)
            if self.align_depth_to_rgb:
                feature = self._sdk.LX_CAMERA_FEATURE.LX_INT_RGBD_ALIGN_MODE
                mode = self._sdk.LX_RGBD_ALIGN_MODE.DEPTH_TO_RGB
                state = self._camera.DcSetIntValue(
                    self._handle, feature, mode
                )
                self._require_success("set depth-to-RGB alignment", state)
            self._set_bool("LX_BOOL_ENABLE_SYNC_FRAME", self.sync_frame)
            self._intrinsics = self._query_intrinsics()
            state = self._camera.DcStartStream(self._handle)
            self._require_success("start RGB-D stream", state)
            self._stream_started = True
            self._started = True
        except Exception:
            self.close()
            raise

    def _is_transient_state(self, state: Any) -> bool:
        assert self._sdk is not None
        transient_names = (
            "LX_E_TIME_OUT",
            "LX_E_FRAME_ID_NOT_MATCH",
            "LX_E_FRAME_MULTI_MACHINE",
        )
        return any(
            hasattr(self._sdk.LX_STATE, name)
            and state == getattr(self._sdk.LX_STATE, name)
            for name in transient_names
        )

    def _query_fps_if_due(self) -> None:
        now = time.monotonic()
        if now - self._last_fps_query_s < 1.0:
            return
        assert self._camera is not None and self._handle is not None
        for feature_name, attribute in (
            ("LX_FLOAT_3D_DEPTH_FPS", "_depth_fps"),
            ("LX_FLOAT_2D_IMAGE_FPS", "_rgb_fps"),
        ):
            feature = getattr(self._sdk.LX_CAMERA_FEATURE, feature_name)
            state, value = self._camera.DcGetFloatValue(
                self._handle, feature
            )
            if state == self._sdk.LX_STATE.LX_SUCCESS:
                setattr(self, attribute, float(value.cur_value))
        self._last_fps_query_s = now

    def _frame_timestamp_ns(self, depth_timestamp: int) -> int:
        if self.timestamp_source == "sensor_us" and depth_timestamp > 0:
            return depth_timestamp * 1_000
        return time.monotonic_ns()

    def _frame_ids(self, frame_info: Any) -> tuple[int, int] | None:
        """Read SDK extension IDs without requiring them on older devices."""
        reserve_data = getattr(frame_info, "reserve_data", None)
        if not reserve_data:
            return None
        if hasattr(reserve_data, "depth_frame_id") and hasattr(
            reserve_data, "rgb_frame_id"
        ):
            extended = reserve_data
        else:
            frame_extend_type = getattr(self._sdk, "FrameExtendInfo", None)
            if frame_extend_type is None:
                return None
            try:
                extended = cast(
                    reserve_data, POINTER(frame_extend_type)
                ).contents
            except (TypeError, ValueError):
                return None
        return int(extended.depth_frame_id), int(extended.rgb_frame_id)

    def read(self, timeout_ms: int = 1000) -> RGBDFrame:
        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive.")
        if not self._started or self._closed:
            raise RuntimeError(
                "LxCameraRGBDSource must be started before read()."
            )
        assert self._camera is not None and self._handle is not None
        assert self._intrinsics is not None and self._sdk is not None
        deadline = time.monotonic() + timeout_ms / 1000.0
        while not self._closed:
            state, frame_info = self._camera.getFrame(self._handle)
            if state != self._sdk.LX_STATE.LX_SUCCESS:
                if self._is_transient_state(state):
                    self._transient_frame_error_count += 1
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            "No synchronized RGB-D SDK frame before timeout."
                        )
                    continue
                self._require_success("get synchronized RGB-D frame", state)
            if frame_info is None:
                raise LxCameraSDKError("SDK returned an empty frame pointer.")
            depth_timestamp = int(frame_info.depth_data.sensor_timestamp)
            rgb_timestamp = int(frame_info.rgb_data.sensor_timestamp)
            self._last_depth_sensor_timestamp = depth_timestamp
            self._last_rgb_sensor_timestamp = rgb_timestamp
            frame_ids = self._frame_ids(frame_info)
            if frame_ids is None:
                if self.sync_frame and self.require_matching_frame_ids:
                    self._frame_id_unavailable_count += 1
            else:
                depth_frame_id, rgb_frame_id = frame_ids
                self._last_depth_frame_id = depth_frame_id
                self._last_rgb_frame_id = rgb_frame_id
                if (
                    self.sync_frame
                    and self.require_matching_frame_ids
                    and depth_frame_id != rgb_frame_id
                ):
                    self._frame_id_mismatch_count += 1
                    self._transient_frame_error_count += 1
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            "No matching RGB-D SDK frame IDs before timeout."
                        )
                    continue
            rgb_state, rgb_view = self._camera.getRGBImage(frame_info)
            self._require_success("get RGB image", rgb_state)
            depth_state, depth_view = self._camera.getDepthImage(frame_info)
            self._require_success("get depth image", depth_state)

            # Both are views of SDK-owned buffers and must be copied now.
            rgb = np.asarray(rgb_view).copy()
            depth_raw = np.asarray(depth_view).copy()
            if rgb.dtype != np.uint8 or rgb.shape != (
                self._intrinsics.height,
                self._intrinsics.width,
                3,
            ):
                raise LxCameraSDKError(
                    f"Unexpected RGB frame: shape={rgb.shape}, dtype={rgb.dtype}."
                )
            if depth_raw.shape != (
                self._intrinsics.height,
                self._intrinsics.width,
            ):
                raise LxCameraSDKError(
                    "Unexpected aligned depth frame: "
                    f"shape={depth_raw.shape}."
                )
            if not np.issubdtype(depth_raw.dtype, np.number):
                raise LxCameraSDKError(
                    f"Unexpected depth dtype: {depth_raw.dtype}."
                )
            rgb_bgr = (
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                if self.rgb_order == "rgb"
                else rgb
            )
            depth_m = depth_raw.astype(np.float32) * self.depth_scale
            self._query_fps_if_due()
            frame = RGBDFrame(
                rgb_bgr=rgb_bgr,
                depth_m=depth_m,
                intrinsics=self._intrinsics,
                timestamp_ns=self._frame_timestamp_ns(depth_timestamp),
                frame_number=self._delivered_frame_count,
                source_id=self.source_id,
            )
            self._delivered_frame_count += 1
            return frame
        raise RuntimeError("LxCameraRGBDSource is closed.")

    def close(self) -> None:
        camera = self._camera
        handle = self._handle
        # Clear Python ownership before returning so the ctypes/CDLL wrapper is
        # finalized deterministically on the same thread that owns the stream,
        # rather than during interpreter shutdown beside CUDA native libraries.
        self._camera = None
        self._handle = None
        if camera is not None and handle is not None:
            if self._stream_started:
                try:
                    camera.DcStopStream(handle)
                except Exception:
                    pass
            try:
                camera.DcCloseDevice(handle)
            except Exception:
                pass
        self._stream_started = False
        self._started = False
        self._closed = True
