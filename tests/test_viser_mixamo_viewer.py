"""Tests for the browser-based Mixamo viewer (Filament offscreen renderer)."""

import gc

import numpy as np
import pytest

from rgbd_avatar.avatar import MixamoSequenceCache
from rgbd_avatar.visualization.viser_mixamo_viewer import MixamoFilamentRenderer


def _camera_c2w(center: np.ndarray, distance: float) -> np.ndarray:
    """OpenCV-convention camera (viser/nerfview c2w): +Z forward, +Y down."""
    eye = center + np.array([0.0, -distance * 0.8, distance * 0.5])
    forward = center - eye
    forward /= np.linalg.norm(forward)
    up_ref = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up_ref)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    c2w = np.eye(4)
    c2w[:3, :3] = np.column_stack([right, down, forward])
    c2w[:3, 3] = eye
    return c2w


def _small_textured_cache() -> MixamoSequenceCache:
    """A two-frame animated cube with a multicolor texture plus one NaN frame."""
    import cv2

    vertices = np.array(
        [
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2],
            [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0],
        ],
        dtype=np.int32,
    )
    quad_uv = np.array(
        [[0, 0], [1, 0], [1, 1], [0, 0], [1, 1], [0, 1]],
        dtype=np.float32,
    )
    triangle_uvs = np.tile(quad_uv, (len(faces) // 2, 1)).reshape(
        len(faces), 3, 2
    )
    # Multicolor texture (a checker would collapse to two colors under the
    # unlit shader and defeat the color-diversity assertions).
    rng = np.random.default_rng(0)
    image_bgr = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    ok, png = cv2.imencode(".png", image_bgr)
    assert ok
    frame_0 = vertices.copy()
    frame_1 = vertices + np.array([0.3, 0.0, 0.0], dtype=np.float32)
    frame_nan = np.full_like(vertices, np.nan, dtype=np.float32)
    return MixamoSequenceCache(
        frame_indices=np.array([0, 1, 2]),
        present=np.array([True, True, False]),
        vertices_display_m=np.stack([frame_0, frame_1, frame_nan]),
        faces=faces,
        triangle_uvs=triangle_uvs,
        diffuse_png=np.frombuffer(png, dtype=np.uint8),
        bone_names=("Hips",),
        bone_global_m=np.repeat(
            np.eye(4, dtype=np.float32)[None, None], 3, axis=0
        ),
        scale=1.0,
        metadata={"pose_layer": "constrained"},
    )


@pytest.fixture()
def renderer():
    """One Filament engine per test; destroyed before the next engine.

    This Open3D build segfaults when a second OffscreenRenderer is created
    while the first is still alive, so each test gets its own renderer and
    drops it (refcount + gc) afterwards.
    """
    instance = MixamoFilamentRenderer(_small_textured_cache(), 160, 120)
    yield instance
    del instance
    gc.collect()


def test_filament_renderer_renders_textured_frames(renderer) -> None:
    c2w = _camera_c2w(center=np.array([0.5, 0.5, 0.5]), distance=3.0)

    image = renderer.render(0, c2w, 0.6, 160, 120)
    assert image.shape == (120, 160, 3)
    assert image.dtype == np.uint8
    # Textured + lit cube: far more colors than a flat-shaded box.
    assert len(np.unique(image.reshape(-1, 3), axis=0)) > 50

    # Resolution change goes through set_view_size().
    image = renderer.render(0, c2w, 0.6, 200, 150)
    assert image.shape == (150, 200, 3)


def test_filament_renderer_animates_and_holds_absent_frames(renderer) -> None:
    c2w = _camera_c2w(center=np.array([0.5, 0.5, 0.5]), distance=3.0)

    frame_0 = renderer.render(0, c2w, 0.6, 160, 120)
    frame_1 = renderer.render(1, c2w, 0.6, 160, 120)
    assert np.abs(frame_0.astype(int) - frame_1.astype(int)).sum() > 1000

    # NaN frame holds the previous valid pose instead of uploading NaNs.
    # The Filament output has small pixel-level nondeterminism (dithering /
    # buffer reuse), so compare with a tolerance and against the previous
    # valid frame's content.
    held = renderer.render(2, c2w, 0.6, 160, 120)
    assert np.abs(held.astype(int) - frame_1.astype(int)).max() <= 4
    assert np.abs(held.astype(int) - frame_0.astype(int)).sum() > 1000


def test_filament_renderer_first_frame_absent_uses_first_present(renderer) -> None:
    c2w = _camera_c2w(center=np.array([0.5, 0.5, 0.5]), distance=3.0)
    # A fresh renderer whose first request is the absent frame must still
    # draw the first present pose instead of crashing on NaN geometry.
    image = renderer.render(2, c2w, 0.6, 160, 120)
    assert image.shape == (120, 160, 3)
    assert len(np.unique(image.reshape(-1, 3), axis=0)) > 50
