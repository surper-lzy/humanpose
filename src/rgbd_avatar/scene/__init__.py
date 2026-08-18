"""Static-scene placement contracts for the metric avatar world."""

from .alignment import (
    ManualScenePlacement,
    SceneAlignment,
    build_manual_scene_alignment,
    first_avatar_ground_anchor,
    fit_ground_plane,
    fit_ground_plane_robust,
)
from .colmap import (
    ColmapCamera,
    ColmapImage,
    load_sparse_cameras,
    quaternion_wxyz_to_rotation,
    read_cameras_binary,
    read_images_binary,
)
from .gaussian_view import GaussianAlignmentView

__all__ = [
    "ManualScenePlacement",
    "SceneAlignment",
    "build_manual_scene_alignment",
    "first_avatar_ground_anchor",
    "fit_ground_plane",
    "fit_ground_plane_robust",
    "ColmapCamera",
    "ColmapImage",
    "load_sparse_cameras",
    "quaternion_wxyz_to_rotation",
    "read_cameras_binary",
    "read_images_binary",
    "GaussianAlignmentView",
]
