"""Depth sampling, deprojection, and 3D pose recovery."""

from .deprojection import (
    deproject_pixel,
    depth_to_organized_point_cloud,
)
from .depth_connected import recover_pose3d_from_depth_connected
from .io import load_depth_m
from .pointcloud_recovery import (
    PointCloudRecoveryConfig,
    PointCloudRecoveryResult,
    recover_pose3d_from_point_cloud,
)
from .ground_plane import (
    GroundPlaneConfig,
    GroundPlaneEstimate,
    fit_ground_plane_ransac,
    sample_ground_candidates,
)
from .recovery import recover_pose3d
from .sampling import DepthSample, sample_joint_depth

__all__ = [
    "DepthSample",
    "GroundPlaneConfig",
    "GroundPlaneEstimate",
    "PointCloudRecoveryConfig",
    "PointCloudRecoveryResult",
    "deproject_pixel",
    "depth_to_organized_point_cloud",
    "load_depth_m",
    "fit_ground_plane_ransac",
    "recover_pose3d",
    "recover_pose3d_from_depth_connected",
    "recover_pose3d_from_point_cloud",
    "sample_ground_candidates",
    "sample_joint_depth",
]
