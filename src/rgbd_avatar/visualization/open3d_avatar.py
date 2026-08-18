"""Open3D conversion helpers shared by offline and live avatar viewers."""

from __future__ import annotations

from typing import Any

import numpy as np

from rgbd_avatar.avatar import ProceduralAvatarFrame


def rotation_from_local_z(direction: np.ndarray) -> np.ndarray:
    """Build a proper rotation whose local Z axis follows ``direction``."""

    z_axis = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(z_axis))
    if not np.isfinite(norm) or norm <= 1e-9:
        raise ValueError("Avatar segment direction must be finite and non-zero.")
    z_axis /= norm
    helper = (
        np.array([0.0, 0.0, 1.0])
        if abs(z_axis[2]) < 0.9
        else np.array([1.0, 0.0, 0.0])
    )
    x_axis = np.cross(helper, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    return np.column_stack((x_axis, y_axis, z_axis))


def replace_procedural_avatar_mesh(
    o3d: Any,
    geometry: Any,
    avatar: ProceduralAvatarFrame,
    *,
    sphere_resolution: int = 6,
) -> None:
    """Replace an Open3D triangle mesh with one procedural avatar frame."""

    if sphere_resolution < 3:
        raise ValueError("sphere_resolution must be at least 3.")
    geometry.clear()
    for primitive in avatar.capsules:
        direction = primitive.end_m - primitive.start_m
        resolution = max(8, 2 * sphere_resolution)
        angles = np.linspace(0.0, 2.0 * np.pi, resolution, endpoint=False)
        start_ring = np.column_stack(
            (
                primitive.radius_m * np.cos(angles),
                primitive.radius_m * np.sin(angles),
                np.full(resolution, -0.5 * primitive.length_m),
            )
        )
        end_radius = primitive.resolved_end_radius_m
        end_ring = np.column_stack(
            (
                end_radius * np.cos(angles),
                end_radius * np.sin(angles),
                np.full(resolution, 0.5 * primitive.length_m),
            )
        )
        vertices = np.vstack((start_ring, end_ring, np.zeros((2, 3))))
        vertices[-2, 2] = -0.5 * primitive.length_m
        vertices[-1, 2] = 0.5 * primitive.length_m
        start_center = 2 * resolution
        end_center = start_center + 1
        triangles: list[tuple[int, int, int]] = []
        for index in range(resolution):
            following = (index + 1) % resolution
            triangles.extend(
                (
                    (index, following, resolution + following),
                    (index, resolution + following, resolution + index),
                    (start_center, following, index),
                    (end_center, resolution + index, resolution + following),
                )
            )
        segment = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(vertices),
            triangles=o3d.utility.Vector3iVector(
                np.asarray(triangles, dtype=np.int32)
            ),
        )
        segment.rotate(rotation_from_local_z(direction), center=np.zeros(3))
        segment.translate(0.5 * (primitive.start_m + primitive.end_m))
        segment.paint_uniform_color(primitive.color)
        geometry += segment

    for primitive in avatar.ellipsoids:
        ellipsoid = o3d.geometry.TriangleMesh.create_sphere(
            radius=1.0,
            resolution=max(4, sphere_resolution + 2),
        )
        transform = np.eye(4)
        transform[:3, :3] = primitive.rotation @ np.diag(primitive.radii_m)
        transform[:3, 3] = primitive.center_m
        ellipsoid.transform(transform)
        ellipsoid.paint_uniform_color(primitive.color)
        geometry += ellipsoid
    if geometry.has_vertices():
        geometry.compute_vertex_normals()
