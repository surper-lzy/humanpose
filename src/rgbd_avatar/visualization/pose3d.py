"""Static visualizations for metric point clouds and 3D skeletons."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from rgbd_avatar.pose import HALPE26_LINKS, Pose2D, Pose3D
from rgbd_avatar.pose.visualization import draw_pose


def draw_pose_depths(
    image_bgr: np.ndarray,
    pose2d: Pose2D,
    pose3d: Pose3D,
    score_threshold: float = 0.3,
) -> np.ndarray:
    canvas = draw_pose(image_bgr, pose2d, score_threshold)
    for index, point in enumerate(pose2d.keypoints):
        if not pose3d.valid[index]:
            continue
        x, y = np.rint(point).astype(int)
        cv2.putText(
            canvas,
            f"{pose3d.depth_m[index]:.2f}m",
            (x + 5, y + 11),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.30,
            (255, 100, 0),
            1,
            cv2.LINE_AA,
        )
    return canvas


def save_pose3d_scene(
    output_path: str | Path,
    organized_points_m: np.ndarray,
    rgb: np.ndarray,
    pose2d: Pose2D,
    pose3d: Pose3D,
    pixel_stride: int = 3,
) -> None:
    """Save a cropped point-cloud and skeleton diagnostic as a PNG."""
    # Matplotlib is only needed by this static export path.  Importing it
    # lazily keeps the interactive Open3D player free of font/cache startup.
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    output = Path(output_path)
    x1, y1, x2, y2 = np.rint(pose2d.bbox_xyxy).astype(int)
    height, width = organized_points_m.shape[:2]
    margin = 30
    x1, x2 = max(0, x1 - margin), min(width, x2 + margin)
    y1, y2 = max(0, y1 - margin), min(height, y2 + margin)

    points = organized_points_m[
        y1:y2:pixel_stride, x1:x2:pixel_stride
    ].reshape(-1, 3)
    colors = rgb[y1:y2:pixel_stride, x1:x2:pixel_stride].reshape(-1, 3)
    valid_points = np.isfinite(points).all(axis=1)
    points = points[valid_points]
    colors = colors[valid_points].astype(np.float32) / 255.0

    figure = plt.figure(figsize=(9, 8))
    axis = figure.add_subplot(111, projection="3d")
    axis.scatter(
        points[:, 0],
        points[:, 2],
        -points[:, 1],
        c=colors,
        s=0.4,
        linewidths=0,
        alpha=0.5,
    )

    for start, end in HALPE26_LINKS:
        if not pose3d.valid[start] or not pose3d.valid[end]:
            continue
        segment = pose3d.joints_m[[start, end]]
        axis.plot(
            segment[:, 0],
            segment[:, 2],
            -segment[:, 1],
            color="red",
            linewidth=2.5,
        )
    valid_joints = pose3d.joints_m[pose3d.valid]
    axis.scatter(
        valid_joints[:, 0],
        valid_joints[:, 2],
        -valid_joints[:, 1],
        c="yellow",
        edgecolors="black",
        s=25,
        depthshade=False,
    )

    axis.set_xlabel("X right (m)")
    axis.set_ylabel("Z forward (m)")
    axis.set_zlabel("-Y up (m, visualization only)")
    axis.set_title(
        f"Metric RGB-D pose: {np.count_nonzero(pose3d.valid)}/26 joints"
    )
    axis.view_init(elev=12, azim=-75)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)
