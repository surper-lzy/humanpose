#!/usr/bin/env python3
"""Pick placement measurements on a true Gaussian RGB render."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np

from rgbd_avatar.avatar import SMPLSequenceCache
from rgbd_avatar.pipeline.scene_alignment import (
    DEFAULT_SCENE_ROOT,
    DEFAULT_SMPL_CACHE,
    write_alignment,
)
from rgbd_avatar.scene import (
    GaussianAlignmentView,
    ManualScenePlacement,
    first_avatar_ground_anchor,
    fit_ground_plane_robust,
)


LOGGER = logging.getLogger("pick_3dgs_alignment_view")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--known-distance-m", type=float, required=True)
    parser.add_argument(
        "--known-length-direction",
        choices=("any", "vertical", "horizontal"),
        default="any",
        help=(
            "Expected direction relative to the fitted floor. Use 'vertical' "
            "when the known distance is an object's height."
        ),
    )
    parser.add_argument("--scene-root", type=Path, default=DEFAULT_SCENE_ROOT)
    parser.add_argument("--scene-ply", type=Path, default=None)
    parser.add_argument("--smpl-cache", type=Path, default=DEFAULT_SMPL_CACHE)
    parser.add_argument(
        "--anchor-mode",
        choices=("feet", "pelvis", "origin"),
        default="feet",
    )
    parser.add_argument("--patch-radius", type=int, default=2)
    parser.add_argument("--minimum-alpha", type=float, default=0.05)
    parser.add_argument(
        "--description",
        default="Placement picked from true Gaussian RGB+expected-depth",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _draw_picks(image_bgr: np.ndarray, picks: list[tuple[int, int]]) -> np.ndarray:
    import cv2

    canvas = image_bgr.copy()
    for index, (x, y) in enumerate(picks, start=1):
        cv2.circle(canvas, (x, y), 7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            str(index),
            (x + 9, y - 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return canvas


def _collect_pixels(
    view: GaussianAlignmentView,
    *,
    title: str,
    instruction: str,
    minimum: int,
    maximum: int | None,
    patch_radius: int,
    minimum_alpha: float,
) -> tuple[list[tuple[int, int]], list[np.ndarray]]:
    import cv2

    print("\n" + instruction)
    print("左键选择，右键撤销，R 清空，Enter 确认，Esc 取消。")
    source = cv2.cvtColor(view.rgb_uint8, cv2.COLOR_RGB2BGR)
    picks: list[tuple[int, int]] = []
    points: list[np.ndarray] = []

    def callback(event: int, x: int, y: int, _: int, __: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            if maximum is not None and len(picks) >= maximum:
                print(f"当前阶段最多选择 {maximum} 个点；右键可撤销。")
                return
            try:
                point, depth, alpha = view.unproject_pixel(
                    (x, y),
                    patch_radius=patch_radius,
                    minimum_alpha=minimum_alpha,
                )
            except ValueError as error:
                print(f"拒绝像素 {(x, y)}：{error}")
                return
            picks.append((x, y))
            points.append(point)
            print(
                f"pick[{len(picks)}] pixel={(x, y)} depth_g={depth:.5f} "
                f"alpha={alpha:.3f} point_g={point.tolist()}"
            )
        elif event == cv2.EVENT_RBUTTONDOWN and picks:
            picks.pop()
            points.pop()
            print("撤销最后一个点。")

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, view.width, view.height)
    cv2.setMouseCallback(title, callback)
    while True:
        cv2.imshow(title, _draw_picks(source, picks))
        key = cv2.waitKey(20) & 0xFF
        if key in (10, 13):
            if len(picks) < minimum:
                print(f"当前只有 {len(picks)} 个点，至少需要 {minimum} 个。")
                continue
            break
        if key == 27:
            cv2.destroyWindow(title)
            raise RuntimeError("Gaussian alignment picking was cancelled.")
        if key in (ord("r"), ord("R")):
            picks.clear()
            points.clear()
    cv2.destroyWindow(title)
    return picks, points


def _save_annotated_picks(
    view: GaussianAlignmentView,
    groups: list[tuple[str, list[tuple[int, int]], tuple[int, int, int]]],
    output: Path,
) -> None:
    import cv2

    canvas = cv2.cvtColor(view.rgb_uint8, cv2.COLOR_RGB2BGR)
    for label, pixels, color in groups:
        for index, (x, y) in enumerate(pixels, start=1):
            cv2.circle(canvas, (x, y), 7, color, 2, cv2.LINE_AA)
            cv2.putText(
                canvas,
                f"{label}{index}",
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), canvas):
        raise RuntimeError(f"Could not write annotated picks: {output}")


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    try:
        if not np.isfinite(args.known_distance_m) or args.known_distance_m <= 0:
            raise ValueError("--known-distance-m must be finite and positive.")
        if args.patch_radius < 0:
            raise ValueError("--patch-radius must be non-negative.")
        if not np.isfinite(args.minimum_alpha) or not 0 <= args.minimum_alpha <= 1:
            raise ValueError("--minimum-alpha must lie in [0,1].")
        scene_root = args.scene_root.expanduser().resolve()
        scene_ply = (
            args.scene_ply.expanduser().resolve()
            if args.scene_ply is not None
            else scene_root / "point_cloud.ply"
        )
        output = (
            args.output.expanduser().resolve()
            if args.output is not None
            else scene_root / "scene_alignment.json"
        )
        if output.exists() and not args.overwrite:
            raise FileExistsError(
                f"Refusing to overwrite {output}; pass --overwrite to replace it."
            )
        view_path = args.view.expanduser().resolve()
        view = GaussianAlignmentView.load(view_path)
        LOGGER.info(
            "Loaded true Gaussian view %s: camera=%s size=%dx%d",
            view_path,
            view.camera_name,
            view.width,
            view.height,
        )

        length_pixels, length_points = _collect_pixels(
            view,
            title="1/4 Known metric length",
            instruction="阶段 1/4：在清晰的高斯画面上选择已知真实距离的两个端点。",
            minimum=2,
            maximum=2,
            patch_radius=args.patch_radius,
            minimum_alpha=args.minimum_alpha,
        )
        ground_pixels, ground_points = _collect_pixels(
            view,
            title="2/4 Ground plane",
            instruction=(
                "阶段 2/4：只在可见地面上选择至少五个大范围分散的点；"
                "不要点击机器人、立柱或墙面。"
            ),
            minimum=5,
            maximum=None,
            patch_radius=args.patch_radius,
            minimum_alpha=args.minimum_alpha,
        )
        up_pixels, up_points = _collect_pixels(
            view,
            title="3/4 Above-ground reference",
            instruction="阶段 3/4：选择一个明确位于地面上方物体表面的点。",
            minimum=1,
            maximum=1,
            patch_radius=args.patch_radius,
            minimum_alpha=args.minimum_alpha,
        )
        placement_pixels, placement_points = _collect_pixels(
            view,
            title="4/4 Spawn and forward",
            instruction="阶段 4/4：先点击地面出生点，再点击地面上的面朝方向点。",
            minimum=2,
            maximum=2,
            patch_radius=args.patch_radius,
            minimum_alpha=args.minimum_alpha,
        )

        smpl_cache_path = args.smpl_cache.expanduser().resolve()
        cache = SMPLSequenceCache.load(smpl_cache_path)
        anchor = first_avatar_ground_anchor(cache, mode=args.anchor_mode)
        scene_length_g = float(
            np.linalg.norm(length_points[1] - length_points[0])
        )
        scale_g_per_m = scene_length_g / args.known_distance_m
        ground_normal, ground_offset, _, ground_inliers = (
            fit_ground_plane_robust(
                np.asarray(ground_points),
                up_reference_g=up_points[0],
                scale_g_per_m=scale_g_per_m,
            )
        )
        depth_placement_points = np.asarray(placement_points)
        placement_points = [
            view.intersect_pixel_with_plane(
                pixel,
                plane_normal_g=ground_normal,
                plane_offset_g=ground_offset,
            )
            for pixel in placement_pixels
        ]
        LOGGER.info(
            "Ground consensus uses %d/%d picks; spawn/forward use pixel-ray "
            "intersections with that plane",
            int(np.count_nonzero(ground_inliers)),
            len(ground_inliers),
        )
        placement = ManualScenePlacement(
            known_point_a_g=length_points[0],
            known_point_b_g=length_points[1],
            known_distance_m=args.known_distance_m,
            ground_points_g=np.asarray(ground_points),
            up_reference_g=up_points[0],
            spawn_point_g=placement_points[0],
            forward_point_g=placement_points[1],
            avatar_anchor_w_m=anchor,
            known_length_direction=args.known_length_direction,
            description=args.description,
        )
        annotated_groups = [
            ("L", length_pixels, (0, 255, 255)),
            ("G", ground_pixels, (0, 255, 0)),
            ("U", up_pixels, (255, 255, 0)),
            ("P", placement_pixels, (255, 0, 255)),
        ]
        attempt_annotated = output.with_name(
            output.stem + "_attempt_picks.png"
        )
        _save_annotated_picks(view, annotated_groups, attempt_annotated)
        LOGGER.info("Saved current pick attempt to %s", attempt_annotated)
        write_alignment(
            placement,
            output_path=output,
            scene_root=scene_root,
            scene_ply=scene_ply,
            smpl_cache_path=smpl_cache_path,
            overwrite=args.overwrite,
            extra_metadata={
                "selection_mode": "true_gaussian_rgb_expected_depth",
                "selection_view": str(view_path),
                "selection_camera": view.camera_name,
                "ground_point_inliers": ground_inliers.tolist(),
                "spawn_forward_point_source": "pixel_ray_ground_plane_intersection",
                "spawn_forward_expected_depth_points_g": (
                    depth_placement_points.tolist()
                ),
                "selection_pixels": {
                    "known_length": length_pixels,
                    "ground": ground_pixels,
                    "up_reference": up_pixels,
                    "spawn_and_forward": placement_pixels,
                },
                "avatar_anchor_mode": args.anchor_mode,
            },
        )
        annotated = output.with_name(output.stem + "_picks.png")
        _save_annotated_picks(
            view,
            annotated_groups,
            annotated,
        )
        LOGGER.info("Saved annotated picks to %s", annotated)
        return 0
    except Exception as error:
        LOGGER.error("%s", error)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
