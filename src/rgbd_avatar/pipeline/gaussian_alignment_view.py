#!/usr/bin/env python3
"""Render a true 3DGS RGB+depth view for visually clear alignment picking."""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from rgbd_avatar.pipeline.scene_alignment import DEFAULT_SCENE_ROOT
from rgbd_avatar.scene import GaussianAlignmentView, load_sparse_cameras


LOGGER = logging.getLogger("render_3dgs_alignment_view")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-root", type=Path, default=DEFAULT_SCENE_ROOT)
    parser.add_argument(
        "--scene-ply",
        type=Path,
        default=None,
        help="Default: point_cloud.ply under --scene-root.",
    )
    parser.add_argument(
        "--sparse-dir",
        type=Path,
        default=None,
        help="Default: sparse/0 under --scene-root.",
    )
    camera = parser.add_mutually_exclusive_group()
    camera.add_argument(
        "--camera-index",
        type=int,
        default=None,
        help="Zero-based index after sorting COLMAP images by filename; default: middle view.",
    )
    camera.add_argument("--camera-name", type=str, default=None)
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="Print sorted camera indices/names without loading the Gaussian PLY.",
    )
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Default preserves the COLMAP camera aspect ratio.",
    )
    parser.add_argument(
        "--sh-degree",
        type=int,
        default=None,
        help="Default uses every SH coefficient stored in the PLY.",
    )
    parser.add_argument(
        "--background",
        choices=("black", "white"),
        default="black",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output NPZ; a PNG with the same stem is also written.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _indexed_names(names: set[str], prefix: str) -> list[str]:
    indexed: list[tuple[int, str]] = []
    for name in names:
        marker = prefix + "_"
        if not name.startswith(marker):
            continue
        suffix = name[len(marker) :]
        if suffix.isdigit():
            indexed.append((int(suffix), name))
    indexed.sort()
    if indexed and [index for index, _ in indexed] != list(range(len(indexed))):
        raise ValueError(f"PLY {prefix}_* properties are not contiguous from zero.")
    return [name for _, name in indexed]


def load_gaussian_ply_tensors(
    path: Path,
    *,
    device: Any,
    torch: Any,
) -> tuple[dict[str, Any], int]:
    """Load standard GraphDECO fields while ignoring auxiliary ins_feat data."""

    from plyfile import PlyData

    ply = PlyData.read(str(path), mmap="c")
    if "vertex" not in ply:
        raise ValueError(f"Gaussian PLY contains no vertex element: {path}")
    vertex = ply["vertex"].data
    names = set(vertex.dtype.names or ())
    required = {"x", "y", "z", "opacity"}
    missing = sorted(required - names)
    if missing:
        raise ValueError(f"Gaussian PLY missing fields: {missing}")
    dc_names = _indexed_names(names, "f_dc")
    rest_names = _indexed_names(names, "f_rest")
    scale_names = _indexed_names(names, "scale")
    rotation_names = _indexed_names(names, "rot")
    if len(dc_names) != 3 or len(scale_names) != 3 or len(rotation_names) != 4:
        raise ValueError(
            "Gaussian PLY requires f_dc_0..2, scale_0..2, and rot_0..3."
        )
    if len(rest_names) % 3 != 0:
        raise ValueError("Gaussian f_rest_* property count must be divisible by 3.")
    coefficient_count = 1 + len(rest_names) // 3
    root = math.isqrt(coefficient_count)
    if root * root != coefficient_count:
        raise ValueError("Gaussian SH coefficient count must be a square.")
    sh_degree = root - 1

    def stack(field_names: list[str]) -> np.ndarray:
        return np.column_stack(
            [np.asarray(vertex[name], dtype=np.float32) for name in field_names]
        )

    point_count = len(vertex)
    means = stack(["x", "y", "z"])
    sh0 = stack(dc_names)[:, None, :]
    if rest_names:
        shn = stack(rest_names).reshape(point_count, 3, -1).transpose(0, 2, 1)
    else:
        shn = np.empty((point_count, 0, 3), dtype=np.float32)
    raw_scales = stack(scale_names)
    raw_quaternions = stack(rotation_names)
    # Binary PLY vertex rows also contain trailing uint8 RGB, giving scalar
    # field views a 275-byte stride that torch cannot wrap directly.
    raw_opacities = np.ascontiguousarray(
        vertex["opacity"],
        dtype=np.float32,
    )

    tensors = {
        "means": torch.from_numpy(means).to(device=device),
        "quats": torch.from_numpy(raw_quaternions).to(device=device),
        "scales": torch.from_numpy(raw_scales).to(device=device),
        "opacities": torch.from_numpy(raw_opacities).to(device=device),
        "colors": torch.from_numpy(np.concatenate((sh0, shn), axis=1)).to(
            device=device
        ),
    }
    return tensors, sh_degree


def _select_image(images: list[Any], args: argparse.Namespace) -> tuple[int, Any]:
    if args.camera_name is not None:
        matches = [
            (index, image)
            for index, image in enumerate(images)
            if image.name == args.camera_name
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one COLMAP image named {args.camera_name!r}.")
        return matches[0]
    index = len(images) // 2 if args.camera_index is None else args.camera_index
    if not 0 <= index < len(images):
        raise ValueError(f"--camera-index must lie in [0,{len(images) - 1}].")
    return index, images[index]


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    try:
        scene_root = args.scene_root.expanduser().resolve()
        scene_ply = (
            args.scene_ply.expanduser().resolve()
            if args.scene_ply is not None
            else scene_root / "point_cloud.ply"
        )
        sparse_dir = (
            args.sparse_dir.expanduser().resolve()
            if args.sparse_dir is not None
            else scene_root / "sparse/0"
        )
        cameras, images = load_sparse_cameras(sparse_dir)
        if args.list_cameras:
            for index, image in enumerate(images):
                print(f"{index:04d}  {image.name}  camera_id={image.camera_id}")
            return 0
        if args.width <= 0:
            raise ValueError("--width must be positive.")
        camera_index, image = _select_image(images, args)
        camera = cameras[image.camera_id]
        height = (
            int(round(camera.height * args.width / camera.width))
            if args.height is None
            else args.height
        )
        if height <= 0:
            raise ValueError("--height must be positive.")
        if not scene_ply.is_file():
            raise FileNotFoundError(f"Gaussian PLY not found: {scene_ply}")
        output = (
            args.output.expanduser().resolve()
            if args.output is not None
            else scene_root / "alignment_views" / f"{Path(image.name).stem}.npz"
        )
        png_output = output.with_suffix(".png")
        for target in (output, png_output):
            if target.exists() and not args.overwrite:
                raise FileExistsError(
                    f"Refusing to overwrite {target}; pass --overwrite to replace it."
                )

        import torch
        import torch.nn.functional as functional
        from gsplat.rendering import rasterization

        if not torch.cuda.is_available():
            raise RuntimeError("True Gaussian rendering requires a CUDA-visible GPU.")
        device = torch.device("cuda")
        LOGGER.info("Loading %s on %s", scene_ply, device)
        splats, available_sh_degree = load_gaussian_ply_tensors(
            scene_ply,
            device=device,
            torch=torch,
        )
        sh_degree = (
            available_sh_degree if args.sh_degree is None else args.sh_degree
        )
        if not 0 <= sh_degree <= available_sh_degree:
            raise ValueError(
                f"--sh-degree must lie in [0,{available_sh_degree}]."
            )
        intrinsic = camera.intrinsic_matrix(width=args.width, height=height)
        world_to_camera = image.world_to_camera
        background_value = 1.0 if args.background == "white" else 0.0
        LOGGER.info(
            "Rendering camera[%d]=%s at %dx%d with %d Gaussians",
            camera_index,
            image.name,
            args.width,
            height,
            len(splats["means"]),
        )
        with torch.inference_mode():
            rendered, alpha, _ = rasterization(
                means=splats["means"],
                quats=functional.normalize(splats["quats"], dim=-1),
                scales=torch.exp(splats["scales"]),
                opacities=torch.sigmoid(splats["opacities"]),
                colors=splats["colors"],
                viewmats=torch.as_tensor(
                    world_to_camera,
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0),
                Ks=torch.as_tensor(
                    intrinsic,
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0),
                width=args.width,
                height=height,
                sh_degree=sh_degree,
                packed=True,
                render_mode="RGB+ED",
            )
        rendered_rgb = rendered[0, ..., :3]
        rendered_alpha = alpha[0, ..., :1]
        if background_value != 0.0:
            rendered_rgb = rendered_rgb + (1.0 - rendered_alpha) * background_value
        rgb = rendered_rgb.clamp(0.0, 1.0).cpu().numpy()
        depth = rendered[0, ..., 3].clamp_min(0.0).cpu().numpy()
        alpha_np = alpha[0, ..., 0].clamp(0.0, 1.0).cpu().numpy()
        rgb_uint8 = np.rint(rgb * 255.0).astype(np.uint8)
        view = GaussianAlignmentView(
            rgb_uint8=rgb_uint8,
            expected_depth_g=depth,
            alpha=alpha_np,
            intrinsic_matrix=intrinsic,
            camera_to_world_g=image.camera_to_world,
            camera_name=image.name,
            metadata={
                "scene_root": str(scene_root),
                "scene_ply": str(scene_ply),
                "sparse_dir": str(sparse_dir),
                "camera_index": camera_index,
                "camera_image_id": image.image_id,
                "camera_model": camera.model,
                "source_width": camera.width,
                "source_height": camera.height,
                "render_mode": "RGB+ED",
                "sh_degree": sh_degree,
                "gaussian_count": len(splats["means"]),
            },
        )
        view.save(output)
        from PIL import Image

        png_output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb_uint8, mode="RGB").save(png_output)
        LOGGER.info("Saved %s and %s", output, png_output)
        return 0
    except Exception as error:
        LOGGER.error("%s", error)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
