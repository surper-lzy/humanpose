#!/usr/bin/env python3
"""Create a case-one 3DGS placement from measured scene points."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from rgbd_avatar.avatar import SMPLSequenceCache
from rgbd_avatar.io import load_json_mapping
from rgbd_avatar.scene import (
    ManualScenePlacement,
    build_manual_scene_alignment,
    first_avatar_ground_anchor,
)


LOGGER = logging.getLogger("create_scene_alignment")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SCENE_ROOT = PROJECT_ROOT.parent / "data/3DGS"
DEFAULT_SMPL_CACHE = (
    PROJECT_ROOT
    / "outputs/sequences/4_pointcloud_exit_gate/smpl_sequence.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        type=Path,
        required=True,
        help="JSON containing known length, floor, spawn, and forward picks.",
    )
    parser.add_argument(
        "--scene-root",
        type=Path,
        default=DEFAULT_SCENE_ROOT,
    )
    parser.add_argument(
        "--scene-ply",
        type=Path,
        default=None,
        help="Default: point_cloud.ply under --scene-root.",
    )
    parser.add_argument(
        "--smpl-cache",
        type=Path,
        default=DEFAULT_SMPL_CACHE,
    )
    parser.add_argument(
        "--anchor-mode",
        choices=("feet", "pelvis", "origin"),
        default="feet",
        help=(
            "Used only when the spec omits avatar_anchor_w_m. The selected "
            "anchor is projected to W ground Z=0."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: scene_alignment.json under --scene-root.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def placement_from_spec(
    payload: dict[str, Any],
    *,
    smpl_cache_path: Path,
    anchor_mode: str,
) -> ManualScenePlacement:
    spec = dict(payload)
    if spec.get("avatar_anchor_w_m") is None:
        cache = SMPLSequenceCache.load(smpl_cache_path)
        spec["avatar_anchor_w_m"] = first_avatar_ground_anchor(
            cache,
            mode=anchor_mode,
        ).tolist()
    return ManualScenePlacement.from_mapping(spec)


def write_alignment(
    placement: ManualScenePlacement,
    *,
    output_path: Path,
    scene_root: Path,
    scene_ply: Path,
    smpl_cache_path: Path,
    overwrite: bool,
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {output_path}; pass --overwrite to replace it."
        )
    if not scene_ply.is_file():
        raise FileNotFoundError(f"3DGS PLY not found: {scene_ply}")
    metadata: dict[str, Any] = {
        "scene_root": str(scene_root),
        "scene_ply": str(scene_ply),
        "scene_ply_size_bytes": scene_ply.stat().st_size,
        "smpl_cache": str(smpl_cache_path),
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    alignment = build_manual_scene_alignment(
        placement,
        metadata=metadata,
    )
    alignment.save(output_path)
    LOGGER.info(
        "Saved %s: scale=%.8f G/m spawn=%s forward=%s up=%s",
        output_path,
        alignment.scale_g_per_m,
        alignment.spawn_point_g.tolist(),
        alignment.forward_g.tolist(),
        alignment.ground_normal_g.tolist(),
    )


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
        smpl_cache_path = args.smpl_cache.expanduser().resolve()
        output_path = (
            args.output.expanduser().resolve()
            if args.output is not None
            else scene_root / "scene_alignment.json"
        )
        spec_path = args.spec.expanduser().resolve()
        placement = placement_from_spec(
            load_json_mapping(spec_path),
            smpl_cache_path=smpl_cache_path,
            anchor_mode=args.anchor_mode,
        )
        write_alignment(
            placement,
            output_path=output_path,
            scene_root=scene_root,
            scene_ply=scene_ply,
            smpl_cache_path=smpl_cache_path,
            overwrite=args.overwrite,
            extra_metadata={
                "placement_spec": str(spec_path),
                "avatar_anchor_mode": args.anchor_mode,
            },
        )
        return 0
    except Exception as error:
        LOGGER.error("%s", error)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
