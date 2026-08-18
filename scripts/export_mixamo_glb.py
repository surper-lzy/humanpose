"""Export a checked-in Mixamo FBX character as a self-contained GLB."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from rgbd_avatar.avatar.mixamo_gltf import export_mixamo_fbx_glb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("assets/models/mixamo/Ch09_nonPBR.fbx"),
        help="Source Mixamo binary FBX.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "frontend/mixamo-avatar-delivery/public/avatars/character-a.glb"
        ),
        help="Destination self-contained GLB.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file.",
    )
    parser.add_argument(
        "--keep-fbx-v",
        action="store_true",
        help="Do not flip FBX V texture coordinates for glTF convention.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    summary = export_mixamo_fbx_glb(
        args.input,
        args.output,
        overwrite=args.overwrite,
        flip_v=not args.keep_fbx_v,
    )
    logging.info(
        "Exported %s: source_vertices=%d exported_vertices=%d "
        "triangles=%d bones=%d size=%.2f MiB",
        summary.output_path,
        summary.source_vertex_count,
        summary.exported_vertex_count,
        summary.triangle_count,
        summary.bone_count,
        summary.byte_count / (1024 * 1024),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
