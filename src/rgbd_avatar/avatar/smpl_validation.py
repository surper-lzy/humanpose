#!/usr/bin/env python3
"""Validate and optionally display a standalone SMPL neutral model."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np


LOGGER = logging.getLogger("test_smpl_model")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ORIGINAL_MODEL = PROJECT_ROOT / "assets/models/smpl/SMPL_NEUTRAL.pkl"
CLEAN_MODEL = PROJECT_ROOT / "assets/models/smpl/SMPL_NEUTRAL_CLEAN.pkl"


def _default_model_path() -> Path:
    return CLEAN_MODEL if CLEAN_MODEL.is_file() else ORIGINAL_MODEL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=_default_model_path(),
    )
    parser.add_argument(
        "--legacy-chumpy-compat",
        action="store_true",
        help=(
            "Install process-local NumPy aliases for an unconverted legacy "
            "Chumpy pickle. Clean models do not need this."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run all checks without opening an Open3D window.",
    )
    parser.add_argument(
        "--benchmark-iterations",
        type=int,
        default=20,
    )
    parser.add_argument("--window-width", type=int, default=900)
    parser.add_argument("--window-height", type=int, default=900)
    return parser.parse_args()


def _install_legacy_numpy_aliases() -> None:
    """Allow legacy Chumpy-based SMPL pickles to load on NumPy >= 1.24."""

    aliases: dict[str, Any] = {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "unicode": str,
        "str": str,
    }
    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)


def _resolve_device(requested: str, torch: Any) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable.")
    return requested


def _edge_statistics(faces: np.ndarray) -> tuple[int, int, int]:
    edges = np.sort(
        np.concatenate(
            (
                faces[:, [0, 1]],
                faces[:, [1, 2]],
                faces[:, [2, 0]],
            ),
            axis=0,
        ),
        axis=1,
    )
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_count = int(np.count_nonzero(counts == 1))
    nonmanifold_count = int(np.count_nonzero(counts > 2))
    return len(unique_edges), boundary_count, nonmanifold_count


def _validate_mesh(vertices: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Expected Vx3 vertices, got {vertices.shape}.")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Expected Fx3 faces, got {faces.shape}.")
    if not np.isfinite(vertices).all():
        raise ValueError("SMPL produced non-finite vertices.")
    if faces.size == 0 or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("SMPL face indices are outside the vertex array.")

    repeated_indices = (
        (faces[:, 0] == faces[:, 1])
        | (faces[:, 1] == faces[:, 2])
        | (faces[:, 2] == faces[:, 0])
    )
    triangle_area = 0.5 * np.linalg.norm(
        np.cross(
            vertices[faces[:, 1]] - vertices[faces[:, 0]],
            vertices[faces[:, 2]] - vertices[faces[:, 0]],
        ),
        axis=1,
    )
    unique_edges, boundary_edges, nonmanifold_edges = _edge_statistics(faces)
    if np.any(repeated_indices) or np.any(triangle_area < 1e-12):
        raise ValueError("SMPL contains degenerate triangles.")
    if boundary_edges or nonmanifold_edges:
        raise ValueError(
            "SMPL topology is not a closed two-manifold: "
            f"boundary={boundary_edges}, nonmanifold={nonmanifold_edges}."
        )

    extents = np.ptp(vertices, axis=0)
    return {
        "vertices": len(vertices),
        "triangles": len(faces),
        "unique_edges": unique_edges,
        "euler_characteristic": (
            len(vertices) - unique_edges + len(faces)
        ),
        "axis_extents_m": extents,
        "largest_extent_m": float(np.max(extents)),
    }


def _to_open3d_mesh(vertices: np.ndarray, faces: np.ndarray) -> Any:
    try:
        import open3d as o3d
    except ImportError as error:
        raise RuntimeError("Open3D is required to display SMPL.") from error

    # SMPL uses Y-up. Rotate +90 degrees about X into the project viewer's
    # proper right-handed display convention: X right, Y forward, Z up.
    display_from_smpl = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    display_vertices = vertices @ display_from_smpl.T
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(display_vertices),
        o3d.utility.Vector3iVector(faces.astype(np.int32)),
    )
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color((0.72, 0.74, 0.76))
    return o3d, mesh


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    try:
        if args.benchmark_iterations <= 0:
            raise ValueError("--benchmark-iterations must be positive.")
        model_path = args.model.expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"SMPL model not found: {model_path}")

        if args.legacy_chumpy_compat:
            _install_legacy_numpy_aliases()
        import smplx
        import torch

        device = _resolve_device(args.device, torch)
        model = smplx.create(
            str(model_path),
            model_type="smpl",
            gender="neutral",
            ext=model_path.suffix.lstrip("."),
            batch_size=1,
        ).to(device).eval()

        started = time.perf_counter()
        with torch.no_grad():
            output = None
            for _ in range(args.benchmark_iterations):
                output = model(return_verts=True)
        elapsed_s = time.perf_counter() - started
        assert output is not None
        vertices = output.vertices[0].detach().cpu().numpy()
        joints = output.joints[0].detach().cpu().numpy()
        faces = np.asarray(model.faces, dtype=np.int64)
        if not np.isfinite(joints).all():
            raise ValueError("SMPL produced non-finite joints.")
        metrics = _validate_mesh(vertices, faces)

        LOGGER.info("SMPL model is usable: %s", model_path)
        LOGGER.info(
            "device=%s vertices=%d triangles=%d joints=%d betas=%d",
            device,
            metrics["vertices"],
            metrics["triangles"],
            len(joints),
            model.num_betas,
        )
        LOGGER.info(
            "topology: edges=%d Euler=%d closed_manifold=yes",
            metrics["unique_edges"],
            metrics["euler_characteristic"],
        )
        LOGGER.info(
            "zero-pose extents XYZ=%s m; largest=%.4f m",
            np.round(metrics["axis_extents_m"], 4).tolist(),
            metrics["largest_extent_m"],
        )
        LOGGER.info(
            "forward mean=%.3f ms over %d iterations",
            1000.0 * elapsed_s / args.benchmark_iterations,
            args.benchmark_iterations,
        )

        if args.validate_only:
            return 0

        o3d, mesh = _to_open3d_mesh(vertices, faces)
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)
        o3d.visualization.draw_geometries(
            [mesh, axis],
            window_name=(
                "SMPL Neutral zero pose | X right, Y forward, Z up"
            ),
            width=args.window_width,
            height=args.window_height,
            mesh_show_back_face=True,
        )
        return 0
    except Exception as error:
        LOGGER.error("%s", error)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
