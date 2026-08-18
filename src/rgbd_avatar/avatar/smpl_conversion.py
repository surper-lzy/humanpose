#!/usr/bin/env python3
"""Create a Chumpy-free SMPL pickle while preserving the original file."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import pickle
import tempfile
from typing import Any

import numpy as np


LOGGER = logging.getLogger("convert_smpl_model")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = PROJECT_ROOT / "assets/models/smpl/SMPL_NEUTRAL.pkl"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "assets/models/smpl/SMPL_NEUTRAL_CLEAN.pkl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-7,
        help="Maximum allowed zero-pose vertex/joint difference in meters.",
    )
    return parser.parse_args()


def _install_legacy_numpy_aliases() -> None:
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


def _clean_value(value: Any, chumpy_type: type) -> tuple[Any, int]:
    if isinstance(value, chumpy_type):
        return np.asarray(value.r).copy(), 1
    if isinstance(value, dict):
        cleaned: dict[Any, Any] = {}
        converted = 0
        for key, item in value.items():
            cleaned_item, item_count = _clean_value(item, chumpy_type)
            cleaned[key] = cleaned_item
            converted += item_count
        return cleaned, converted
    if isinstance(value, list):
        cleaned_items = []
        converted = 0
        for item in value:
            cleaned_item, item_count = _clean_value(item, chumpy_type)
            cleaned_items.append(cleaned_item)
            converted += item_count
        return cleaned_items, converted
    if isinstance(value, tuple):
        cleaned_items = []
        converted = 0
        for item in value:
            cleaned_item, item_count = _clean_value(item, chumpy_type)
            cleaned_items.append(cleaned_item)
            converted += item_count
        return tuple(cleaned_items), converted
    return value, 0


def _count_chumpy(value: Any, chumpy_type: type) -> int:
    if isinstance(value, chumpy_type):
        return 1
    if isinstance(value, dict):
        return sum(
            _count_chumpy(item, chumpy_type) for item in value.values()
        )
    if isinstance(value, (list, tuple)):
        return sum(_count_chumpy(item, chumpy_type) for item in value)
    return 0


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as file:
        return pickle.load(file, encoding="latin1")


def _verify_forward_equivalence(
    original_path: Path,
    clean_path: Path,
    *,
    tolerance: float,
) -> tuple[float, float]:
    import smplx
    import torch

    original = smplx.SMPL(str(original_path), batch_size=1).eval()
    cleaned = smplx.SMPL(str(clean_path), batch_size=1).eval()
    with torch.no_grad():
        original_output = original(return_verts=True)
        clean_output = cleaned(return_verts=True)
    vertex_difference = float(
        torch.max(
            torch.abs(
                original_output.vertices - clean_output.vertices
            )
        ).item()
    )
    joint_difference = float(
        torch.max(
            torch.abs(original_output.joints - clean_output.joints)
        ).item()
    )
    if vertex_difference > tolerance or joint_difference > tolerance:
        raise ValueError(
            "Converted model changed the zero-pose output: "
            f"vertex_max={vertex_difference:.3e}, "
            f"joint_max={joint_difference:.3e}, "
            f"tolerance={tolerance:.3e}."
        )
    if not np.array_equal(original.faces, cleaned.faces):
        raise ValueError("Converted model changed the face topology.")
    return vertex_difference, joint_difference


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    temporary_path: Path | None = None
    try:
        input_path = args.input.expanduser().resolve()
        output_path = args.output.expanduser().resolve()
        if not input_path.is_file():
            raise FileNotFoundError(f"Input SMPL model not found: {input_path}")
        if input_path == output_path:
            raise ValueError("Input and output paths must be different.")
        if output_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing output: {output_path}"
            )
        if not np.isfinite(args.tolerance) or args.tolerance < 0:
            raise ValueError("--tolerance must be finite and non-negative.")

        _install_legacy_numpy_aliases()
        import chumpy

        original_data = _load_pickle(input_path)
        before_count = _count_chumpy(original_data, chumpy.Ch)
        if before_count == 0:
            raise ValueError("Input model contains no Chumpy objects.")
        cleaned_data, converted_count = _clean_value(
            original_data,
            chumpy.Ch,
        )
        if converted_count != before_count:
            raise RuntimeError(
                "Internal conversion count mismatch: "
                f"found={before_count}, converted={converted_count}."
            )
        if _count_chumpy(cleaned_data, chumpy.Ch):
            raise RuntimeError("Chumpy objects remain after conversion.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{output_path.stem}.",
            suffix=".pkl",
            dir=output_path.parent,
            delete=False,
        ) as temporary_file:
            pickle.dump(cleaned_data, temporary_file, protocol=4)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
            temporary_path = Path(temporary_file.name)

        reloaded = _load_pickle(temporary_path)
        if _count_chumpy(reloaded, chumpy.Ch):
            raise RuntimeError("Serialized clean model still contains Chumpy.")
        vertex_difference, joint_difference = _verify_forward_equivalence(
            input_path,
            temporary_path,
            tolerance=args.tolerance,
        )
        os.replace(temporary_path, output_path)
        output_path.chmod(input_path.stat().st_mode & 0o777)
        temporary_path = None

        LOGGER.info("Converted %d Chumpy object(s).", converted_count)
        LOGGER.info("Original preserved: %s", input_path)
        LOGGER.info("Clean model written: %s", output_path)
        LOGGER.info(
            "Zero-pose equivalence: vertex_max=%.3e m joint_max=%.3e m",
            vertex_difference,
            joint_difference,
        )
        return 0
    except Exception as error:
        LOGGER.error("%s", error)
        return 2
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
