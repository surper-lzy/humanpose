#!/usr/bin/env python3
"""Validate and summarize ONNX artifacts before copying them to Jetson."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx


def _dimension_value(dimension: Any) -> int | str | None:
    if dimension.HasField("dim_value"):
        return int(dimension.dim_value)
    if dimension.HasField("dim_param"):
        return str(dimension.dim_param)
    return None


def _tensor_description(value_info: Any) -> dict[str, Any]:
    tensor_type = value_info.type.tensor_type
    return {
        "name": value_info.name,
        "element_type": int(tensor_type.elem_type),
        "shape": [
            _dimension_value(dimension)
            for dimension in tensor_type.shape.dim
        ],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "--kind", choices=("detector", "pose"), required=True
    )
    parser.add_argument(
        "--allow-custom-domain",
        action="store_true",
        help="Allow non-standard ONNX domains. Disabled for plugin-free TRT.",
    )
    parser.add_argument("--json-output", type=Path)
    parser.add_argument(
        "--runtime-check",
        action="store_true",
        help="Execute a synthetic input with ONNX Runtime and check outputs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = args.model.expanduser().resolve()
    model = onnx.load(str(path), load_external_data=True)
    onnx.checker.check_model(model, full_check=True)

    inputs = [_tensor_description(item) for item in model.graph.input]
    outputs = [_tensor_description(item) for item in model.graph.output]
    operator_domains = sorted(
        {
            (node.domain or "ai.onnx")
            for node in model.graph.node
        }
    )
    custom_domains = [
        domain for domain in operator_domains if domain != "ai.onnx"
    ]
    if custom_domains and not args.allow_custom_domain:
        raise SystemExit(
            "Custom ONNX operators are not allowed for the plugin-free "
            f"TensorRT path: {custom_domains}"
        )

    input_names = [item["name"] for item in inputs]
    output_names = [item["name"] for item in outputs]
    if input_names != ["input"]:
        raise SystemExit(f"Unexpected model inputs: {input_names}")

    if args.kind == "detector":
        if output_names != ["dets", "labels"]:
            raise SystemExit(f"Unexpected detector outputs: {output_names}")
        expected_input_tail = [3, 640, 640]
        if inputs[0]["shape"][-3:] != expected_input_tail:
            raise SystemExit(
                f"Unexpected detector input shape: {inputs[0]['shape']}"
            )
    else:
        if output_names != ["simcc_x", "simcc_y"]:
            raise SystemExit(f"Unexpected pose outputs: {output_names}")
        expected_input_tail = [3, 256, 192]
        if inputs[0]["shape"][-3:] != expected_input_tail:
            raise SystemExit(
                f"Unexpected pose input shape: {inputs[0]['shape']}"
            )
        simcc_x_shape = outputs[0]["shape"]
        simcc_y_shape = outputs[1]["shape"]
        if simcc_x_shape[-1] != 384 or (
            isinstance(simcc_x_shape[-2], int) and simcc_x_shape[-2] != 26
        ):
            raise SystemExit(
                f"Unexpected simcc_x shape: {simcc_x_shape}"
            )
        if simcc_y_shape[-1] != 512 or (
            isinstance(simcc_y_shape[-2], int) and simcc_y_shape[-2] != 26
        ):
            raise SystemExit(
                f"Unexpected simcc_y shape: {simcc_y_shape}"
            )

    summary = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "ir_version": int(model.ir_version),
        "opsets": {
            (entry.domain or "ai.onnx"): int(entry.version)
            for entry in model.opset_import
        },
        "operator_domains": operator_domains,
        "inputs": inputs,
        "outputs": outputs,
        "node_count": len(model.graph.node),
    }

    if args.runtime_check:
        import onnxruntime as ort

        session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        if args.kind == "detector":
            runtime_input = np.zeros((1, 3, 640, 640), dtype=np.float32)
        else:
            runtime_input = np.zeros((2, 3, 256, 192), dtype=np.float32)
        runtime_outputs = session.run(None, {"input": runtime_input})
        runtime_shapes = [list(value.shape) for value in runtime_outputs]
        if args.kind == "detector":
            if len(runtime_shapes) != 2:
                raise SystemExit(
                    f"Detector runtime output count is invalid: {runtime_shapes}"
                )
            if runtime_shapes[0][0] != 1 or runtime_shapes[0][-1] != 5:
                raise SystemExit(
                    f"Detector runtime dets shape is invalid: {runtime_shapes[0]}"
                )
            if runtime_shapes[1] != runtime_shapes[0][:-1]:
                raise SystemExit(
                    "Detector labels do not match dets: "
                    f"{runtime_shapes}"
                )
        elif runtime_shapes != [[2, 26, 384], [2, 26, 512]]:
            raise SystemExit(
                f"Pose runtime output shapes are invalid: {runtime_shapes}"
            )
        summary["onnxruntime_check"] = {
            "version": ort.__version__,
            "input_shape": list(runtime_input.shape),
            "output_shapes": runtime_shapes,
            "finite_outputs": [
                bool(np.isfinite(value).all()) for value in runtime_outputs
            ],
        }
        if not all(summary["onnxruntime_check"]["finite_outputs"]):
            raise SystemExit("ONNX Runtime produced non-finite output values.")

    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    print(rendered)
    if args.json_output is not None:
        output = args.json_output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
