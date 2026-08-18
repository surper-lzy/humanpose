#!/usr/bin/env python3
"""Compare MMDeploy-rewritten PyTorch outputs with exported ONNX outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("detector", "pose"), required=True)
    parser.add_argument("--deploy-config", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=2e-4)
    parser.add_argument("--rtol", type=float, default=2e-4)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def main() -> int:
    args = parse_args()

    from mmdeploy.apis.utils import build_task_processor
    from mmdeploy.core import RewriterContext, patch_model
    from mmdeploy.utils import Backend, IR, get_input_shape, load_config

    deploy_cfg, model_cfg = load_config(
        str(args.deploy_config.resolve()), str(args.model_config.resolve())
    )
    task_processor = build_task_processor(model_cfg, deploy_cfg, "cpu")
    pytorch_model = task_processor.build_pytorch_model(
        str(args.checkpoint.resolve())
    ).eval()

    data, model_inputs = task_processor.create_input(
        str(args.image.resolve()),
        get_input_shape(deploy_cfg),
        data_preprocessor=getattr(pytorch_model, "data_preprocessor", None),
    )
    if isinstance(model_inputs, list) and len(model_inputs) == 1:
        model_inputs = model_inputs[0]
    if not isinstance(model_inputs, torch.Tensor):
        raise SystemExit(
            f"Expected tensor model input, got {type(model_inputs).__name__}"
        )

    patched_model = patch_model(
        pytorch_model,
        cfg=deploy_cfg,
        backend=Backend.ONNXRUNTIME.value,
        ir=IR.ONNX,
    )
    with RewriterContext(
        cfg=deploy_cfg,
        backend=Backend.ONNXRUNTIME.value,
        ir=IR.ONNX,
        opset=11,
    ), torch.no_grad():
        pytorch_outputs = patched_model(
            model_inputs,
            data_samples=data["data_samples"],
            mode="predict",
        )
    if not isinstance(pytorch_outputs, (tuple, list)):
        pytorch_outputs = (pytorch_outputs,)
    pytorch_arrays = [_as_numpy(value) for value in pytorch_outputs]

    session = ort.InferenceSession(
        str(args.onnx.resolve()), providers=["CPUExecutionProvider"]
    )
    onnx_arrays = session.run(
        None, {"input": model_inputs.detach().cpu().numpy()}
    )
    if len(pytorch_arrays) != len(onnx_arrays):
        raise SystemExit(
            "Output count differs: "
            f"PyTorch={len(pytorch_arrays)}, ONNX={len(onnx_arrays)}"
        )

    comparisons: list[dict[str, Any]] = []
    for index, (expected, actual) in enumerate(
        zip(pytorch_arrays, onnx_arrays)
    ):
        if expected.shape != actual.shape:
            raise SystemExit(
                f"Output {index} shape differs: "
                f"PyTorch={expected.shape}, ONNX={actual.shape}"
            )
        difference = np.abs(expected.astype(np.float64) - actual)
        close = bool(
            np.allclose(expected, actual, atol=args.atol, rtol=args.rtol)
        )
        comparisons.append(
            {
                "index": index,
                "shape": list(expected.shape),
                "pytorch_dtype": str(expected.dtype),
                "onnx_dtype": str(actual.dtype),
                "max_abs_error": float(difference.max(initial=0.0)),
                "mean_abs_error": float(difference.mean()),
                "allclose": close,
            }
        )
        if not close:
            raise SystemExit(
                f"Output {index} exceeds parity tolerance: "
                f"max_abs_error={comparisons[-1]['max_abs_error']}"
            )

    report = {
        "kind": args.kind,
        "image": str(args.image.resolve()),
        "onnx": str(args.onnx.resolve()),
        "input_shape": list(model_inputs.shape),
        "atol": args.atol,
        "rtol": args.rtol,
        "outputs": comparisons,
        "status": "PASS",
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.json_output is not None:
        output = args.json_output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
