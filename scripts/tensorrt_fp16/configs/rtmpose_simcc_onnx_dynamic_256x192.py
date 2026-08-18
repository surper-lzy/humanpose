"""Dynamic-batch ONNX export settings for RTMPose SimCC Halpe26."""

onnx_config = dict(
    type="onnx",
    export_params=True,
    keep_initializers_as_inputs=False,
    opset_version=11,
    save_file="rtmpose_m_halpe26_256x192.onnx",
    input_names=["input"],
    output_names=["simcc_x", "simcc_y"],
    input_shape=[192, 256],
    dynamic_axes={
        "input": {0: "batch"},
        "simcc_x": {0: "batch"},
        "simcc_y": {0: "batch"},
    },
    optimize=True,
)

backend_config = dict(type="onnxruntime")

codebase_config = dict(
    type="mmpose",
    task="PoseDetection",
    export_postprocess=False,
)
