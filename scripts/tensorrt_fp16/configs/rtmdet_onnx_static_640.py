"""Plugin-free static ONNX export settings for RTMDet-M 640x640.

The ONNX Runtime backend is intentional even though the final backend is
TensorRT.  It exports the standard ONNX NonMaxSuppression operator rather
than MMDeploy's legacy ``mmdeploy::TRTBatchedNMS`` custom plugin.
"""

onnx_config = dict(
    type="onnx",
    export_params=True,
    keep_initializers_as_inputs=False,
    opset_version=11,
    save_file="rtmdet_m_person_640.onnx",
    input_names=["input"],
    output_names=["dets", "labels"],
    input_shape=(640, 640),
    optimize=True,
)

backend_config = dict(type="onnxruntime")

codebase_config = dict(
    type="mmdet",
    task="ObjectDetection",
    model_type="end2end",
    post_processing=dict(
        score_threshold=0.05,
        confidence_threshold=0.005,
        iou_threshold=0.6,
        max_output_boxes_per_class=200,
        pre_top_k=1000,
        keep_top_k=100,
        background_label_id=-1,
    ),
)
