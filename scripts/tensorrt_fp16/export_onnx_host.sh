#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/../.." && pwd)"

python_bin="${PYTHON_BIN:-python}"
mmdeploy_dir="${MMDEPLOY_DIR:-$project_root/../../vendor-packages/mmdeploy-1.3.1}"
test_image="${TEST_IMAGE:-${1:-$project_root/../data/1/20260730_145911656_r.png}}"
artifact_root="${ARTIFACT_ROOT:-${2:-$project_root/outputs/tensorrt_fp16}}"

detector_model_cfg="$script_dir/configs/rtmdet_m_person_640.py"
detector_deploy_cfg="$script_dir/configs/rtmdet_onnx_static_640.py"
detector_checkpoint="$project_root/assets/models/cache/hub/checkpoints/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"

pose_model_cfg="$script_dir/configs/rtmpose_m_halpe26_export.py"
pose_deploy_cfg="$script_dir/configs/rtmpose_simcc_onnx_dynamic_256x192.py"
pose_checkpoint="$project_root/assets/models/rtmpose/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.pth"

for required in \
  "$mmdeploy_dir/tools/torch2onnx.py" \
  "$test_image" \
  "$detector_checkpoint" \
  "$pose_checkpoint"; do
  if ! test -f "$required"; then
    echo "Required file not found: $required" >&2
    exit 1
  fi
done

"$python_bin" - <<'PY'
required = ("torch", "mmengine", "mmcv", "mmdet", "mmpose", "onnx")
for name in required:
    module = __import__(name)
    print(name, getattr(module, "__version__", "installed"), module.__file__)
PY

export PYTHONPATH="$mmdeploy_dir:$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/humanpose-trt-mpl}"
mkdir -p "$MPLCONFIGDIR" "$artifact_root/onnx/detector" "$artifact_root/onnx/pose"

echo "Exporting RTMDet-M person detector to standard ONNX..."
"$python_bin" "$mmdeploy_dir/tools/torch2onnx.py" \
  "$detector_deploy_cfg" \
  "$detector_model_cfg" \
  "$detector_checkpoint" \
  "$test_image" \
  --work-dir "$artifact_root/onnx/detector" \
  --device cpu \
  --log-level INFO \
  2>&1 | tee "$artifact_root/onnx/detector/export.log"

echo "Exporting RTMPose-M Halpe26 SimCC to dynamic-batch ONNX..."
"$python_bin" "$mmdeploy_dir/tools/torch2onnx.py" \
  "$pose_deploy_cfg" \
  "$pose_model_cfg" \
  "$pose_checkpoint" \
  "$test_image" \
  --work-dir "$artifact_root/onnx/pose" \
  --device cpu \
  --log-level INFO \
  2>&1 | tee "$artifact_root/onnx/pose/export.log"

detector_onnx="$artifact_root/onnx/detector/rtmdet_m_person_640.onnx"
pose_onnx="$artifact_root/onnx/pose/rtmpose_m_halpe26_256x192.onnx"

"$python_bin" "$script_dir/inspect_onnx.py" \
  "$detector_onnx" --kind detector \
  --runtime-check \
  --json-output "$artifact_root/onnx/detector/model-info.json"
"$python_bin" "$script_dir/inspect_onnx.py" \
  "$pose_onnx" --kind pose \
  --runtime-check \
  --json-output "$artifact_root/onnx/pose/model-info.json"

"$python_bin" "$script_dir/validate_onnx_parity.py" \
  --kind detector \
  --deploy-config "$detector_deploy_cfg" \
  --model-config "$detector_model_cfg" \
  --checkpoint "$detector_checkpoint" \
  --onnx "$detector_onnx" \
  --image "$test_image" \
  --json-output "$artifact_root/onnx/detector/parity.json"
"$python_bin" "$script_dir/validate_onnx_parity.py" \
  --kind pose \
  --deploy-config "$pose_deploy_cfg" \
  --model-config "$pose_model_cfg" \
  --checkpoint "$pose_checkpoint" \
  --onnx "$pose_onnx" \
  --image "$test_image" \
  --json-output "$artifact_root/onnx/pose/parity.json"

(
  cd "$artifact_root"
  sha256sum \
    onnx/detector/rtmdet_m_person_640.onnx \
    onnx/pose/rtmpose_m_halpe26_256x192.onnx \
    > SHA256SUMS
)

echo "ONNX export complete: $artifact_root"
echo "Next step: copy this directory to Nano, then run build_engines_nano.sh."
