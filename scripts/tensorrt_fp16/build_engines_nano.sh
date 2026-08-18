#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/../.." && pwd)"
artifact_root="${ARTIFACT_ROOT:-${1:-$project_root/outputs/tensorrt_fp16}}"

detector_onnx="$artifact_root/onnx/detector/rtmdet_m_person_640.onnx"
pose_onnx="$artifact_root/onnx/pose/rtmpose_m_halpe26_256x192.onnx"
engine_dir="$artifact_root/engines"
log_dir="$artifact_root/logs"

trtexec_bin="${TRTEXEC:-}"
if test -z "$trtexec_bin"; then
  trtexec_bin="$(command -v trtexec || true)"
fi
if test -z "$trtexec_bin" && test -x /usr/src/tensorrt/bin/trtexec; then
  trtexec_bin=/usr/src/tensorrt/bin/trtexec
fi
if test -z "$trtexec_bin" || ! test -x "$trtexec_bin"; then
  echo "trtexec was not found. Install the JetPack TensorRT CLI package first." >&2
  exit 1
fi

for model in "$detector_onnx" "$pose_onnx"; do
  if ! test -f "$model"; then
    echo "ONNX model not found: $model" >&2
    exit 1
  fi
done

mkdir -p "$engine_dir" "$log_dir"

echo "TensorRT CLI: $trtexec_bin"
"$trtexec_bin" --help 2>&1 \
  | sed -n '1,12p' \
  | tee "$log_dir/trtexec-version.log"

echo "Building RTMDet-M static batch-1 FP16 engine..."
"$trtexec_bin" \
  --onnx="$detector_onnx" \
  --saveEngine="$engine_dir/rtmdet_m_person_640_fp16.engine" \
  --fp16 \
  --memPoolSize=workspace:1024 \
  --skipInference \
  2>&1 | tee "$log_dir/build-rtmdet.log"

echo "Building RTMPose-M dynamic batch 1/2/4 FP16 engine..."
"$trtexec_bin" \
  --onnx="$pose_onnx" \
  --saveEngine="$engine_dir/rtmpose_m_halpe26_256x192_fp16.engine" \
  --fp16 \
  --minShapes=input:1x3x256x192 \
  --optShapes=input:2x3x256x192 \
  --maxShapes=input:4x3x256x192 \
  --memPoolSize=workspace:1024 \
  --skipInference \
  2>&1 | tee "$log_dir/build-rtmpose.log"

echo "Benchmarking detector engine..."
"$trtexec_bin" \
  --loadEngine="$engine_dir/rtmdet_m_person_640_fp16.engine" \
  --warmUp=2000 \
  --duration=10 \
  --noDataTransfers \
  --useCudaGraph \
  2>&1 | tee "$log_dir/benchmark-rtmdet.log"

echo "Benchmarking pose engine at the two-person batch size..."
"$trtexec_bin" \
  --loadEngine="$engine_dir/rtmpose_m_halpe26_256x192_fp16.engine" \
  --shapes=input:2x3x256x192 \
  --warmUp=2000 \
  --duration=10 \
  --noDataTransfers \
  --useCudaGraph \
  2>&1 | tee "$log_dir/benchmark-rtmpose-batch2.log"

(
  cd "$artifact_root"
  sha256sum engines/*.engine > ENGINE_SHA256SUMS
)

echo "TensorRT FP16 engines built under: $engine_dir"
echo "Do not copy these .engine files to another GPU/TensorRT version."
