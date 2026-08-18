#!/usr/bin/env bash
set -eu

echo "=== PLATFORM ==="
date --iso-8601=seconds
hostname
uname -m
head -n 1 /etc/nv_tegra_release
python3 --version
nvcc --version 2>/dev/null | tail -n 1 || true

echo "=== JETPACK ==="
dpkg-query -W nvidia-jetpack nvidia-l4t-core 2>&1 || true

echo "=== TENSORRT PACKAGES ==="
dpkg-query -W \
  'tensorrt*' 'libnvinfer*' 'python3-libnvinfer*' \
  2>/dev/null | sort || true

echo "=== TRTEXEC ==="
command -v trtexec || true
find /usr/src/tensorrt /usr/local/bin /usr/bin \
  -maxdepth 3 -type f -name trtexec 2>/dev/null || true

echo "=== PYTHON TENSORRT ==="
python3 - <<'PY'
try:
    import tensorrt as trt
    print("TensorRT:", trt.__version__)
    print("TensorRT file:", trt.__file__)
except Exception as exc:
    print("TensorRT import failed:", repr(exc))
PY

echo "=== CUDA PYTORCH ENVIRONMENT ==="
humanpose_python="${HUMANPOSE_PYTHON:-$HOME/program/humanpose/.venv-nano/bin/python}"
if test -x "$humanpose_python"; then
  "$humanpose_python" - <<'PY'
import platform
import torch

print("machine:", platform.machine())
print("torch:", torch.__version__)
print("torch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PY
else
  echo "Humanpose Python not found: $humanpose_python"
fi

echo "=== RESOURCES ==="
free -h
df -h /
sudo -n nvpmodel -q 2>/dev/null || nvpmodel -q 2>/dev/null || true
