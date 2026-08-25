# RTMDet + RTMPose TensorRT FP16 独立实验

这个目录是与现有 PyTorch 实时链路隔离的部署实验。它不会覆盖
`view_live_multi_person.py`，也不会改写现有 `.pth` 权重。

## 边界

- x86 RTX 主机：加载原始 OpenMMLab 模型，导出并检查 ONNX。
- Jetson Orin Nano：使用 Nano 自带的 TensorRT 从 ONNX 构建 FP16
  Engine，并运行基准测试。
- `.engine` 与 GPU 架构、TensorRT/CUDA 版本和构建配置绑定，不能在
  RTX 主机生成后直接复制到 Nano。

当前实验锁定的模型为：

- detector：RTMDet-M person-only，输入 `1x3x640x640`；
- pose：RTMPose-M Halpe26 SimCC，输入 `Bx3x256x192`，动态 batch
  `1..4`；
- pose 输出：`simcc_x = Bx26x384`，`simcc_y = Bx26x512`；
- 精度：FP16；
- 快速路径：单次 pose forward，暂不执行原配置中的 flip test。

已确认的目标 Nano 环境（2026-08-14）：

```text
Jetson Orin Nano Engineering Reference Developer Kit Super
aarch64 / JetPack 6.2.2 / L4T R36.5
CUDA Toolkit 12.6
TensorRT 10.3.0.30（JetPack Debian 包基于 CUDA 12.5）
trtexec=/usr/src/tensorrt/bin/trtexec
```

不要因为 TensorRT Debian 包名带 `cuda12.5` 而单独降级 CUDA Toolkit；
先以 JetPack 已安装组合运行 Parser/Engine 验收。

## 为什么不直接使用 MMDeploy TensorRT NMS

MMDeploy v1.3.1 的 TensorRT detector 配置会产生
`mmdeploy::TRTBatchedNMS` 自定义算子，需要编译旧式 TensorRT 插件。
JetPack 6 使用 TensorRT 10 系列，插件 ABI 是额外风险。因此 detector
先通过 ONNX Runtime 部署配置导出标准 `NonMaxSuppression`，再由 Nano
上的 TensorRT ONNX Parser 实际验收。如果 Nano 的具体 TensorRT 版本
不能解析该图，构建日志会明确失败，而不会生成一个不可验证的 Engine。

## 1. 查询 Nano 信息

在 Nano 上运行：

```bash
cd ~/program/humanpose
bash scripts/tensorrt_fp16/probe_nano.sh \
  | tee diagnostics/nano-tensorrt-probe.txt
```

必须先确认 TensorRT Python 版本和 `trtexec` 的真实路径。

## 2. 主机导出 ONNX

推荐给现有 `rgbd-avatar` Conda 环境建立一个可丢弃的、继承其包的
venv，只在其中增加 MMDeploy/ONNX 依赖：

```bash
cd /home/fr1511b/program/workspace/humanpose
conda run -n rgbd-avatar python -m venv \
  --system-site-packages .venv-trt-export
source .venv-trt-export/bin/activate

python -m pip install --no-deps \
  protobuf==3.20.2 \
  onnx==1.13.1 \
  onnxruntime==1.16.3 \
  aenum==3.1.15 \
  grpcio==1.74.0 \
  multiprocess==0.70.18 \
  prettytable==3.16.0 \
  dill==0.4.0 \
  coloredlogs==15.0.1 \
  flatbuffers==25.2.10 \
  humanfriendly==10.0
```

设置官方 MMDeploy v1.3.1 源码目录后执行：

```bash
export MMDEPLOY_DIR=/home/fr1511b/program/vendor-packages/mmdeploy-1.3.1
export PYTHON_BIN="$PWD/.venv-trt-export/bin/python"
bash scripts/tensorrt_fp16/export_onnx_host.sh
```

默认输出在：

```text
outputs/tensorrt_fp16/
├── SHA256SUMS
└── onnx/
    ├── detector/
    │   ├── rtmdet_m_person_640.onnx
    │   └── model-info.json
    └── pose/
        ├── rtmpose_m_halpe26_256x192.onnx
        └── model-info.json
```

只有 ONNX checker、ONNX Runtime 实际执行、形状检查、“无自定义算子”
检查，以及同输入下的 PyTorch/ONNX 数值一致性检查全部通过，才能复制
到 Nano。

## 3. 复制 ONNX 到 Nano

从主机执行，目标目录必须是当前 Nano 的 humanpose 工程：

```bash
rsync -av --progress \
  outputs/tensorrt_fp16/ \
  nvidia@192.168.8.119:/home/nvidia/program/humanpose/outputs/tensorrt_fp16/
```

在 Nano 上校验：

```bash
cd ~/program/humanpose/outputs/tensorrt_fp16
sha256sum -c SHA256SUMS
```

## 4. 只在 Nano 构建 Engine

```bash
cd ~/program/humanpose
bash scripts/tensorrt_fp16/build_engines_nano.sh
```

脚本会：

1. 构建静态 batch-1 RTMDet FP16 Engine；
2. 构建 batch 1/2/4 RTMPose FP16 Engine；
3. 分别运行 `trtexec` 基准；
4. 保存完整 parser/build/benchmark 日志和 Engine SHA256。

## 5. 接入条件

在新增实时 TensorRT backend 前必须全部满足：

1. detector ONNX 与 PyTorch 的人物数量、框坐标和分数通过对照；
2. pose ONNX/TRT 与 PyTorch 单次 forward 的26关节误差在约定阈值内；
3. batch 1、2、4 都能执行；
4. 空画面、单人、两人和遮挡画面均通过；
5. TensorRT 耗时相对当前约 `117.5 ms` 有实际收益；
6. 不改变现有 `depth_connected`、ID 跟踪和 WebSocket schema。

## 6. 独立实时入口

原 PyTorch 入口仍为 `scripts/view_live_multi_person.py`。TensorRT 实验
使用不同脚本，不加载 MMPose 模型：

```bash
PYTHONPATH=src python scripts/view_live_multi_person_tensorrt.py \
  --source sdk \
  --device cuda:0 \
  --detector auto \
  --identity-tracker geometry \
  --recovery-method depth_connected \
  --max-persons 4 \
  --publish-stickmen \
  --headless
```

占用相机前，先用单张图验证 TensorRT Python 运行时、动态 NMS 输出
分配、预处理和 SimCC 解码：

```bash
PYTHONPATH=src python scripts/tensorrt_fp16/test_single_image_nano.py \
  --image diagnostics/person-test.png
```

两个命令默认从 `outputs/tensorrt_fp16/engines` 加载 Engine。需要覆盖
路径时设置 `HUMANPOSE_TRT_DETECTOR_ENGINE` 和
`HUMANPOSE_TRT_POSE_ENGINE`。原来的 `rtmpose` 入口保持不变，可随时
回退。当前 RTMPose Engine 的动态 batch profile 为 `1/2/4`，因此
`--max-persons 4` 不需要重新构建 Engine；四人场景仍需在 Nano 上实测
整链路帧率和温度。
