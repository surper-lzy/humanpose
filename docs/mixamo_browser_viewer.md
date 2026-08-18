# Mixamo 浏览器查看器

## 背景

Open3D 0.18 / 0.19 的环境（包含 `gsplat` 和 `rgbd-avatar` 两个 conda 环境）存在同一个 bug：**legacy Visualizer（`VisualizerWithKeyCallback` / `Visualizer`）无法渲染带纹理的 `TriangleMesh`**。只要窗口里存在 `mesh.textures = [...]` 的网格，第一次 `poll_events()` 即段错误（SIGSEGV）。

### 已排除的因素

| 变量 | 结论 |
|---|---|
| 纹理尺寸 | 4×4 / 64×64 / 256×256 / 4096×4096 — 全部崩溃 |
| 纹理通道 | RGB / RGBA / float32 — 全部崩溃 |
| Open3D 版本 | 0.18.0 / 0.19.0 — 全部崩溃 |
| CPU / CUDA 构建 | 两个 pybind 均崩溃 |
| 硬件/软件 GL | NVIDIA 595.84 驱动 / MESA 软件渲染 — 全部崩溃 |
| 数据 | `mixamo_sequence.npz` 校验完全正常 |

本质是 Open3D 自带的 GLFW 窗口刷新回调在触发纹理上传时越界访问。

### 解决方案

用 **viser + nerfview + Filament 离屏渲染** 替代 legacy Visualizer 的窗口渲染：

- **viser**：在浏览器里提供轨道相机 + GUI 面板，通过 WebSocket 通信
- **nerfview**：把 viser 的相机状态传给 Python 端的 `render_fn`，渲染结果以 JPEG 推回浏览器
- **Filament OffscreenRenderer**：Open3D 的另一套渲染后端（EGL 无头模式），不经过 GLFW 窗口系统，**带纹理渲染完全正常**

---

## 离线回放

用于查看预先拟合好的 Mixamo 序列（`mixamo_sequence.npz`）。

### 依赖

`gsplat` 环境（已含 `viser`、`nerfview`、`open3d`）。

### 使用

```bash
cd /home/fr1511b/program/workspace/humanpose
conda activate gsplat

EGL_PLATFORM=surfaceless PYTHONPATH=src \
python scripts/view_mixamo_sequence_viser.py \
    --results-dir outputs/sequences/4_pointcloud_exit_gate
```

浏览器打开 `http://127.0.0.1:8090`。

### 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--results-dir` | `outputs/sequences/4_pointcloud_exit_gate` | 包含 `mixamo_sequence.npz` 的目录 |
| `--mixamo-cache` | 自动推导 | 显式指定 npz 路径 |
| `--port` | `8090` | 本地服务端口 |
| `--fps` | `30` | 播放帧率 |
| `--res` | `1024` | 渲染分辨率 |

### 浏览器操作

- 鼠标拖拽：环绕相机
- 滚轮：缩放
- Play 按钮：播放 / 暂停
- Time 滑块：跳转帧
- Speed 滑块：播放速度
- Loop 复选框：循环播放
- Reset View 按钮：恢复初始视角

---

## 实时观看（双窗口）

在原有的 Open3D 火柴人窗口之外，增加一个浏览器窗口显示带纹理的 Mixamo
假人，实时跟随骨骼运动。两个渲染器使用独立进程：父进程只创建 GLFW
火柴人窗口，子进程只创建 Filament/EGL 引擎，避免 Open3D 原生图形上下文
冲突导致 `eglMakeCurrent failed` 和段错误。

### 依赖

依赖已经固定在 `environment.yml`。对于尚未重建的现有
`rgbd-avatar` 环境，需要补装：

```bash
conda activate rgbd-avatar
pip install viser==1.0.30 nerfview==0.1.3 \
    splines==0.3.3 jaxtyping==0.3.7
```

### 使用

```bash
cd /home/fr1511b/program/workspace/humanpose
conda activate rgbd-avatar

PYTHONPATH=src python scripts/view_live_mannequin.py \
    --mixamo-cache outputs/sequences/4_pointcloud_exit_gate/mixamo_sequence.npz
```

启动后：

- **Open3D 窗口**保持原样（火柴人 / RGB 图），全部键盘控制（M 切换样式、Q 退出）不变
- **浏览器**打开 `http://127.0.0.1:8095`，显示带纹理的 Mixamo 假人

两个窗口各自独立运行。没有相机实时源时，可加 `--source directory`，从
`live.yaml` 配置的目录读取 RGB-D 帧来模拟实时流；管线不会从 SDK 自动回退。

实时命令不需要设置 `EGL_PLATFORM`。子进程会在导入 Open3D 前自行设置
`EGL_PLATFORM=surfaceless`；如果父终端残留了这个变量，主管线会在打开
GLFW 窗口前将它移除。

### 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--mixamo-cache` | 无 | `mixamo_sequence.npz` 路径（提供 IK profile、scale、FBX 路径） |
| `--mixamo-viewer-port` | `8095` | Mixamo 浏览器窗口端口 |
| `--mixamo-res` | `1024` | 渲染分辨率 |

不传 `--mixamo-cache` 时行为与原来完全一致（只开 Open3D 窗口）。

### 原理

```
相机 → RTMPose → LivePoseResult
                      ├─→ LiveMannequinRenderer (Open3D 窗口，不动)
                      └─→ 精简 PosePacket（26 个关节，不传 RGB）
                              ↓ multiprocessing.Queue(maxsize=2)
                   spawn 子进程: ViserLiveMixamoViewer
                              │
                   update: 进程间和进程内都只保留最新姿态
                              ↓
                   求解线程: 传感器时间戳
                           → MixamoAnalyticalIK.solve()（每输入帧至多一次）
                           → skin_mixamo_vertices()
                           → 发布最新顶点版本并触发 rerender
                              ↓
                   渲染线程: 顶点版本变化时才重新上传网格
                           → Filament OffscreenRenderer
                           → JPEG → viser → 浏览器
```

IK 求解的 profile 从 `mixamo_sequence.npz` 元数据读取，scale 使用缓存中
实际拟合时采用的 `cache.scale`（包括 `--scale` 覆盖值）。FBX 资产按元数据
中记录的路径加载，无需重新标定。

实时链路采用 latest-only 策略：浏览器渲染速度低于相机帧率时会跳过过期
的待显示姿态，不会累积延迟；下一次渲染始终读取最近完成的顶点。相机旋转
或 nerfview 的静态高质量重绘不会重复推进 IK。

首次得到有效蒙皮顶点后，查看器使用实时人物包围盒（不是 FBX 绑定姿态）
自动设置相机中心和距离；人物退出并重新进入画面后会再次对焦。浏览器中的
`Reset View` 也会恢复到最近一次实时人物包围盒。

正常启动并收到第一帧后，子进程日志依次包含：

```text
Live Mixamo child received first pose: ...
Live Mixamo camera focused on frame=... center=... distance=... m
Live Mixamo mesh uploaded: version=... vertices=... triangles=...
```

如果只出现第一条而没有第二条，说明 IK 尚未得到有效根关节；如果出现第二条
而没有第三条，说明浏览器尚未建立渲染连接。

---

## 工程结构

| 文件 | 作用 |
|---|---|
| `src/rgbd_avatar/visualization/viser_mixamo_viewer.py` | 离线序列查看器（viser + nerfview + Filament 离屏） |
| `scripts/view_mixamo_sequence_viser.py` | 离线查看器 CLI |
| `tests/test_viser_mixamo_viewer.py` | 离线查看器测试（3 个） |
| `src/rgbd_avatar/visualization/viser_live_mixamo.py` | 实时 Mixamo 渲染器（IK + 蒙皮 + Filament 离屏 + viser） |
| `src/rgbd_avatar/visualization/live_mixamo_process.py` | 隔离 GLFW/Filament 的 spawn 子进程控制器与精简姿态消息 |
| `src/rgbd_avatar/pipeline/live_mannequin.py` | 实时管线（新增 `--mixamo-cache` 三参数 + 双渲染器集成） |
| `tests/test_live_mixamo_process.py` | 子进程消息、非阻塞 latest-only 队列测试 |

两个 `scripts/` 文件仍是薄 CLI 入口；实时集成位于
`pipeline/live_mannequin.py`。

---

## 已知限制

1. **光照偏暗**：光线强度参数在此 Open3D 构建中无效，角色亮度约 35/255。背景与光照无 `--brightness` 参数可调，等待后续优化。
2. **Filament 进程隔离**：此 Open3D 构建无法可靠地在 GLFW 窗口所在进程
   创建 Filament EGL 引擎。实时查看器固定使用 `spawn` 子进程；不要把它改回
   与 `LiveMannequinRenderer` 同进程。
3. **nerfview 的分辨率切换** 会触发引擎重建；实时查看器改为固定分辨率渲染，离线查看器仅当窗口宽高比变化时重建。
4. 实时模式的 IK/蒙皮在独立 latest-only 线程中按传感器时间戳推进，每个
   输入帧至多求解一次，不阻塞 Open3D 窗口；Filament/JPEG 较慢时只降低
   浏览器显示帧率，不会形成待渲染帧队列。
