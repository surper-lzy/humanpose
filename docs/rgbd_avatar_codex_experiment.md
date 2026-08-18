# 基于 RGB-D 相机的实时人体骨架驱动与仿生人同步系统

## 1. 项目概述

本项目使用一台 RGB-D 相机实时获取人体的：

- RGB 图像
- 深度图
- 强度图
- 三维点云

系统从 RGB 图像中提取人体二维骨架关键点，再结合深度图或有组织点云恢复米制三维骨架；随后估计人体身高、地面平面和根节点全局轨迹，并将人体动作实时重定向到一个公开的标准骨架仿生人模型。

第一阶段不处理 3D Gaussian Splatting 场景，先完成单独的 RGB-D 人体动作驱动闭环。后续再将仿生人与静态 3DGS 场景进行尺度和坐标对齐。

## 2. 当前已确认需求

### 2.1 硬件

- 单台 RGB-D 相机
- 相机可输出：
  - RGB 图
  - 深度图
  - 强度图
  - 点云
- NVIDIA GPU
- RGB-D 相机默认固定安装
- 人体在相机视野范围内移动

### 2.2 功能目标

系统需要实现：

1. RGB 图像中的人体骨架实时提取。
2. 由深度图恢复米制三维人体关键点。
3. 从点云中估计人体真实身高。
4. 从点云中估计地面平面。
5. 获取人体根节点的真实空间位置和移动轨迹。
6. 将人体局部动作映射到标准骨架仿生人。
7. 根据真人身高对仿生人进行整体统一缩放。
8. 真人向前移动 2 米时，仿生人在虚拟空间中也移动约 2 米。
9. 实现脚底贴地、减少穿地和滑步。
10. 系统骨架提取和驱动达到 30 FPS。
11. 使用 Python 为主，后续可通过 CUDA、TensorRT 或 C++ 扩展优化。

### 2.3 当前不包含

第一阶段暂不处理：

- 激光雷达
- 多传感器外参标定
- 多相机系统
- 手指动作
- 面部表情
- 多人跟踪
- 动态 3DGS
- 3DGS 场景训练

### 2.4 当前数据与相机标定状态

当前离线数据位于：

```text
/home/fr1511b/program/workspace/data/
├── 1/
├── 2/
├── 3/
└── 4/
```

同一时间戳的数据文件约定为：

```text
*_r.png  RGB 图
*_d.pgm  深度图
*_a.pgm  强度图
*_t.pcd  相机软件导出的点云
```

文件名前缀 `YYYYMMDD_HHMMSSmmm` 表示采集时间，其中末三位为毫秒。
连续序列必须按前缀排序，并以完全相同的前缀严格配对 `*_r.png` 与
`*_d.pgm`；不使用“最近时间戳”近似配对。滤波时间步长使用相邻
文件名中的真实时间差，不假定固定帧率。不同编号目录视为独立序列，
不跨目录延续滤波状态。

当前四组数据分别包含 `43 / 52 / 29 / 35` 帧，共 `159` 帧；各组
中位帧间隔约为 `0.504 s`，即约 `1.98 FPS`。组 1 内有一个
`193.318 s` 的采集中断，组 3 内有一个 `2.619 s` 的采集中断。
当前连续帧基线使用 `2.0 s` 作为断点阈值，断点处切分序列并重置
时序状态。因此这批离线数据可用于正确性验证，但不能用于验收
`30 FPS` 实时采集目标。

当前 RGB 图与深度图的分辨率均为 `816 × 612`。相机输出的 RGB
与深度已经完成像素级对齐，且当前图像已经去畸变。后续处理不得再次
对图像执行去畸变，也不需要在 RGB 和深度之间重复进行图像配准。

设备软件给出的 2D 和 3D 有效内参相同，参数顺序为
`fx, fy, cx, cy`：

```text
fx = 390.697235107
fy = 390.601348877
cx = 408.284454346
cy = 321.971252441
```

对应内参矩阵：

\[
\mathbf{K}=
\begin{bmatrix}
390.697235107 & 0 & 408.284454346\\
0 & 390.601348877 & 321.971252441\\
0 & 0 & 1
\end{bmatrix}
\]

2D 和 3D 的畸变模型均显示为 `RADTAN_5`，但当前有效畸变系数均为：

```text
k1 = 0
k2 = 0
p1 = 0
p2 = 0
k3 = 0
```

设备软件给出的 2D/3D 对齐后有效外参为：

\[
\mathbf{R}_{3D\leftarrow2D}=
\begin{bmatrix}
1 & 0 & 0\\
0 & 1 & 0\\
0 & 0 & 1
\end{bmatrix},
\qquad
\mathbf{t}_{3D\leftarrow2D}=
\begin{bmatrix}
0\\
0\\
0
\end{bmatrix}
\]

这组单位旋转和零平移描述的是当前已配准输出的有效坐标关系，不应
反向推断为设备内部两个物理成像器件的原始机械外参。

相机在当前应用/安装坐标系中的外参由软件界面记录为：

```text
Roll  =   70.61 degree
Pitch = -179.79 degree
Yaw   =   90.04 degree

x =    0.00 mm
y =    0.00 mm
z = 1783.28 mm
```

内部几何计算时，平移统一转换为米：

```text
t_application = [0.0, 0.0, 1.78328] m
```

相机采用右手坐标系：

```text
+X：向右
+Y：向下
+Z：向前
```

欧拉角采用 `ZYX` 旋转顺序，对应 `Yaw-Pitch-Roll`。对于列向量，
若记：

```text
yaw   = ψ
pitch = θ
roll  = φ
```

则旋转矩阵按以下顺序构造：

\[
\mathbf{R}
=
\mathbf{R}_z(\psi)
\mathbf{R}_y(\theta)
\mathbf{R}_x(\phi)
\]

即先对局部向量应用 Roll，再应用 Pitch，最后应用 Yaw。角度输入
计算函数前必须从 degree 转换为 radian。

## 3. 系统总体流程

```text
RGB-D 相机
├── RGB 图像
├── 深度图
├── 强度图
└── 点云
        ↓
相机数据同步与单位统一
        ↓
RGB 人体检测与二维骨架提取
        ↓
关键点深度提取与三维反投影
        ↓
三维骨架滤波、补点和骨长约束
        ↓
人体点云分割
        ↓
地面、身高、根节点与人体朝向估计
        ↓
标准骨架动作重定向
        ↓
仿生人统一缩放
        ↓
根节点米制移动同步
        ↓
足部接触、Foot Lock 与腿部 IK
        ↓
GPU 仿生人渲染
        ↓
后续接入静态 3DGS 场景
```

## 4. 推荐技术选型

### 4.1 二维人体姿态估计

优先方案：

- RTMPose-S 或 RTMPose-M
- MMPose
- 后续部署到 ONNX / TensorRT

可选基线：

- OpenPose BODY_25

OpenPose 可以直接输出身体和足部二维关键点，适合作为经典基线；但最终实时系统更建议使用 RTMPose，以便更容易接入 PyTorch、CUDA 和 TensorRT。

### 4.2 人体分割

可选：

- YOLO 系列人体检测框
- YOLO-Seg
- MediaPipe Selfie Segmentation
- SAM2 仅用于离线验证，不建议第一版直接作为实时主模块

第一版可以只使用人体检测框加深度范围过滤；第二版再加入人体分割掩膜。

### 4.3 点云与几何处理

推荐：

- Open3D
- NumPy
- PyTorch
- 后续可加入 NVIDIA Warp 或自定义 CUDA Kernel

### 4.4 仿生人模型

优先：

- SMPL 标准人体模型
- 或公开的带骨骼绑定的 FBX / GLB / VRM 模型

第一版建议统一内部骨架接口为 24 关节或 25 关节格式。

### 4.5 渲染

第一版：

- PyTorch3D
- Open3D 可视化
- pyrender

性能优化版：

- nvdiffrast
- CUDA 光栅化
- 自定义 OpenGL / Vulkan 渲染

## 5. 建议项目目录结构

```text
rgbd_avatar/
├── README.md
├── requirements.txt
├── configs/
│   ├── camera.yaml
│   ├── pose.yaml
│   ├── tracking.yaml
│   └── avatar.yaml
├── assets/
│   ├── avatar/
│   └── calibration/
├── data/
│   ├── recordings/
│   └── outputs/
├── src/
│   ├── camera/
│   │   ├── base_camera.py
│   │   ├── rgbd_camera.py
│   │   ├── frame_types.py
│   │   └── recorder.py
│   ├── pose/
│   │   ├── pose_estimator.py
│   │   ├── rtmpose_backend.py
│   │   ├── openpose_backend.py
│   │   └── keypoint_formats.py
│   ├── depth/
│   │   ├── depth_sampler.py
│   │   ├── deprojection.py
│   │   └── depth_quality.py
│   ├── pointcloud/
│   │   ├── human_segmentation.py
│   │   ├── ground_plane.py
│   │   ├── height_estimator.py
│   │   └── torso_tracker.py
│   ├── tracking/
│   │   ├── one_euro_filter.py
│   │   ├── root_tracker.py
│   │   ├── bone_constraints.py
│   │   └── occlusion_recovery.py
│   ├── retargeting/
│   │   ├── skeleton_map.py
│   │   ├── retargeter.py
│   │   ├── forward_kinematics.py
│   │   ├── foot_contact.py
│   │   └── two_bone_ik.py
│   ├── avatar/
│   │   ├── avatar_model.py
│   │   ├── smpl_avatar.py
│   │   ├── skinning.py
│   │   └── renderer.py
│   ├── utils/
│   │   ├── geometry.py
│   │   ├── timing.py
│   │   ├── logging.py
│   │   └── visualization.py
│   └── main.py
├── scripts/
│   ├── inspect_camera_streams.py
│   ├── record_rgbd.py
│   ├── test_pose_2d.py
│   ├── test_pose_3d.py
│   ├── estimate_height.py
│   ├── test_retargeting.py
│   └── benchmark_pipeline.py
└── tests/
    ├── test_deprojection.py
    ├── test_ground_plane.py
    ├── test_height_estimator.py
    ├── test_retargeting.py
    └── test_root_motion.py
```

## 6. 统一数据结构

建议定义统一帧结构：

```python
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass
class RGBDFrame:
    timestamp: float
    rgb: np.ndarray
    depth: np.ndarray
    intensity: Optional[np.ndarray]
    point_cloud: Optional[np.ndarray]
    intrinsics: CameraIntrinsics
    depth_scale: float
```

约定：

- RGB：`H x W x 3`，BGR 或 RGB 必须在配置中固定。
- 深度：`H x W`。
- 深度内部可以保留原始整数值，但所有几何计算前必须转换为米。
- 点云：建议统一为 `N x 3`，单位为米。
- 有组织点云可以保留为 `H x W x 3`。
- 所有帧必须带时间戳。
- 当前数据的 `H=612`、`W=816`。
- 当前 RGB 和深度已对齐并去畸变，同一像素 `(u, v)` 可以直接建立
  RGB、深度和内部有组织点云之间的对应关系。
- 当前深度 PGM 和相机导出 PCD 使用毫米，读取后统一乘以 `0.001`
  转换为米。
- 相机导出的 PCD 为 `HEIGHT=1` 的无组织点云；内部应优先从深度图
  重新生成 `H x W x 3` 有组织点云，以支持通过 `(v, u)` 查询三维点。
- 相机三维坐标系为右手系：`+X` 向右、`+Y` 向下、`+Z` 向前。

## 7. 二维关键点数据结构

```python
@dataclass
class Keypoint2D:
    x: float
    y: float
    confidence: float


@dataclass
class Pose2D:
    keypoints: np.ndarray  # shape: [K, 3]
    bbox: np.ndarray       # [x1, y1, x2, y2]
    score: float
    timestamp: float
```

三维关键点：

```python
@dataclass
class Pose3D:
    joints: np.ndarray      # shape: [K, 3], unit: meter
    confidence: np.ndarray  # shape: [K]
    valid: np.ndarray       # shape: [K], bool
    timestamp: float
```

## 8. RGB-D 三维骨架恢复

### 8.0 当前数据的对齐前提

当前 RGB 和深度已经由相机软件完成去畸变与像素级对齐，二者分辨率
同为 `816 × 612`，并使用相同的有效内参。因此第一阶段不需要求解
RGB 到深度的额外配准变换。

处理流程应为：

1. 读取对齐后的 RGB 与 16 位深度 PGM。
2. 将原始毫米深度乘以 `0.001` 转换为米。
3. 使用当前有效内参生成 `H x W x 3` 有组织点云。
4. 在 RGB 上获得二维骨架关键点 `(u, v)`。
5. 使用相同像素位置的邻域深度恢复三维关键点。

尽管图像已经对齐，仍需处理无效深度、人体轮廓混入背景以及单相机
自遮挡；“已对齐”不等于每个二维关节都有可靠深度。

### 8.1 深度反投影

对于二维关键点：

\[
\mathbf{k}_i=(u_i,v_i,c_i)
\]

从深度图中获得：

\[
Z_i=D(u_i,v_i)
\]

通过相机内参反投影：

\[
X_i=\frac{(u_i-c_x)Z_i}{f_x}
\]

\[
Y_i=\frac{(v_i-c_y)Z_i}{f_y}
\]

得到三维点：

\[
\mathbf{p}_i=[X_i,Y_i,Z_i]^T
\]

### 8.2 不要直接读取单个深度像素

禁止只使用：

```python
z = depth[int(v), int(u)]
```

推荐方法：

1. 以关键点为中心建立 5×5 或 7×7 窗口。
2. 排除深度为 0、NaN、Inf 的像素。
3. 排除超出有效距离的像素。
4. 如果存在人体掩膜，只保留人体区域。
5. 使用中位数作为深度。
6. 若局部深度跨度过大，认为关键点落在深度边缘。
7. 边缘关键点可沿骨骼方向向人体内部偏移后再次采样。
8. 深度无效时使用上一帧预测或相邻骨骼约束。

### 8.3 建议函数接口

```python
def sample_joint_depth(
    depth_m: np.ndarray,
    u: float,
    v: float,
    mask: np.ndarray | None = None,
    radius: int = 3,
    min_depth: float = 0.2,
    max_depth: float = 8.0,
) -> tuple[float, float]:
    """返回深度值和深度置信度。"""
```

```python
def deproject_pixel(
    u: float,
    v: float,
    z: float,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """将像素与深度转换为相机坐标系三维点，单位为米。"""
```

### 8.4 当前实现：有组织点云局部表面反演

当前已经保留两套可切换方法：

- `window_median`：原始 baseline，在二维关键点周围直接做深度窗口
  中位数和最近深度簇选择。
- `pointcloud_cluster`：当前默认方法，每帧先通过内参生成一次
  `H × W × 3` 有组织点云，再从局部三维表面恢复关节。

`pointcloud_cluster` 的处理顺序为：

1. 根据 RTMDet 人体检测框建立候选区域。
2. 使用 `neck`、`hip`、左右肩和左右髋的高置信局部点簇估计躯干
   参考深度。
3. 用检测框和宽松的躯干深度带建立代理人体掩膜。
4. 根据检测框高度为关节选择自适应圆形邻域；首次没有有效簇时扩大
   半径一次。
5. 对候选点执行八邻域区域生长。只有图像相邻且三维欧氏距离小于
   自适应阈值的点才属于同一个连通表面。
6. 综合点簇距二维关键点中心的距离、点数、深度 MAD、局部占比和
   躯干深度一致性选择点簇。
7. 对选中点簇求空间加权中位深度 \(\hat Z_i\)，然后仍在原始二维
   关键点射线上反投影：

\[
\mathbf{p}_i=
\operatorname{deproject}(u_i,v_i,\hat Z_i,K)
\]

因此最终关节重新投影后仍严格回到 RTMPose 的亚像素坐标
`(u_i, v_i)`。点簇中的真实表面 medoid 只作为诊断证据单独保存，
不直接代替关节坐标，避免三维质心导致 RGB 与骨架投影错位。

当前没有人体实例分割模型，所以检测框加深度带只能称为
`bbox_depth_proxy`，不能称为精确人体 mask。接口已经接受可选
`person_mask`；后续接入实例分割后会自动与检测框、深度带和局部
圆盘求交。

若两个不同深度表面的选择分数几乎相同，或局部没有足够点，不输出
高置信错误原始关节，而是标记 `ambiguous_clusters`、
`no_valid_depth` 或 `no_supported_cluster`，交给现有时序滤波做
短时预测。逐关节诊断还记录搜索半径、是否扩半径、候选点数、点簇
数、选中点数、表面 medoid、深度 MAD、选择 margin 和深度置信度。

## 9. 三维骨架滤波与约束

### 9.1 分层处理

- 2D 关键点：置信度门控。
- 3D 位置：One Euro Filter。
- 骨长：固定或缓慢更新。
- 根节点：卡尔曼滤波或指数平滑。
- 骨骼旋转：四元数球面插值或指数平滑。
- 丢失关键点：短时恒速度预测。

### 9.2 骨长约束

初始化阶段统计稳定站立时的骨长：

\[
L_{ij}=\|\mathbf{p}_j-\mathbf{p}_i\|_2
\]

运行时把异常骨长投影回合理范围：

\[
\mathbf{p}'_j=\mathbf{p}_i+
L_{ij}
\frac{\mathbf{p}_j-\mathbf{p}_i}
{\|\mathbf{p}_j-\mathbf{p}_i\|_2}
\]

只对低置信度关节执行强约束，避免过度破坏真实动作。

当前实现采用以下更保守的约束边界：

1. 骨长标定只读取尚未滤波和尚未约束的原始 `Pose3D` 真实观测，
   不使用 `predicted` 或约束后的坐标，避免反馈污染。
2. 骨段两端均要求二维关键点置信度不低于 `0.6`、深度置信度不低于
   `0.7`。每条骨段至少需要 `12` 个内点，以 `20` 个内点为冻结
   目标，最多收集 `30` 个候选。
3. 使用中位数建立临时骨长先验，并排除偏差超过
   `max(0.02 m, 0.12 × median)` 的样本；相对 MAD 必须不高于
   `0.10`。
4. 第一版只约束 14 条核心身体骨段。面部连线包含独立三角环，足部
   深度仍受地面和轮廓影响，因此面部与足部暂不进入骨长求解。
5. 运行时使用置信度加权的 Jacobi 迭代投影。置信度不低于 `0.55`
   的 observed 关节保持为不可移动锚点；髋中心 `hip(19)` 始终固定，
   只允许低置信度 observed 或 predicted 关节移动。
6. missing 关节保持 missing，不由骨长约束凭空生成；单帧低置信度
   observed 修正上限为 `0.08 m`，predicted 修正上限为 `0.12 m`，
   并保留原始状态来源和修正量。

当前数据是动态序列，因此得到的是“单人序列临时骨长先验”，不能直接
称为稳定站立解剖标定，也不等价于人体身高。更严格的个人骨长标定仍
应采集独立的稳定直立片段。

## 10. 人体点云分割

推荐第一版流程：

1. 从 RGB 中获取人体检测框或人体掩膜。
2. 将检测框或掩膜映射到深度图。
3. 根据图像像素与有组织点云的对应关系提取人体候选点。
4. 根据深度范围过滤背景。
5. 使用欧式聚类保留主要人体点云簇。
6. 删除地面点。

如果点云与 RGB 已经对齐，可直接通过像素掩膜提取：

```python
human_points = organized_point_cloud[person_mask]
```

如果点云不是有组织点云，需要通过相机 SDK 提供的映射接口完成 RGB、深度和点云之间的对应。

当前完成状态需要区分：

- 已完成：每帧有组织点云、检测框裁剪、躯干参考深度带、局部三维
  连通表面聚类，以及外部 `person_mask` 接口。
- 尚未完成：真正的人体实例分割、整个人体点云的全局欧式聚类、
  地面删除和遮挡人体分离。

因此阶段 2 使用的是“人体候选点云过滤”，阶段 3 的“人体点云分割”
仍保持未完成。

## 11. 地面平面估计

地面平面表示为：

\[
\mathbf{n}^T\mathbf{p}+d=0
\]

推荐实现：

- 对点云下半部分进行候选筛选。
- 使用 RANSAC 拟合平面。
- 选择法向接近竖直方向、面积较大的平面。
- 连续多帧平滑地面参数。

建议接口：

```python
@dataclass
class Plane:
    normal: np.ndarray  # shape [3], normalized
    offset: float
    confidence: float


def estimate_ground_plane(points: np.ndarray) -> Plane:
    ...
```

点到地面距离：

\[
h(\mathbf{p})=\mathbf{n}^T\mathbf{p}+d
\]

## 12. 身高估计

人体身高沿地面法向计算：

\[
H=Q_{0.99}(\mathbf{n}^TP_h)-Q_{0.01}(\mathbf{n}^TP_h)
\]

其中：

- \(P_h\) 为人体点云。
- \(Q_{0.99}\) 和 \(Q_{0.01}\) 为分位数。

不要直接使用点云最大值减最小值，避免离群点影响。

### 12.1 身高有效条件

只有满足以下条件时才更新身高：

- 人体基本直立。
- 头顶点云可见。
- 双脚接近地面。
- 膝盖没有明显弯曲。
- 人体点云完整度较高。
- 关键点置信度较高。

收集 30～90 帧有效身高后取中位数并冻结。

### 12.2 仿生人缩放

\[
s_{avatar}=\frac{H_{person}}{H_{avatar-original}}
\]

缩放同时应用到：

- 网格顶点
- 骨骼偏移
- 根节点高度
- 足底偏移

## 13. 根节点和全局移动

骨盆中心：

\[
\mathbf{p}_{pelvis}
=
\frac{\mathbf{p}_{left\ hip}+\mathbf{p}_{right\ hip}}{2}
\]

躯干点云中心：

\[
\mathbf{p}_{torso}
=
\operatorname{centroid}(P_{torso})
\]

融合根节点：

\[
\mathbf{p}_{root}
=
w_s\mathbf{p}_{pelvis}+w_p\mathbf{p}_{torso}
\]

其中权重由骨架置信度和点云完整度动态决定。

人体位移：

\[
\Delta\mathbf{p}(t)
=
\mathbf{p}_{root}(t)-\mathbf{p}_{root}(0)
\]

仿生人根节点：

\[
\mathbf{t}_{avatar}(t)
=
\mathbf{t}_{avatar}(0)+\Delta\mathbf{p}(t)
\]

要求：真人移动约 2 米时，仿生人移动误差尽量小于 10 厘米，后续根据设备实际精度调整。

## 14. 人体朝向估计

可使用肩部和髋部构造人体横向轴：

\[
\mathbf{x}_{body}
=
\operatorname{normalize}
\left(
\frac{\mathbf{p}_{right\ shoulder}-\mathbf{p}_{left\ shoulder}}{2}
+
\frac{\mathbf{p}_{right\ hip}-\mathbf{p}_{left\ hip}}{2}
\right)
\]

竖直轴使用地面法向：

\[
\mathbf{y}_{body}=\mathbf{n}_{ground}
\]

前向轴：

\[
\mathbf{z}_{body}
=
\operatorname{normalize}
(\mathbf{x}_{body}\times\mathbf{y}_{body})
\]

需要检查坐标系手性并根据实际相机坐标系调整叉乘顺序。

## 15. 动作重定向

### 15.1 输入输出

输入：

- 米制三维人体关键点
- 标准骨架静止姿态
- 骨骼父子关系
- 真人身高
- 根节点位置与朝向

输出：

- 仿生人根节点位置
- 仿生人根节点旋转
- 各关节局部旋转
- 仿生人统一缩放系数

### 15.2 骨骼方向

\[
\mathbf{d}_i
=
\frac{\mathbf{p}_{child}-\mathbf{p}_i}
{\|\mathbf{p}_{child}-\mathbf{p}_i\|}
\]

计算参考方向到当前方向的旋转。

单个骨骼方向只能确定摆动旋转，无法确定绕骨骼轴的扭转。因此肩、上臂、大腿、躯干等需要使用多个关节点建立局部坐标系。

### 15.3 推荐求解顺序

1. 根节点位置和朝向。
2. 骨盆。
3. 脊柱和胸部。
4. 颈部和头部。
5. 左右大腿、小腿和脚。
6. 左右上臂、前臂和手腕。
7. 世界旋转转为父骨骼局部旋转。
8. 关节角限制。
9. 正向运动学验证。

## 16. 足部接触与防滑步

脚部接触条件：

\[
c_f(t)=
\left(h_f<h_{th}\right)
\land
\left(\|\mathbf{v}_f\|<v_{th}\right)
\]

其中：

- \(h_f\)：脚部到地面的距离。
- \(\mathbf{v}_f\)：脚部速度。

脚部接触后：

1. 记录接触脚世界坐标。
2. 在接触期间锁定脚部水平位置。
3. 使用两骨骼 IK 调整髋、膝、踝。
4. 修正根节点高度。
5. 脚部离地后解除锁定。

第一版可以只做：

- 根节点高度修正
- 脚底不穿地

第二版再加入完整 Foot Lock 和腿部 IK。

## 17. 强度图使用策略

第一阶段不把强度图作为主姿态输入。

建议用途：

- 深度质量判断
- 弱光条件辅助
- 反光或低反射区域检测
- 人体边缘辅助
- 深度无效区域分析

需要先确认相机输出的“强度图”具体代表：

- 红外强度
- 灰度
- 反射强度
- 置信度图

如果 SDK 另有置信度图，应优先把置信度用于深度筛选。

## 18. 实时系统设计

目标：30 FPS。

单帧预算：

\[
1000/30\approx33.3\text{ ms}
\]

建议时间预算：

| 模块 | 目标耗时 |
|---|---:|
| 相机取帧与预处理 | 1～3 ms |
| 人体检测与二维姿态 | 8～15 ms |
| 深度采样与三维恢复 | 1～3 ms |
| 点云和地面处理 | 3～7 ms |
| 骨架滤波与根节点跟踪 | 1～2 ms |
| 动作重定向与 IK | 1～3 ms |
| 蒙皮和渲染 | 4～8 ms |

### 18.1 异步线程建议

```text
线程 A：相机采集
线程 B：二维人体姿态推理
线程 C：深度、点云和三维骨架
线程 D：动作重定向
线程 E：渲染和显示
```

使用带时间戳的环形缓冲区。

渲染线程读取最近一组完整状态，不等待所有模块严格同步完成。

## 19. 配置文件建议

`configs/camera.yaml`

```yaml
camera:
  model: "FILL_CAMERA_MODEL"
  sdk: "FILL_CAMERA_SDK"
  rgb_width: 816
  rgb_height: 612
  depth_width: 816
  depth_height: 612
  fps: 30
  depth_unit: "millimeter"
  depth_scale: 0.001
  min_depth_m: 0.3
  max_depth_m: 6.0
  align_depth_to_rgb: true
  images_undistorted: true
  internal_point_cloud_organized: true
  exported_pcd_organized: false

  intrinsics:
    fx: 390.697235107
    fy: 390.601348877
    cx: 408.284454346
    cy: 321.971252441

  distortion:
    model: "RADTAN_5"
    coefficients: [0.0, 0.0, 0.0, 0.0, 0.0]

  aligned_extrinsics:
    rotation:
      - [1.0, 0.0, 0.0]
      - [0.0, 1.0, 0.0]
      - [0.0, 0.0, 1.0]
    translation_m: [0.0, 0.0, 0.0]

  application_pose:
    roll_deg: 70.61
    pitch_deg: -179.79
    yaw_deg: 90.04
    translation_m: [0.0, 0.0, 1.78328]
    euler_order: "ZYX"
    euler_sequence: "Yaw-Pitch-Roll"
    coordinate_system: "right_handed"
    axis_x: "right"
    axis_y: "down"
    axis_z: "forward"
```

这里的 `fps: 30` 是目标实时采集帧率，不代表当前离线数据的实际
保存频率。`application_pose` 使用 `ZYX` 欧拉角顺序，即
`R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`。

`configs/pose.yaml`

```yaml
pose:
  backend: "rtmpose"
  model_size: "m"
  keypoint_format: "halpe26"
  detector: "auto"
  bbox_threshold: 0.3
  keypoint_threshold: 0.3
  min_valid_keypoints: 10
  min_mean_keypoint_score: 0.3
  single_person: true
```

`configs/tracking.yaml`

```yaml
tracking:
  one_euro:
    min_cutoff_hz: 0.5
    beta: 2.0
    derivative_cutoff_hz: 1.0
  reset_gap_s: 2.0
  max_prediction_s: 1.1
  min_observation_confidence: 0.05
  bone_statistics:
    min_joint_confidence: 0.05
  bone_constraint:
    enabled: true
    calibration:
      min_samples_per_bone: 12
      target_samples_per_bone: 20
      max_samples_per_bone: 30
      min_keypoint_confidence: 0.6
      min_depth_confidence: 0.7
      max_relative_mad: 0.10
      outlier_relative_tolerance: 0.12
      outlier_absolute_tolerance_m: 0.02
      min_length_m: 0.03
      max_length_m: 1.2
    solver:
      anchor_confidence: 0.55
      iterations: 6
      max_joint_correction_m: 0.08
      max_predicted_correction_m: 0.12
      fixed_joint_indices: [19]
```

这组 One Euro 参数针对当前约 `2 FPS` 的离线数据；以后接入
`30 FPS` 实时流时需要重新调参。骨长 profile 在同一单人序列的
时间断点之间保留，但更换序列或人物身份时必须重置。当前 runner
仍按最高检测框分数选择单人，尚未实现多人身份跟踪，因此不能跨人物
混合标定。

`configs/avatar.yaml`

```yaml
avatar:
  type: "smpl"
  model_path: "assets/avatar/model.pkl"
  original_height_m: 1.75
  enable_uniform_scale: true
  enable_root_motion: true
  enable_ground_constraint: true
  enable_foot_lock: false
```

## 20. 开发阶段划分

### 阶段 0：相机接口验证

目标：确认四种输出能正确读取。

任务：

- [ ] 读取 RGB 图
- [ ] 读取深度图
- [ ] 读取强度图
- [ ] 读取点云
- [ ] 获取相机内参
- [ ] 获取深度单位
- [ ] 确认 RGB 与深度是否对齐
- [ ] 确认点云是否有组织
- [ ] 记录帧时间戳
- [ ] 录制一段原始数据用于离线调试

验收：

- 连续运行 5 分钟无崩溃。
- 四种数据可同步显示。
- 深度和点云距离一致。

### 阶段 1：二维人体骨架

任务：

- [x] 集成 RTMPose-M Halpe26
- [x] 使用 RTMDet 进行单人检测
- [x] 关键点可视化
- [x] 输出关键点置信度
- [ ] 测量推理 FPS

验收：

- 站立、抬手、下蹲时关键点基本正确。
- GPU 推理达到或接近 30 FPS。

当前实现：

- `src/rgbd_avatar/pose/rtmpose_backend.py`
- `scripts/test_rtmpose_single.py`
- `configs/pose.yaml`
- 已验证正面站立、远处移动人物和无人场景。
- 已拦截 MMPose 在无人检测结果下自动使用全图框的 fallback，避免空场景
  输出低置信度假骨架。

### 阶段 2：三维人体骨架

任务：

- [x] 深度邻域采样
- [x] 2D 到 3D 反投影
- [x] 深度置信度
- [x] 无效深度处理
- [x] 三维骨架静态可视化
- [x] 连续帧时间戳解析与 RGB/深度严格配对
- [x] 单次初始化 RTMPose 后顺序处理整段数据
- [x] One Euro Filter
- [x] 短时缺失预测及 observed/predicted/missing 状态区分
- [x] 骨长稳健统计
- [x] 高置信原始观测的临时骨长标定
- [x] 低置信度/预测关节的骨长约束
- [x] 每帧深度图生成有组织点云
- [x] 检测框加躯干深度带的代理人体过滤
- [x] 关节点局部三维连通表面聚类
- [x] 点簇深度沿原二维射线反投影
- [x] 耳部候选点簇的脸部深度与同侧 eye-ear 拓扑门控
- [x] 点云恢复逐关节诊断与 baseline 切换
- [x] 输出逐帧 JSONL、分段叠加视频和序列汇总
- [x] Open3D 连续彩色点云与三维骨架交互回放
- [x] 由 Halpe26 三维关节实时生成程序化假人

验收：

- 三维骨架与人体点云基本重合。
- 静止时关节抖动显著降低。
- 前后移动时深度变化合理。

当前实现：

- `src/rgbd_avatar/depth/`
- `src/rgbd_avatar/depth/pointcloud_recovery.py`
- `src/rgbd_avatar/visualization/pose3d.py`
- `src/rgbd_avatar/visualization/sequence3d.py`
- `src/rgbd_avatar/avatar/procedural.py`
- `scripts/test_pose_3d_single.py`
- `scripts/view_pose3d_sequence.py`
- `configs/camera.yaml`
- `window_median` 保留原来的 `7 × 7` 邻域、有效深度过滤和中位数
  baseline。
- 默认 `pointcloud_cluster` 使用完整有组织点云、代理人体深度
  mask、自适应局部圆盘和三维连通表面聚类。
- 人体轮廓上的耳部关键点额外使用已经恢复的鼻、眼深度中位数和
  同侧 `eye-ear` 三维距离过滤候选表面；如果局部只有不合理的背景
  表面，则保留 missing，而不把墙面伪装成高置信三维关节。
- 新方法最终用点簇的稳健深度在原二维关键点射线上反投影，因此不会
  因为改用点云质心而破坏 RGB/骨架像素对应。
- 几何 golden 帧 `20260730_145911656` 生成 `497347` 个米制点；
  该帧的历史 `window_median` baseline 恢复 `26/26` 个关节。
- 该 golden 帧的自重建点云与同帧相机导出 PCD 点数一致，点对点
  RMSE 约为
  `2.27e-7 m`，最大误差约为 `8.51e-7 m`。

连续序列公共模块：

- `src/rgbd_avatar/data/sequence.py`：发现、校验并按文件名时间戳
  排序 RGB-D 帧，严格配对 RGB 与深度，并按长时间间隔切分序列。
- `src/rgbd_avatar/tracking/one_euro.py`：按关节对米制三维坐标
  执行 One Euro 滤波；XYZ 共用由三维速度模长确定的截止频率。
  无效关节不更新观测状态，短时缺失使用恒速度预测并衰减置信度，
  超时后输出 missing。
- `src/rgbd_avatar/tracking/bone_statistics.py`：统计有效骨段长度的
  样本数、中位数、MAD 和相对 MAD。
- `src/rgbd_avatar/tracking/bone_constraints.py`：从高置信原始
  RGB-D 观测建立鲁棒临时骨长先验，并通过置信度加权 Jacobi 投影
  修正低置信度或预测关节；高置信锚点和髋根保持不动。
- `scripts/process_rgbd_sequence.py`：模型只初始化一次，逐帧输出
  `pose3d_raw`、`pose3d_temporal` 和 `pose3d_constrained` 三层
  JSONL，并输出分段叠加视频、manifest 和汇总 JSON。视频中紫色
  标记表示骨长约束产生的三维修正投影。
- `configs/tracking.yaml`：保存当前离线数据的 One Euro、短时预测和
  断点参数。

第 4 组 `window_median` 历史 baseline 结果：

- 输入和输出均为 `35` 帧，逐帧 JSONL 没有漏帧。
- `34/35` 帧检出人体，检出率约 `97.1%`。
- 共 `875` 个通过阈值的二维关节，深度恢复成功 `875` 个，当前
  有效二维关节条件下的三维恢复率为 `100%`。
- 倒数第二帧只有 `17` 个原始三维关节，最后一帧未检出人体；时序
  输出仍保留短时预测，并通过 `observed`、`predicted`、`missing`
  三种状态避免把预测值误写为真实观测。
- 25 条 Halpe26 骨段的长度相对 MAD 中位数从原始观测的
  `0.0646` 降至滤波观测的 `0.0539`，约下降 `16.6%`。该指标只
  说明 One Euro 滤波减少了骨长波动。
- 14 条核心身体骨段最终全部形成可用临时先验，其中 `9/14` 已达到
  `20` 个鲁棒内点并冻结；从第 12 帧开始逐步启用，整段共有
  `24` 帧具有至少一条可用骨长先验。
- 骨长投影在 `13` 帧中修正了 `35` 个关节帧，其中 `19` 个为
  低置信度 observed，`16` 个为 predicted。修正量中位数约
  `0.0358 m`，总体 P95 和最大值为 predicted 安全上限
  `0.12 m`。低置信度 observed 的图像投影改变量中位数约
  `2.09 px`、P95 约 `26.39 px`。
- 对至少一端允许移动的骨段，逐帧骨长相对误差中位数的序列中位值
  从 `9.77%` 降至 `5.00%`；逐帧 P95 相对误差的序列中位值从
  `11.14%` 降至 `5.00%`。
- 高置信度锚点最大位移为 `0 m`，髋根最大位移为 `0 m`，没有出现
  骨向量翻转。累计仍有 `69` 条锚点两端均为高置信度但骨长超出
  允许范围的观测；安全策略只记录这些 violation，不强行移动锚点。
  这说明后续仍需人体掩膜和骨长感知的深度异常拒绝。
- 骨长标定和约束平均约 `1.6 ms/frame`。当前尚未使用独立静止片段
  评测静态关节抖动，也没有估计人体身高。
- 本次 Codex 执行环境使用 CPU：RTMPose/RTMDet 平均推理约
  `0.29 s/frame`，深度恢复约 `2 ms/frame`，端到端约 `3.3 FPS`。
  这只是 CPU 离线基线，距离 `30 FPS` 目标仍需 CUDA、ONNX 或
  TensorRT 验证与优化。

第 4 组 `pointcloud_cluster` 验证结果：

- 输入和输出仍为 `35/35` 帧，人体检出仍为 `34/35`，说明几何
  恢复切换没有改变二维检测流程。
- 固定使用同一批已保存的二维关键点进行 A/B：`875` 个有效二维
  关节中恢复 `852` 个，原始三维恢复率为 `97.37%`。其中
  `17` 个局部没有代理人体深度点，`6` 个存在近似同分的不同深度
  表面；这些关节不伪造原始观测，而由时序层短时预测。
- 第 `15` 帧 `right_wrist` 的 baseline 深度为 `3.528 m`，实际
  取到了后方表面。新方法在半径 `9 px` 的局部点云中选中
  `16` 点人体簇，深度变为 `2.009 m`，`right_elbow-right_wrist`
  距离从约 `1.577 m` 变为 `0.207 m`。
- 第 `21` 帧 `right_wrist` 的 baseline 深度为 `3.754 m`。新方法
  扩展到 `13 px` 后选中 `39` 点人体簇，深度变为 `2.655 m`，
  前臂距离从约 `1.024 m` 变为 `0.261 m`。
- 原始骨长相对 MAD 中位数约为 `0.0639`，时序滤波后约为
  `0.0549`。该结果只说明骨长波动下降，不等价于解剖关节误差。
- CPU 完整点云生成通常约为 `3–5 ms/frame`；当前纯 Python
  八邻域点云聚类通常约为 `57–67 ms/frame`。这足够当前约
  `2 FPS` 的离线数据，但几何模块本身尚未达到未来 `30 FPS`
  目标，后续需要向量化、Numba、C++ 或 GPU 聚类。
- 当前仍有侧身髋部误差：深度相机观察到的是人体可见表面，不是
  解剖关节中心。下一步应把髋部左右关系、时序预测和个人骨长先验
  加入点簇评分，而不是继续扩大人体深度带。

第 4 组耳部轮廓异常门控实验：

- 原因不是 RTMPose 把二维耳点画到了墙上，而是二维耳点靠近人体
  轮廓时，局部圆盘同时包含头部和后墙两个三维连通表面。旧评分对
  “离二维点更近、点数更多”的墙面簇给分更高，宽松的全身深度带
  又没有把它排除，因此耳点会跳到后墙。
- 当前实验只对 Halpe26 的 `left_ear/right_ear` 启用门控。至少
  两个鼻/眼锚点有效时，以其深度中位数作为脸部参考，候选点簇需要
  同时满足脸部深度差不超过
  `max(0.20 m, 0.08 × face_depth)`，且同侧 `eye-ear` 距离不超过
  `0.25 m`。通过门控的候选仍沿原二维耳点射线反投影，不会改变
  RGB 像素对应。
- 合成的“`2.00 m` 人体表面 + `2.55 m` 后墙”自动测试中，关闭
  门控时耳点选择后墙，开启后改选人体表面；如果搜索半径内没有
  合规人体簇，测试确认该关节状态为 `joint_topology_rejected`。
- 在第 4 组 `35` 帧上，旧结果共有 `12` 条有效
  `eye-ear > 0.25 m` 的异常连线，新结果为 `0`。门控对 `68`
  个有效检测帧耳点生效，排除 `23` 个不合理候选簇，并使 `13`
  个耳点的选择深度改变超过 `5 cm`；该序列没有耳点因无可行簇而
  被丢弃。
- 截图对应的第 `18` 帧 `20260730_150522286` 中，
  `left_eye-left_ear` 从 `0.821 m` 降为 `0.161 m`，左耳深度从
  后墙的 `3.402 m` 改选为头部表面的 `2.766 m`。
- 原始三维有效关节总数从 `852` 变为 `853`。除两个耳点外，其余
  `24` 类关节的 valid 状态和 XYZ 逐项不变，说明该实验没有扩大到
  身体其他部位。
- 这仍是局部拓扑安全门，不等于人体实例分割。它不处理足尖贴地、
  侧身髋部可见表面偏差或严重遮挡；这些问题下一步仍应通过真正的
  人体 mask、时序/个人骨长先验和足部接触模型解决。

第 4 组逐帧关节位置与异常连接排查：

- 新增 `scripts/analyze_pose3d_sequence.py`。它不修改骨架结果，而是
  展开记录每帧 26 个关节的 RGB 二维坐标、raw/temporal/constrained
  三层 XYZ、原始深度与置信度、躯干参考深度、局部点簇数量、选择
  裕量和恢复状态；同时记录三层的 25 条连接长度与两端深度差。
- 当前 `35` 帧共产生 `910` 行关节记录和 `2625` 行连接记录。
  基于整段序列的骨长中位数/MAD，raw 层标出 `16` 条异常连接、
  `4` 个相对躯干深度异常和 `7` 个相邻有效帧深度跳变。这些是无
  真值情况下的排查线索，不直接等价于解剖误差。
- 第 `5` 帧 `20260730_150515786` 的 `left_eye` 是确认的局部表面
  选择错误：当前选中 `2.770 m` 点簇，局部另有 `2.645 m` 候选，
  而同帧鼻子和右眼分别约为 `2.650 m`、`2.649 m`。两个候选的
  原评分约为 `0.9382` 和 `0.8923`，只差约 `4.9%`；现有拓扑门只
  约束耳点，因此没有纠正眼点。
- 第 `0–3` 帧的主要异常位于左脚。例如第 `1` 帧左脚踝选择
  `3.884 m`，局部同时存在 `3.677 m` 候选；第 `2` 帧选择
  `3.911 m`，另有 `3.668 m` 候选。脚趾/脚跟又分别落在不同地面
  表面，形成 `0.2–0.45 m` 的异常足部连接。
- 第 `14` 帧 `right_shoulder` 选择 `1.629 m` 的大前景点簇，
  局部仍有约 `1.906 m` 的肩部候选。RGB 中抬起的右臂遮挡肩部，
  因此这是自遮挡时选择了可见手臂表面，不是后墙吸附。
- 第 `25` 帧左手与左髋在图像中重叠，`left_hip` 取到约
  `1.835 m` 的前景手/髋表面，而中心 `hip` 取到 `2.081 m`，
  使 `left_hip→hip` 达到 `0.255 m`。这同样属于自遮挡和解剖
  关节不可见问题。
- 第 `32` 帧右脚接近图像底部，右脚踝约为 `1.410 m`，两个右脚趾
  只能找到约 `1.82–1.84 m` 的地面点簇，导致两条连接约为
  `0.58 m`。此时没有可重排的人体候选，正确策略应是拒绝脚趾
  原始观测并交给时序层短时预测。
- 第 `33` 帧人物大幅出画，多个足部二维置信度低于阈值；
  `right_knee→right_hip` 仅 `0.071 m` 主要是二维关键点在图像
  边界塌缩，不应归因于深度关联。
- 第 `26–27` 帧左臂相对躯干深度变化较大，但 RGB 显示人物确实
  向前/向上摆臂，因此这部分属于稳健统计的动作假阳性，不能据此
  收紧全身深度带。

逐帧诊断输出位于：

```text
outputs/sequences/4_pointcloud_face_gate/diagnostics/
├── joint_positions.csv
├── bone_connections.csv
├── frame_summary.csv
├── anomalies.json
└── report.md
```

当前排查结论支持把后续修复分成三条独立规则：面部鼻眼耳联合深度
一致性、足部 ankle-toe/heel 拓扑与地面拒绝、以及肩髋自遮挡时使用
时序/骨长先验。不能再用一个更严的全身深度阈值同时处理三类问题。

第 4 组三类局部拓扑规则实现与 A/B 结果：

- 三类决策已从点云遍历代码中拆出到
  `src/rgbd_avatar/depth/topology.py`。该模块只接收带 XYZ、局部
  评分和人体一致性评分的候选，不依赖 Open3D 或图像 I/O，便于单元
  测试和后续替换点云聚类实现。点云恢复层只负责缓存每个关节的候选
  簇、调用拓扑选择器并写回结果，避免重复搜索邻域。
- 鼻、左眼、右眼不再逐点贪心选择，而是在每个关节最多 `5` 个局部
  表面候选和“缺失”状态之间联合搜索。可行组合必须满足脸部深度跨度、
  `nose-eye ≤ 0.15 m`、`eye-eye ≤ 0.18 m`，并拒绝明显位于颈部
  后方的表面。鼻眼确定后，再用最终脸部共识重新选择左右耳。第 `5`
  帧左眼因此从错误的 `2.770 m` 改选为 `2.645 m`，与鼻子和右眼
  的约 `2.65 m` 深度一致。
- 每只脚把 knee、ankle、big toe、small toe、heel 作为一个拓扑
  单元，其中 ankle/toe/heel 的候选联合评分，knee 作为稳定锚点。
  当前约束为 `knee-ankle ≤ 0.75 m`、`ankle-toe ≤ 0.30 m`、
  `ankle-heel ≤ 0.22 m`、两脚趾间距 `≤ 0.20 m`。脚趾或脚跟
  没有合规人体表面时允许保持缺失，宁可交给时序预测，也不强制吸附
  到地面。第 `0–3` 帧的左脚踝/脚趾已改选到相互一致的人体表面；
  第 `32` 帧两个约 `1.82–1.84 m` 的右脚地面候选被拒绝。
- 肩部以 neck、左右 shoulder 为组，髋部以 hip center、左右 hip
  为组。只有一侧同时超过绝对骨长上限并比另一侧长 `1.6` 倍以上时
  才拒绝，避免把对称的大动作误判成遮挡。当前肩颈上限为 `0.32 m`，
  髋中心到单侧髋上限为 `0.22 m`。被拒绝的 raw 观测不会参与滤波
  更新，而是由已有时序状态预测，再由个人骨长先验有限幅度修正。
- 第 `14` 帧错误的右肩前景观测 `z=1.629 m` 被拒绝；时序层恢复到
  `z=1.763 m`，骨长层进一步得到 `z=1.783 m`。第 `25` 帧错误的
  左髋 `z=1.835 m` 同样被拒绝，时序/骨长层分别恢复到约
  `2.127 m/2.117 m`。两处都保留了预测和修正标记，未伪装成真实
  深度观测。
- 对同一批已保存的二维姿态重放后，raw 骨长异常从 `16` 条降到
  `2` 条，相邻有效帧深度跳变从 `7` 降到 `3`，低选择裕量的多
  表面记录从 `67` 降到 `25`。剩余两条骨长异常都在第 `33` 帧，
  原因是人物出画后 knee/hip 二维关键点塌缩，不是深度选面错误。
- 新结果共有 `851` 个有效 raw 三维关节；另外明确记录了 `3` 个
  `foot_topology_ground_rejected` 和 `2` 个
  `self_occlusion_topology_rejected`。缺失是门控的预期安全输出，
  因而不能只用 raw 有效点数量评价改进。
- 逐帧 CSV 已增加三类门控的启用状态、候选序号、联合目标值、可行
  组合数、足部拒绝原因，以及肩髋左右骨长。纯拓扑、点云集成和已有
  流水线测试共 `59 passed`。

三类拓扑规则的独立输出位于：

```text
outputs/sequences/4_pointcloud_topology/
├── manifest.json
├── poses.jsonl
├── summary.json
└── diagnostics/
    ├── joint_positions.csv
    ├── bone_connections.csv
    ├── frame_summary.csv
    ├── anomalies.json
    └── report.md
```

第 4 组人物出画终止规则：

- “某个关节被身体遮挡”和“人物正在离开画面”采用不同策略。前者
  仍允许短时预测；后者终止整条 track，raw、temporal 和
  constrained 三层都不再输出骨架。
- 不能只用检测框碰边判断。第 `11`、`12`、`32` 帧的检测框都接触
  图像下边界，但二维姿态仍有 `26` 个有效点和 `8` 个有效足部点，
  因而继续处理。第 `33` 帧检测框碰底的同时，有效点降到 `17`、
  足部有效点变为 `0`、平均关键点置信度降到约 `0.446`，三项证据
  联合确认人物已经部分出画。
- 当前门控参数位于 `configs/tracking.yaml`：边界范围 `2 px`，
  碰边时至少需要 `20` 个有效关键点、平均置信度至少 `0.55`；
  上边界至少保留 `2` 个面部点，下边界至少保留 `4` 个足部点。
  这些条件只在检测框碰边时生效，不会把画面内部的普通遮挡当作
  人物离场。
- 第 `33` 帧状态现在为 `person_partially_out_of_frame`，触发时立即
  清空每个关节的滤波位置、速度和预测寿命，并暂停骨长先验更新。
  第 `33`、`34` 帧的 raw、temporal、constrained 均为 `0` 个
  关节，不再显示离场后的橙色预测骨架。
- 门控触发后会锁定“等待完整重新入画”状态。仍然贴着边界的检测不
  会重新启动骨架；只有检测框完全回到画面内部才建立新轨迹，并在
  新轨迹首次接受观测前重置个人骨长标定。
- 每帧 JSONL 和 `joint_positions.csv` 都记录 `frame_presence` 的
  接受状态、碰到的边界、失败条件和是否触发轨迹重置。加入该规则后
  第 4 组 raw 骨长统计异常从拓扑版本的 `2` 条降为 `0` 条；项目
  当时测试为 `64 passed`，加入后续纯骨架空间后当前为
  `72 passed`。

人物出画门控的独立输出位于：

```text
outputs/sequences/4_pointcloud_exit_gate/
├── manifest.json
├── poses.jsonl
├── summary.json
└── diagnostics/
    ├── joint_positions.csv
    ├── bone_connections.csv
    ├── frame_summary.csv
    ├── anomalies.json
    └── report.md
```

正式输出位于：

```text
outputs/sequences/4/
├── manifest.json
├── poses.jsonl
├── summary.json
└── segment_000_overlay.mp4
```

本次点云方法的独立验证输出位于：

```text
outputs/sequences/4_pointcloud/
├── manifest.json
├── poses.jsonl
└── summary.json
```

耳部轮廓门控的独立 A/B 输出位于：

```text
outputs/sequences/4_pointcloud_face_gate/
├── manifest.json
├── poses.jsonl
└── summary.json
```

运行当前三类拓扑规则的命令：

```bash
conda activate rgbd-avatar
python scripts/process_rgbd_sequence.py \
  --sequence ../data/4 \
  --recovery-method pointcloud_cluster \
  --output-dir outputs/sequences/4_pointcloud_exit_gate \
  --device cuda:0
```

如果当前机器没有可用 CUDA，可将 `--device cuda:0` 改为
`--device cpu`。视频是固定帧率预览，精确时间以 JSONL 中的
`relative_time_s` 和 `dt_s` 为准。

如需与旧方法做严格 A/B，应使用不同输出目录，避免覆盖：

```bash
python scripts/process_rgbd_sequence.py \
  --sequence ../data/4 \
  --recovery-method window_median \
  --output-dir outputs/sequences/4_window_median

python scripts/process_rgbd_sequence.py \
  --sequence ../data/4 \
  --recovery-method pointcloud_cluster \
  --output-dir outputs/sequences/4_pointcloud
```

连续三维点云与骨架回放：

```bash
cd /home/fr1511b/program/workspace/humanpose
conda activate rgbd-avatar

PYTHONPATH=src python scripts/view_pose3d_sequence.py \
  --poses-jsonl outputs/sequences/4_pointcloud_exit_gate/poses.jsonl \
  --pose-layer raw
```

播放器直接复用 `poses.jsonl`，不会再次初始化 RTMPose。每帧根据
`sources.rgb` 和 `sources.depth` 读取已对齐数据，以相机内参从
深度图重建有组织点云，再将已经恢复的 Halpe26 三维关节和 25 条
骨架连线画在同一 Open3D 场景中。默认只显示二维检测框加
`30 px` 边界内的点云，以便观察骨架是否贴合人体。检测短时丢失时
最多复用相邻 `1.1 s` 内的检测框，超过后自动显示完整点云，且不会
跨采集分段传播；按 `B` 可手动切换完整场景点云。若序列开头暂时
没有检测到人体，播放器会先显示场景，并在第一帧有效骨架出现时
自动聚焦一次。

新生成的 `manifest.json` 会保存本次处理实际使用的相机内参、
深度比例、有效深度范围和对齐状态。播放器会将该快照与当前
`configs/camera.yaml` 核对，防止配置改变后把旧骨架和错误点云
静默叠加。已有的 `4_pointcloud` 输出早于相机快照字段，播放时会
给出提示；该结果必须继续使用生成它时的当前相机配置。

三种骨架层的用途不同：

- `raw`：红色，直接由 RGB 二维关键点和局部点云表面反演，适合
  检查“二维点是否正确落在点云上”，也是默认层。
- `temporal`：真实观测为绿色，短时预测为橙色，适合观察连续性和
  抖动。
- `constrained`：在 temporal 基础上加入骨长约束；被约束移动的
  关节和相邻骨段显示为紫色。若结果中没有该层，自动回退到
  temporal，并在终端显示实际层名。

交互按键：

- `Space`：播放或暂停。
- 左右方向键或 `A/D`：前后单帧。
- `R/T/C`：切换 raw、temporal、constrained 骨架层。
- `M`：循环切换骨架、程序化假人、骨架与假人同时显示。
- `G`：恢复 RGB 图像提供的原始彩色点云。
- `X/Y/Z`：按显示坐标的 X/Y/Z 数值显示伪彩色点云。
- `O`：恢复 Open3D 默认点云颜色。
- `N`：法线着色；当前点云未计算法线时不建议使用。
- `B`：循环切换人物检测框点云、完整场景点云和无点云。
- `K`：显示或隐藏米制参考地面网格。
- `L`：切换循环播放。
- `F`：恢复正对相机的观察视角。
- `H`：在终端打印帮助。
- `Q`：退出。

主键盘和小键盘的 `0–9` 均由播放器拦截并恢复 RGB 点云，避免
Open3D 内置数字着色快捷键造成无法可靠返回的状态。项目自身的点云
着色统一使用 `G/X/Y/Z/O/N`。

播放器不带 `--results-dir/--poses-jsonl` 参数启动时，默认读取
`outputs/sequences/4_pointcloud_exit_gate`，窗口标题也会显示实际
结果目录。旧目录 `4_pointcloud`、`4_pointcloud_face_gate` 和
`4_pointcloud_topology` 分别缺少耳部门控、三类联合规则或人物出画
终止规则，不能用它们判断当前完整流水线是否生效。

所有保存数据继续使用右手相机坐标系
`(+X 右、+Y 下、+Z 前)`。存在 `ground_plane.json` 时，播放器对
显示副本应用刚体重力对齐，使 Open3D 场景为
`(+X 右、+Y 前、+Z 上)` 且真实地面为 `Z=0`；没有地面标定或显式
使用 `--no-ground-alignment` 时，才退回固定旋转
`(X,Y,Z) -> (X,Z,-Y)`。两种变换行列式都为 `+1`，不改变米制
距离，也不会改写 JSONL 中的相机坐标。

纯三维运动骨架模式：

```bash
cd /home/fr1511b/program/workspace/humanpose
conda activate rgbd-avatar

PYTHONPATH=src python scripts/view_pose3d_sequence.py \
  --skeleton-only
```

`--skeleton-only` 默认使用 constrained 层、关闭点云并开启
`0.25 m` 间隔的参考地面网格。该模式只读取 `poses.jsonl` 中已经
恢复的三维关节，不读取 RGB、深度，也不执行点云反投影；因此显示
的是一个独立、米制的运动骨架空间。坐标仍是 `X` 向右、`Y` 向前、
`Z` 向上，骨架在相机空间中的真实前后移动会被保留，不会把髋部
强制锁到原点。

查看器会自动读取结果目录中的 `ground_plane.json`，把骨架和点云
统一变换到真实地面 `Z=0`；文件不存在时才退回整段 constrained
足部最低值中位数生成的旧参考网格。按 `K` 可隐藏网格，按 `B`
可以在纯骨架、人物点云和完整点云之间切换。若要检查未经时序约束
的纯 raw 骨架，可以显式增加 `--pose-layer raw`；若不需要网格，
可以增加 `--no-ground-grid`。

程序化假人模式：

```bash
cd /home/fr1511b/program/workspace/humanpose
conda activate rgbd-avatar

PYTHONPATH=src python scripts/view_pose3d_sequence.py \
  --mannequin
```

该模式不需要下载 SMPL、FBX、GLB 或纹理资源，直接复用当前环境中的
NumPy 和 Open3D。`src/rgbd_avatar/avatar/procedural.py` 把地面对齐后
的 constrained Halpe26 关节转换成一组与渲染器无关的几何规格：

- 上臂、前臂、大腿和小腿使用沿骨段放置的渐缩体，近端粗、远端
  细，不再呈现等粗管状外观。
- 胸腔、腹部和骨盆使用相互重叠、随肩髋方向旋转的椭球体；头部为
  无五官的中性椭圆体，并补充独立颈部。
- 肩、肘、髋、膝和踝使用收小的圆润关节盖；手部沿前臂方向生成
  扁长椭球体，不再用腕部球体代替手掌。
- 全身默认使用统一哑光浅灰色。需要检查关节来源时按 `M` 切到
  `both`，由内部彩色骨架显示观测、预测和约束状态。
- 双脚依据 heel、big toe、small toe 的三维位置估计朝向和长度；存在
  `ground_plane.json` 时只抬升脚部显示体积以避免穿过 `Z=0`，不会
  篡改原始关节坐标或骨长。
- 单个关节缺失时只跳过受影响的部件；人物出画后整帧为 0 个部件，
  不沿用上一帧假人。

`--mannequin` 是“constrained、无点云、开启地面网格、只显示假人”
的快捷方式。运行中按 `M` 可切换为骨架或两者同时显示。若要在人物
点云中检查假人的贴合程度，使用：

```bash
PYTHONPATH=src python scripts/view_pose3d_sequence.py \
  --pose-layer constrained \
  --render-style both
```

无窗口环境可验证全部假人规格而不创建 Open3D 窗口：

```bash
PYTHONPATH=src python scripts/view_pose3d_sequence.py \
  --mannequin \
  --validate-only
```

当前第 4 组的第 `0–32` 帧均生成 `27` 个部件，第 `33–34` 帧人物
出画后生成 `0` 个部件。程序化假人目前是骨架的体积化诊断显示，
尚不包含蒙皮、衣服变形、局部关节旋转、Foot Lock 或腿部 IK；这些
仍属于标准角色动作重定向阶段。

真实地面 RANSAC 与重力对齐：

```bash
PYTHONPATH=src python scripts/estimate_ground_plane.py \
  --results-dir outputs/sequences/4_pointcloud_exit_gate
```

脚本从 `16` 个均匀采样的深度帧下半区域提取候选，排除当前人物框，
再使用地面法向、相机高度和距离内点三重约束执行确定性 RANSAC。
参数位于 `configs/ground.yaml`，结果保存为：

```text
outputs/sequences/4_pointcloud_exit_gate/ground_plane.json
```

第 4 组结果为：

- 相机坐标地面向上法向约
  `(0.00229, -0.94139, -0.33732)`。
- 相机离地高度约 `1.8000 m`。
- 光学相机上方向与真实地面法向相差 `19.71°`，这正是旧水平网格
  随人物前后移动产生大幅悬空/穿地的主要原因。
- `50,000` 个候选中有 `35,239` 个地面内点，内点比例
  `70.48%`；平面残差中位数约 `8.6 mm`、P95 约 `21.6 mm`。
- 对齐前逐帧最低脚点相对旧网格约为 `-0.292～+0.394 m`；对齐后
  相对真实地面约为 `-0.0129～+0.0315 m`，最低脚点绝对误差 P95
  约 `3.09 cm`。

这一阶段只纠正了地面几何和坐标系，没有把脚强行吸附到平面。剩余
约厘米级误差来自深度噪声、鞋底/关键点定义和时序滤波；下一阶段
需要左右脚接触状态、鞋底偏移和保持骨长的腿部 IK。

无窗口环境可验证纯骨架链路：

```bash
PYTHONPATH=src python scripts/view_pose3d_sequence.py \
  --skeleton-only \
  --validate-only
```

当前 `35` 帧验证中点云点数始终为 `0`，第 `0–32` 帧 constrained
骨架均为 `26` 个关节和 `25` 条骨段，第 `33–34` 帧人物出画后为
`0` 个关节；整个验证不读取 RGB-D，耗时约 `0.02 s`。

无桌面图形环境时，可以先执行完整数据链路验收而不创建窗口：

```bash
PYTHONPATH=src python scripts/view_pose3d_sequence.py \
  --poses-jsonl outputs/sequences/4_pointcloud_exit_gate/poses.jsonl \
  --pose-layer raw \
  --validate-only
```

当前第 4 组新输出 `35` 帧的 raw 已通过无窗口验收，且相机标定
快照与当前配置一致。
通过出画门控的检测帧，raw 骨架包含 `22–26` 个有效关节；第 `33`
帧人物部分出画以及第 `34` 帧无人时，raw、temporal、constrained
都正确显示 `0` 个关节。检测框点云在默认
`point_stride=3` 下约为 `4114–16601` 点/帧。包括点云裁剪、
坐标旋转、三层状态、动态几何、播放分段和标定一致性在内的项目
测试当前为 `81 passed`。

### 阶段 3：地面、身高与根节点

任务：

- [ ] 人体点云分割
- [x] 地面 RANSAC
- [ ] 身高估计
- [ ] 骨盆中心估计
- [ ] 躯干点云中心估计
- [ ] 根节点滤波
- [ ] 米制轨迹可视化

验收：

- 身高误差目标小于 5 厘米。
- 2 米移动测试误差目标小于 10 厘米。
- 地面高度稳定。

### 阶段 4：仿生人动作重定向

任务：

- [x] 程序化几何假人（骨架体积化诊断 baseline）
- [x] SMPL Neutral 独立加载、前向计算与网格拓扑冒烟测试
- [x] 加载标准骨架模型
- [x] 定义 Halpe26 到 SMPL 的身体关节映射
- [x] 优化 SMPL 局部旋转
- [x] 优化根节点平移和朝向
- [x] 序列级统一米制缩放
- [x] SMPL 正向运动学
- [x] 线性混合蒙皮和 Open3D 序列渲染

验收：

- 抬手、弯腿、转身动作方向正确。
- 真人向前移动时仿生人同步移动。
- 不出现明显骨骼翻转。

SMPL Neutral 独立验证：

```bash
cd /home/fr1511b/program/workspace/humanpose
conda activate rgbd-avatar

# 无窗口完整检查
python scripts/test_smpl_model.py --validate-only

# 单独打开 SMPL 零姿态网格
python scripts/test_smpl_model.py
```

原始模型位于 `assets/models/smpl/SMPL_NEUTRAL.pkl`。其 `shapedirs`
字段是旧 Chumpy 对象；使用以下一次性命令生成 NumPy-only 副本：

```bash
python scripts/convert_smpl_model.py
```

脚本始终保留原文件，并使用临时文件、原子落盘和零姿态数值对比；
若目标 `SMPL_NEUTRAL_CLEAN.pkl` 已存在则拒绝覆盖。转换完成后，
`test_smpl_model.py` 默认优先加载清洗副本。若需要专门复测原文件，
使用：

```bash
python scripts/test_smpl_model.py \
  --model assets/models/smpl/SMPL_NEUTRAL.pkl \
  --legacy-chumpy-compat \
  --validate-only
```

当前独立测试确认：

- `smplx` 成功加载为 `SMPL`，包含 `10` 个 shape coefficients。
- 零姿态输出 `6890` 个有限顶点、`13776` 个合法三角形和 `45` 个
  加载器输出关节；后续映射主要使用其中的标准 SMPL 身体关节。
- 网格包含 `20664` 条唯一边，无边界边、非流形边、退化三角形或
  重复三角形，Euler characteristic 为 `2`。
- Open3D 在零姿态检测到局部表面自相交，因此其严格
  `is_watertight()` 返回 false；这不阻碍显示和线性混合蒙皮，但若
  后续用于碰撞或物理仿真，需要先处理自相交表面。
- 零姿态最大轴向尺寸约 `1.7451 m`，人体尺度合理。
- 当前 CPU 上连续 `20` 次前向计算平均约 `1.3 ms/帧`；当前环境
  未检测到 CUDA，但这不影响独立可用性结论。

清洗过程只转换了 `1` 个 Chumpy 对象。清洗前后零姿态的全部顶点和
关节最大差异均为 `0.000e+00 m`，面拓扑完全一致；在不注入任何旧
NumPy 别名的新进程中，清洗模型可以直接加载，且不会导入 Chumpy。
SMPL 文件受独立模型许可约束，已经通过 `.gitignore` 排除，不能随
项目代码提交或分发。

#### SMPL 接入三维骨架流水线

当前使用三阶段设计：先在 Halpe26 的肘—腕引导框中提取 Hand21，
再将受约束的身体骨架和手部关键点共同拟合并缓存成 SMPL 序列，
最后由查看器快速播放。这样交互播放时不需要逐帧执行模型推理或
梯度优化。

首次处理一组新序列时运行：

```bash
cd /home/fr1511b/program/workspace/humanpose
conda activate rgbd-avatar

PYTHONPATH=src python scripts/extract_hand_pose_sequence.py \
  --results-dir outputs/sequences/4_pointcloud_exit_gate

PYTHONPATH=src python scripts/fit_smpl_sequence.py \
  --results-dir outputs/sequences/4_pointcloud_exit_gate
```

第一条命令输出 `hands.jsonl` 和 `hands_manifest.json`；第二条默认读取
`pose3d_constrained`、`hands.jsonl` 和同目录的 `ground_plane.json`，
输出 `smpl_sequence.npz`。如果 RGB-D、骨架或地面标定更新，需要
依次显式重建：

```bash
PYTHONPATH=src python scripts/extract_hand_pose_sequence.py \
  --results-dir outputs/sequences/4_pointcloud_exit_gate \
  --overwrite

PYTHONPATH=src python scripts/fit_smpl_sequence.py \
  --results-dir outputs/sequences/4_pointcloud_exit_gate \
  --overwrite
```

只看纯净空间中的 SMPL Neutral 动画：

```bash
PYTHONPATH=src python scripts/view_pose3d_sequence.py \
  --results-dir outputs/sequences/4_pointcloud_exit_gate \
  --smpl
```

若 SMPL 网格产生恐怖谷观感，可切换为完全程序化的火柴人模式：

```bash
PYTHONPATH=src python scripts/view_pose3d_sequence.py \
  --results-dir outputs/sequences/4_pointcloud_exit_gate \
  --stickman \
  --start-paused
```

火柴人直接读取同一套 `pose3d_constrained`，不加载 SMPL 模型，也不
需要重新拟合或下载资源。造型采用经典黑色粗线风格：统一近黑色圆柱
表示肢体、肩线、髋线和单根躯干轴，同色小球只用于形成平滑的圆角
连接，头部为独立黑色球体。查看器默认使用浅灰背景并隐藏地面网格和
彩色坐标轴；需要空间参照时可追加 `--ground-grid`。脚部使用踝点到
足部中心的短杆。每个完整帧共 `29` 个几何部件；人物部分出画或离场
帧输出为空。
按 `M` 可在火柴人、来源骨架以及二者叠加之间切换，用于区分骨架输入
异常与虚拟人物表现异常。也可以用完整参数形式
`--mannequin --avatar-model stickman` 启动。

将 SMPL、来源骨架和 RGB-D 点云同时显示，用于检查贴合关系：

```bash
PYTHONPATH=src python scripts/view_pose3d_sequence.py \
  --results-dir outputs/sequences/4_pointcloud_exit_gate \
  --avatar-model smpl \
  --pose-layer constrained \
  --render-style both \
  --cloud-scope bbox
```

其中 `M` 在纯骨架、SMPL 网格以及两者同时显示之间循环。按
`R/T/C` 仍可切换骨架诊断层，但缓存的 SMPL 网格保持为拟合时使用
的 constrained 层，日志会明确提示这一点。

单独调试身体和手部三维骨架点：

```bash
PYTHONPATH=src python scripts/view_pose3d_sequence.py \
  --results-dir outputs/sequences/4_pointcloud_exit_gate \
  --skeleton-only
```

查看器检测到同目录的 `hands.jsonl` 时默认显示 Hand21；可用
`--no-show-hands` 暂时隐藏。身体关节球半径默认 `2.5 cm`，手部为
`0.9 cm`，可通过 `--hand-joint-radius-m` 调整。左手为青色、右手为
黄色；手部模型腕点偏离 Halpe 腕点超过裁剪框 `32%` 的结果会直接
丢弃。其余缺腕、指尖不足、手长异常、掌宽异常、掌面长轴与横轴近似
共线、多个相邻指骨塌缩或单节指骨异常拉长的结果在调试显示中标成红色，
同时不会进入 SMPL 拟合。掌面规则使用 wrist、index/middle/pinky MCP
构造方向坐标系，不会因为正常的握拳动作而把指尖位置误判为掌面异常。
当前序列手部显示范围为每帧 `0–42` 个点和 `0–40` 条连接。

本序列中原先仅检查腕到指尖的整体长度，因此第 0、3、4、5、6、8、10、
12、16、17、21、23、24、27、28、32 帧中的部分手即使缺少掌面点、掌宽
只有数毫米、掌面近似共线或相邻指骨重合，仍会以青色/黄色进入显示和腕部
方向拟合。补充掌面与指骨拓扑门控后，这些观测会明确记录到 SMPL cache 的
`hand_rejection_counts` 和 `hand_rejections_by_frame`；拟合器在相应侧不再
使用错误手部方向，而由身体腕点、上一帧姿态和时序正则维持连续性。

纯三维查看器的默认前视相机不再使用全部骨架点的坐标中位数作为注视点；
腿脚关节数量较多时该中位数偏低，会形成长期仰视感。现在优先使用 SMPL
网格（否则使用骨架）的 `2%–98%` 鲁棒高度范围，将注视点放在人体高度的
`62%`，并把相机抬高 `6°` 形成轻微俯视。交互旋转后按 `F` 可恢复该视角。

拟合实现位于 `src/rgbd_avatar/avatar/smpl_sequence.py`，主要约束
左右髋、膝、踝、肩、肘、腕、颈以及由 Halpe 足部关键点合成的左右
脚。鼻、眼、耳映射到 SMPL 加载器提供的同名表面顶点，用低权重
消除人体正反二义性；Halpe 的头顶点不映射到位于头部内部的 SMPL
head rig joint。手部使用 `wrist→middle_mcp`、
`pinky_mcp→index_mcp` 与低权重 `wrist→thumb_mcp` 约束 SMPL
wrist/hand 旋转，从真实大拇指位置区分手心和手背，同时不让弯曲指尖
改变整只手的朝向。体型使用固定 betas（默认为 neutral，也可加载 preset），
用全序列可靠骨长的中位比例确定一个共享尺度，然后逐帧优化 69 维
body pose、3 维 global orient 和根平移，并以上一帧作为时序初值。

Halpe26 只观测肩、髋和颈部，没有对应 SMPL `spine1/2/3` 的直接
关键点。若只使用关节点拟合与弱全身姿态先验，三节隐式脊柱可能在
肩髋误差仍很小时形成视觉上明显的侧弯或 S 形。当前为三节脊柱增加
独立的姿态正则，默认权重为 `0.05`。整体前倾、侧倾和朝向仍由
global orientation、髋关节以及肩髋目标表达，正则只抑制内部脊柱的
无观测自由度。当前序列侧向脊柱偏离的中位值由 `2.04 cm` 降至
`0.88 cm`，P95 由 `3.47 cm` 降至 `2.10 cm`；三节脊柱最大旋转的
中位值由 `24.1°` 降至 `3.4°`，平均目标误差只增加约 `0.53 mm`。
可用 `--spine-pose-weight` 调整强度。

人物在当前图像中手掌宽度较小，因此手部支路将 Halpe constrained
手腕作为强锚点：先检查手部模型腕点是否落在合理邻域，通过后平移
全部二维关键点，再用相机内参和腕部前景深度恢复相对三维结构，最后
平移全部三维点，使 Hand21 第 `0` 点与 Halpe 手腕严格重合。不存在
可靠 constrained 手腕时不生成该手，避免用肘/肩或墙面位置伪造
连接。拟合前还会拒绝手长小于 `3.5 cm`、大于 `25 cm`、有效指尖
不足或关键点塌缩的观测。某只手检测失败时仅跳过该手约束，身体拟合
不会中断。

手部深度采用“腕部参考面 + Hand21 拓扑门控”，而不是把 21 个像素的
局部深度彼此独立地直接使用。先从原始深度确定腕部参考面，低于
`0.20` 置信度或偏离腕部超过 `0.12 m` 的样本回退到该参考面；然后
按腕—掌指关节—指节顺序检查相邻关节的 Z 增量，超过对应指骨可实现
范围时回退到父关节深度。最后只进行一次到 constrained 手腕的整体
平移。这避免了回退点被重复施加腕部 Z 修正，也能滤除落在衣物、背景
或深度边缘上的孤立高置信度点，同时保留手掌朝向相机时连续且符合
拓扑的真实深度梯度。调试时可用 `--disable-topology-depth-gate` 查看
未做拓扑门控的对照结果。

当前第 4 组数据拟合结果：

- 共 `35` 帧，其中第 `0–32` 帧生成 `6890` 顶点的 SMPL 网格；
  第 `33` 帧人物部分出画、第 `34` 帧无人，均不生成网格。
- Hand21 强腕点门控后在有效人物帧中保留左手 `17` 帧、右手 `27`
  帧，分别恢复 `83` 和 `133` 个有效指尖；末尾两帧均不生成手部
  结果。保留结果的二维和三维腕点最大连接误差分别为 `0 px` 和
  `5.4e-11 m`。
- 修正前当前序列相邻手部关节的最大 Z 跳变为 `0.149 m`，腕部参考面
  修正和拓扑门控后为 `0.089 m`；后者出现在允许较大透视深度变化的
  腕—掌指关节连接中，所有短指节连接均通过对应的解剖深度上限。
- 每帧基础身体/面部目标为 `20` 个，通过质量门控后总目标数为
  `20–28`；序列共享尺度为 `1.06525`。
- 33 个有效帧新增面部和手部目标后的平均误差均值约 `0.027 m`，
  逐帧平均误差的 95 分位约 `0.035 m`。
- 缓存约 `2.5 MB`，保存顶点、24 个 SMPL 关节、姿态、根变换、
  三角面、逐帧误差和有效帧掩码。
- 缓存记录 `poses.jsonl`、`hands.jsonl`、模型和地面文件哈希以及
  完整显示坐标变换；帧号、骨架层、源内容、手部结果或地面对齐
  不匹配时，查看器拒绝加载旧缓存。
- `--smpl --validate-only` 和 `--stickman --validate-only` 均已通过
  全部 35 帧无窗口验收；项目测试为 `98 passed`。

这里完成的是“骨架驱动中性人体网格”的第一版 baseline。由于当前
没有优化人体形状参数，也未做足底接触、Foot Lock 或碰撞约束，体型
细节和脚底稳定性仍属于下一阶段，而不是本阶段拟合误差的 bug。

### 阶段 5：足部约束

任务：

- [ ] 脚部接触检测
- [ ] 根节点高度修正
- [ ] 防穿地
- [ ] Foot Lock
- [ ] 两骨骼 IK

验收：

- 静止站立时脚底贴地。
- 行走时滑步明显减少。
- 下蹲时不出现大幅穿地。

### 阶段 6：性能优化

任务：

- [ ] 异步流水线
- [ ] TensorRT 或 ONNX Runtime CUDA
- [ ] 减少 CPU/GPU 数据复制
- [ ] 点云 GPU 化
- [ ] GPU 蒙皮
- [ ] 性能分析和瓶颈统计

验收：

- 骨架提取和仿生人驱动达到 30 FPS。
- 连续运行 30 分钟无明显内存增长。
- 端到端延迟控制在可接受范围内。

## 21. 首轮 Codex 开发任务

建议先让 Codex 完成以下最小闭环：

```text
相机取帧
→ RGB 显示
→ 深度图显示
→ 点云显示
→ 读取相机内参
→ 加载二维姿态模型
→ 输出二维关键点
→ 根据深度恢复三维关键点
→ Open3D 显示人体点云和三维骨架
```

### 21.1 给 Codex 的第一条任务描述

```text
请根据本项目文档创建一个 Python 项目骨架。

第一阶段只实现：
1. 抽象 RGB-D 相机接口 BaseRGBDCamera；
2. 定义 RGBDFrame 和 CameraIntrinsics 数据结构；
3. 提供一个 MockRGBDCamera，可读取本地 RGB、深度、强度和点云文件；
4. 编写 inspect_camera_streams.py，同时显示 RGB、深度、强度和 Open3D 点云；
5. 所有距离统一为米；
6. 增加清晰的日志、类型标注、异常处理和配置文件；
7. 为深度反投影函数编写单元测试；
8. 暂时不要接入具体相机 SDK，把厂商接口保留为适配器。
```

### 21.2 第二条任务描述

```text
在现有项目中加入二维人体姿态接口 PoseEstimator。

要求：
1. 后端可插拔；
2. 先实现 RTMPoseBackend；
3. 保留 OpenPoseBackend 占位实现；
4. 输出统一 Pose2D 数据结构；
5. 支持单人模式；
6. 将关键点绘制在 RGB 图像上；
7. 统计平均 FPS 和 P95 推理耗时；
8. 不要在推理循环中频繁创建模型或分配大数组。
```

### 21.3 第三条任务描述

```text
实现二维关键点到三维关键点的恢复模块。

要求：
1. 使用对齐后的深度图；
2. 关键点附近使用局部窗口深度中位数；
3. 过滤零值、NaN、Inf 和超范围深度；
4. 输出每个三维关节的置信度和 valid 标记；
5. 使用相机内参进行反投影；
6. 所有坐标单位为米；
7. Open3D 中同时显示人体点云和骨架连线；
8. 编写深度采样与反投影单元测试。
```

## 22. 测试动作清单

第一批实验动作：

- [ ] 正面静止站立
- [ ] 左右抬手
- [ ] 双手平举
- [ ] 弯腰
- [ ] 下蹲
- [ ] 单腿抬起
- [ ] 原地转身
- [ ] 向前移动 2 米
- [ ] 向后移动 2 米
- [ ] 左右横移
- [ ] 行走并摆臂
- [ ] 部分遮挡

## 23. 评价指标

### 骨架

- 2D 关键点置信度
- 3D 关节抖动标准差
- 骨长变化率
- 关键点丢失率
- 遮挡恢复时间

### 身高

\[
E_H=|H_{estimate}-H_{gt}|
\]

### 根节点轨迹

\[
RMSE_{root}
=
\sqrt{
\frac{1}{T}
\sum_{t=1}^{T}
\|\hat{\mathbf{p}}_t-\mathbf{p}_t^{gt}\|_2^2
}
\]

### 仿生人

- 动作方向正确率
- 关节翻转次数
- 脚底穿地距离
- 接触阶段脚部滑动距离
- 真人和仿生人的位移比例误差

### 性能

- 平均 FPS
- P95 帧耗时
- 推理耗时
- 点云处理耗时
- 动作重定向耗时
- 渲染耗时
- 端到端延迟
- GPU 显存占用
- CPU 内存占用

## 24. 风险与注意事项

### 24.1 RGB 和深度对齐状态

当前数据已由相机软件完成去畸变和像素级对齐，可以直接使用同一
像素坐标关联 RGB 与深度。实现仍应在数据加载阶段检查两者分辨率、
时间戳和文件前缀是否一致，防止文件配对错误。若以后切换到相机的
原始未配准流，则必须重新使用 SDK 对齐接口，不能沿用当前单位外参。

### 24.2 点云单位不确定

必须通过已知距离实测确认点云单位。所有内部坐标统一为米。

### 24.3 深度边缘错误

手腕、脚踝、头顶和人体轮廓处容易混入背景深度。必须使用邻域统计和人体掩膜。

### 24.4 身高受姿态影响

弯腰、下蹲和抬腿时不能更新身高。身高只在直立稳定阶段估计。

### 24.5 单相机遮挡

人体侧身、转身或四肢互相遮挡时，二维关键点和深度都会退化。需要时序预测和骨长约束。

### 24.6 人体离开视野

需要定义跟踪丢失状态：

- 短时丢失：预测并缓慢衰减。
- 长时丢失：冻结仿生人或回到默认姿态。
- 再次进入：重新初始化根节点基准。

### 24.7 相机移动

当前默认相机固定。如果相机移动，必须增加 RGB-D SLAM 或视觉里程计，否则无法区分人体移动与相机移动。

## 25. 后续 3DGS 接入接口

当前系统输出统一世界状态：

```python
@dataclass
class AvatarState:
    timestamp: float
    scale: float
    root_translation_m: np.ndarray
    root_rotation: np.ndarray
    local_joint_rotations: np.ndarray
    joint_positions_m: np.ndarray
    left_foot_contact: bool
    right_foot_contact: bool
```

后续与 3DGS 场景通过一个相似变换连接：

\[
\mathbf{p}^{G}
=
s_G\mathbf{R}_G\mathbf{p}^{W}+\mathbf{t}_G
\]

其中：

- \(W\)：当前 RGB-D 世界坐标系。
- \(G\)：3DGS 场景坐标系。
- \(s_G\)：场景尺度。
- \(\mathbf{R}_G\)：坐标轴旋转。
- \(\mathbf{t}_G\)：原点平移。

### 25.1 独立虚拟场景的放置标定

当前实验场景位于：

```text
/home/fr1511b/program/workspace/data/3DGS/
├── point_cloud.ply             # 1,320,761 个完整 Gaussian
└── sparse/0/
    ├── cameras.bin             # 1 个 PINHOLE 内参模型
    ├── images.bin              # 261 个注册相机位姿
    ├── points3D.bin
    └── points3D.ply            # 31,612 个点，用作轻量选点代理
```

该场景和 RGB-D 人体采集不是同一个物理空间，因此不能从 COLMAP
`sparse/0` 自动解出 \(W\rightarrow G\) 外参。这里采用“独立虚拟场景
放置”语义，显式选择：

1. 一段已知真实长度的两个端点，用于计算 `scale_g_per_m`。
2. 至少五个大范围分散的地面点，用于拟合 3DGS 地面并检测误点。
3. 一个明确位于地面上方的点，用于确定平面法向量正方向。
4. 人物出生点和面朝方向点。
5. SMPL 第一帧左右脚中心在当前米制世界中的地面投影。

实现位于 `src/rgbd_avatar/scene/alignment.py`。保存的
`scene_alignment.json` 是带校验的版本化数据，包含尺度、proper
rotation、平移、地面、出生点、坐标轴以及原始选点。SMPL 缓存中的
`vertices_display_m` 已包含人体身高缩放；接入场景时只再应用一次
`scale_g_per_m`，不能重复乘 SMPL cache 的 `scale`。

### 25.2 交互生成 scene_alignment.json

先确认实验场景中一段物体的真实长度，例如门宽、地砖边长或标定尺。
稀疏 COLMAP 点云不能可靠辨认物体，因此正式选点不再使用
`sparse/0/points3D.ply`。先在 `gsplat` CUDA 环境中从完整 Gaussian PLY
渲染真实 RGB 和 expected-depth。已验证相机 `000116.jpg`，对应
`camera-index=115`，黄黑立柱的顶部和底部均未被移动底盘遮挡，也有
足够大的可见地面区域：

```bash
cd /home/fr1511b/program/workspace/humanpose

PYTHONPATH=src /home/fr1511b/miniconda3/envs/gsplat/bin/python \
  scripts/render_3dgs_alignment_view.py \
  --scene-root ../data/3DGS \
  --camera-index 115 \
  --width 1916 \
  --output ../data/3DGS/alignment_views/000116_full.npz \
  --overwrite
```

输出：

```text
../data/3DGS/alignment_views/000116_full.png  # 供人查看的真实高斯画面
../data/3DGS/alignment_views/000116_full.npz  # RGB、expected-depth、alpha、K、c2w
```

若该视角不适合标定，可先列出 261 个 COLMAP 视角，再替换
`--camera-index`：

```bash
PYTHONPATH=src python scripts/render_3dgs_alignment_view.py \
  --scene-root ../data/3DGS --list-cameras
```

选定视角后，在 RGB-D 环境中点击高斯渲染画面。以下 `1.20` 只是命令
格式示例，必须换成所选两个端点之间的实际米制距离：

```bash
conda activate rgbd-avatar

PYTHONPATH=src python scripts/pick_3dgs_alignment_view.py \
  --view ../data/3DGS/alignment_views/000116_full.npz \
  --scene-root ../data/3DGS \
  --known-distance-m 0.75 \
  --known-length-direction vertical \
  --smpl-cache outputs/sequences/4_pointcloud_exit_gate/smpl_sequence.npz
```

程序在同一张清晰的高斯 RGB 画面上依次采集：

```text
1/4 选择已知长度的两个端点
2/4 只在地面选择至少五个大范围分散的点
3/4 选择一个地面上方参考点
4/4 先选择人物出生点，再选择面朝方向点
```

画面中使用左键选择、右键撤销、`R` 清空、`Enter` 确认当前阶段、
`Esc` 取消。每个像素通过局部 alpha 门控的 expected-depth 和相机
`K/c2w` 反投影为精确 `point_g`。生成文件为：

```text
/home/fr1511b/program/workspace/data/3DGS/scene_alignment.json
/home/fr1511b/program/workspace/data/3DGS/scene_alignment_picks.png
```

第二个文件把标尺、地面、上方参考、出生点和方向点以不同颜色画回原始
高斯图，用于审计选点是否选到了前景物体而非背景。
校验失败时也会保存最新一次
`scene_alignment_attempt_picks.png`，错误消息会逐项列出 `G1...Gn`
相对拟合地面的误差，便于只排查误差最大的地面点。

选择“高度”作为已知长度时必须添加
`--known-length-direction vertical`。写文件前会检查：所有地面点相对
拟合平面的有效点误差不超过 5 cm、已知高度与地面法线夹角不超过
20 度。非交互 JSON spec 中的出生点和方向点距离地面不能超过 15 cm；
画面选点模式则直接使用相机射线与地面的交点。校验失败时不会覆盖已有
的 `scene_alignment.json`。

考虑到 3DGS expected-depth 在反光地面上可能出现局部异常，至少五个
地面点中允许自动排除一个不满足 5 cm 阈值的点，但仍要求至少 80%（且
不少于四个）点形成一致地面。出生点和面朝点不再采用噪声较大的表面
depth，而是将对应相机射线直接与该地面求交。

原来的 `scripts/pick_scene_alignment.py` 仅保留为稀疏点云诊断工具，
不再推荐用于场景物体标定。

标定完成后，可把 SMPL 三角网格变换到 `G`，并使用 Gaussian view 的
expected-depth 完成前后遮挡合成：

```bash
PYTHONPATH=src python scripts/render_avatar_in_3dgs.py \
  --view ../data/3DGS/alignment_views/000116_full.npz \
  --scene-root ../data/3DGS \
  --smpl-cache outputs/sequences/4_pointcloud_exit_gate/smpl_sequence.npz \
  --overwrite
```

输出位于 `../data/3DGS/avatar_composites/`。这不是点云预览：背景来自
真实 Gaussian rasterization，人物使用三角网格 Z-buffer，人物和场景
之间依据同一相机坐标系下的 expected-depth 正确处理遮挡。

如果目标是三维坐标接入而不是二维合成，应直接导出已经变换到 `G`
坐标系的静态 SMPL 网格：

```bash
PYTHONPATH=src python scripts/export_static_avatar_to_3dgs.py \
  --scene-root ../data/3DGS \
  --smpl-cache outputs/sequences/4_pointcloud_exit_gate/smpl_sequence.npz \
  --frame-index 0 \
  --overwrite
```

生成 `avatar_static/avatar_frame_000000_3dgs.ply`。该文件中的每个顶点
已经应用
`p_G = scale_g_per_m * rotation_g_from_w * p_W + translation_g_from_w`，
因此可与 `point_cloud.ply` 作为两个几何对象直接载入同一三维世界，
不能再对导出的网格重复应用 `scene_alignment.json`。同目录 JSON 保存
完整变换、三维边界、地面误差和身高，`*_joints.npy` 保存同坐标系关节。

要交互查看真实 Gaussian 外观和静态假人，可启动浏览器查看器：

```bash
PYTHONPATH=src /home/fr1511b/miniconda3/envs/gsplat/bin/python \
  scripts/view_static_avatar_gaussian_scene.py \
  --scene-root ../data/3DGS \
  --initial-view ../data/3DGS/alignment_views/000101_candidate.npz \
  --port 8080
```

然后打开 `http://127.0.0.1:8080`。场景使用原始 1,320,761 个
Gaussians；静态三角网格只在查看器内临时进行表面采样并转换为橙色
Gaussians，以便和场景参与同一次光栅化与深度排序。磁盘上的静态 PLY
仍然是原始三角网格，坐标不会被再次转换。

若要保留原来 `smpl_sequence.npz` 的人体运动，应使用动态查看器，而非
静态 PLY。场景 Gaussians 只加载一次；每个有效人体帧都使用同一个
`scene_alignment.json` 执行 `V_G(t)=s R V_W(t)+t`：

```bash
PYTHONPATH=src /home/fr1511b/miniconda3/envs/gsplat/bin/python \
  scripts/view_dynamic_avatar_gaussian_scene.py \
  --scene-root ../data/3DGS \
  --smpl-cache outputs/sequences/4_pointcloud_exit_gate/smpl_sequence.npz \
  --initial-view ../data/3DGS/alignment_views/000101_candidate.npz \
  --fps 10 \
  --port 8080
```

浏览器的 `Avatar Animation` 面板提供 `Play`、帧滑块、前后帧按钮和
FPS 控制。人体表面采样使用固定三角形及重心坐标，因而相邻帧保持时间
对应，不会每帧随机闪烁。

动态查看器也支持直接显示 SMPL-24 火柴人。该模式使用关节球、23 条
骨骼杆和放大的头部球，不再渲染皮肤表面：

```bash
PYTHONPATH=src /home/fr1511b/miniconda3/envs/gsplat/bin/python \
  scripts/view_dynamic_avatar_gaussian_scene.py \
  --scene-root ../data/3DGS \
  --smpl-cache outputs/sequences/4_pointcloud_exit_gate/smpl_sequence.npz \
  --initial-view ../data/3DGS/alignment_views/000101_candidate.npz \
  --avatar-style stick \
  --fps 10 \
  --port 8080
```

可用 `--stick-bone-radius-m`、`--stick-joint-radius-m` 和
`--stick-head-radius-m` 调整火柴人的线宽、关节和头部大小。

如果不能使用画面点击，可以复制
`../data/3DGS/scene_alignment_spec.example.json`，填写由 CloudCompare 或
其他查看器读取的 3DGS 坐标，然后运行：

```bash
PYTHONPATH=src python scripts/create_scene_alignment.py \
  --scene-root ../data/3DGS \
  --spec ../data/3DGS/scene_alignment_spec.json \
  --smpl-cache outputs/sequences/4_pointcloud_exit_gate/smpl_sequence.npz
```

`avatar_anchor_w_m` 可以保留为 `null`；此时默认从 SMPL 第一帧左右脚
关节中心推导，并强制落在当前世界 `Z=0`。输出已存在时两个命令都会
拒绝覆盖，确认重新标定时显式传入 `--overwrite`。

### 25.3 对齐预览与验收

生成配置后，先用轻量场景点云验证尺寸、贴地、朝向和移动比例：

```bash
PYTHONPATH=src python scripts/view_avatar_in_3dgs.py \
  --scene-root ../data/3DGS \
  --alignment ../data/3DGS/scene_alignment.json \
  --smpl-cache outputs/sequences/4_pointcloud_exit_gate/smpl_sequence.npz
```

预览器用完整 SMPL 三角网格播放动画，并显示以出生点为中心的米制地面
网格和 `W` 坐标轴在 `G` 中的方向。它只用于检查几何对齐，静态场景
显示的是 COLMAP/Gaussian 中心，不是最终的 Gaussian rasterization。

对齐验收至少检查：

- 人物脚底位于选定地面，首帧不悬空或穿地。
- 人物初始面朝所选 forward point。
- 已知身高在场景中比例合理。
- `W` 中前进 `1 m` 时，`G` 中位移长度为 `scale_g_per_m`。
- `rotation_g_from_w` 的行列式为 `+1`，没有镜像翻转。

后续最终渲染复用同一个 `SceneAlignment`，将 SMPL 网格和查看相机转换
到 `G` 后，使用相同相机内外参分别渲染 3DGS RGB/expected-depth 与
SMPL RGB/depth，再做遮挡合成。物理碰撞仍需要额外地面或场景 mesh；
Gaussian PLY 本身不作为可靠碰撞体。

## 26. 最小可行产品定义

MVP 完成标准：

1. 单台 RGB-D 相机稳定输出四种数据。
2. RGB 图中实时获得人体二维骨架。
3. 深度图恢复米制三维骨架。
4. Open3D 中骨架与人体点云基本重合。
5. 估计地面和真人身高。
6. 仿生人按真人身高统一缩放。
7. 仿生人同步上肢、下肢和躯干动作。
8. 真人移动时仿生人同步米制根节点位移。
9. 脚底不明显穿地。
10. 整体达到 30 FPS 或明确定位性能瓶颈。

## 27. 当前仍需补充的信息

在接入真实设备前，需要填写：

- RGB-D 相机型号
- 厂商 SDK
- 操作系统
- CUDA 版本
- GPU 型号
- 相机实时采集的标称帧率
- 深度有效范围
- 强度图具体含义
- 仿生人模型格式

当前已经确认：

- RGB 与深度分辨率均为 `816 × 612`。
- 当前图像已经去畸变。
- RGB 与深度已经像素级对齐。
- 当前有效内参为
  `(390.697235107, 390.601348877, 408.284454346, 321.971252441)`。
- 当前有效畸变系数为全零。
- 当前对齐后 2D/3D 外参为单位旋转和零平移。
- 深度图和导出点云使用毫米，内部统一转换为米。
- 导出 PCD 为无组织点云，内部从深度重建有组织点云。
- 相机坐标系为右手系，`+X` 向右、`+Y` 向下、`+Z` 向前。
- 欧拉角顺序为 `ZYX`，对应 `Yaw-Pitch-Roll`。

仍需补充的信息只影响具体适配代码，不改变总体架构。

## 28. 代码模块化重构（2026-08-04）

此前 `scripts/` 同时承担命令行解析、文件读写、算法编排和可视化，
单文件最高接近 1900 行，不利于单元测试和复用。当前已调整为
“薄命令入口 + 领域包实现”，所有原有运行命令保持不变。

当前职责划分如下：

```text
scripts/                         # 仅保留兼容 CLI，每个入口约 8 行
src/rgbd_avatar/
├── io/                          # JSON/YAML/JSONL、原子写入、相机配置
├── data/                        # RGB-D 序列发现和结果记录契约
├── pipeline/                    # RGB-D、手部、地面、SMPL 流水线编排
│   ├── rgbd_sequence.py
│   ├── hand_sequence.py
│   ├── ground_sequence.py
│   ├── smpl_sequence.py
│   └── metrics.py               # 无状态统计函数
├── diagnostics/                 # 序列诊断和骨长异常可视化
├── visualization/
│   ├── sequence3d.py            # 纯数组/坐标变换
│   ├── hand3d.py                # Hand21 显示几何
│   ├── contracts.py             # 相机清单和地面标定校验
│   └── viewer_app.py            # Open3D 交互应用编排
└── avatar/                       # 程序化、火柴人、SMPL 与模型工具
```

关键约束：

- `scripts/` 不再被测试或其他模块反向导入。
- JSON/JSONL 结果通过临时文件、`fsync` 和 `os.replace` 原子提交，
  避免中断后留下半个结果文件。
- `poses.jsonl` 与 `hands.jsonl` 统一执行 schema、帧序、时间戳、
  segment 及前缀对齐校验。
- 相机配置、manifest 快照和地面标定由独立契约模块校验。
- 算法层不依赖具体 CLI 文件，便于后续接入实时采集或其他查看器。

例如原命令仍然有效：

```bash
PYTHONPATH=src python scripts/process_rgbd_sequence.py --help
PYTHONPATH=src python scripts/view_pose3d_sequence.py --help
PYTHONPATH=src python scripts/fit_smpl_sequence.py --help
```

回归验证应包含：完整 `pytest`、全部脚本的 `--help` 冒烟测试，以及
现有序列的 `--validate-only` 查看器检查。

## 29. 独立 SMPL Neutral 体型编辑器（2026-08-06）

新增 `scripts/edit_smpl_shape.py`，用于在零姿态下独立调整
`SMPL_NEUTRAL_CLEAN.pkl`。该工具不读取 RGB、深度、Halpe26、手部结果
或 `smpl_sequence.npz`，因此不会改动当前姿态估计流水线。

启动命令：

```bash
cd /home/fr1511b/program/workspace/humanpose
conda activate rgbd-avatar

PYTHONPATH=src python scripts/edit_smpl_shape.py
```

界面功能：

- 鼠标旋转、平移和缩放零姿态人体。
- `β0` 至 `β9` 滑条实时改变 SMPL PCA 体型参数。
- “整体尺度”只做统一米制缩放。
- “恢复 Neutral”回到全零 betas 和 `scale=1.0`。
- “保存参数与网格”同时保存可复用 JSON 和静态零姿态 PLY。
- 界面显示当前人体的高、宽、厚包围盒尺寸。

默认输出：

```text
assets/models/smpl/presets/custom_shape.json
assets/models/smpl/presets/custom_shape.ply
```

JSON 保存模型路径、模型 SHA-256、十维 betas、整体 scale 和网格坐标系；
加载时校验 SHA-256，避免把参数误用于不同的 SMPL 模型。PLY 是已经转换为
项目显示坐标系（X 右、Y 前、Z 上）、以米为单位并贴地的静态预览网格。

继续编辑已经保存的体型：

```bash
PYTHONPATH=src python scripts/edit_smpl_shape.py \
  --load-preset assets/models/smpl/presets/custom_shape.json
```

指定其他保存位置：

```bash
PYTHONPATH=src python scripts/edit_smpl_shape.py \
  --output-preset outputs/avatar_shapes/person_a.json \
  --output-mesh outputs/avatar_shapes/person_a.ply
```

无 GUI 验证模型和参数：

```bash
PYTHONPATH=src python scripts/edit_smpl_shape.py --validate-only --device cpu
```

无 GUI 导出 Neutral 或已加载的预设：

```bash
PYTHONPATH=src python scripts/edit_smpl_shape.py \
  --load-preset assets/models/smpl/presets/custom_shape.json \
  --save-and-exit
```

注意：SMPL betas 是统计形状主成分，不是具有固定语义的“腰围”或“肩宽”
参数。不同 beta 可能同时改变多个身体部位；整段动画应固定使用同一组
betas，不能逐帧改变，否则会产生体型闪烁。

### 29.1 将保存体型用于已有动作序列

`fit_smpl_sequence.py` 支持通过 `--shape-preset` 为整段动作固定同一组
betas。该过程只重新执行 SMPL 姿态拟合，不会重新执行 RTMPose、深度恢复、
时序滤波或骨长约束。

使用预设中保存的体型和整体尺度：

```bash
PYTHONPATH=src python scripts/fit_smpl_sequence.py \
  --results-dir outputs/sequences/4_pointcloud_exit_gate \
  --shape-preset assets/models/smpl/presets/custom_shape.json \
  --use-preset-scale \
  --output outputs/sequences/4_pointcloud_exit_gate/smpl_sequence_custom_shape.npz
```

查看调整后体型的原动作：

```bash
PYTHONPATH=src python scripts/view_pose3d_sequence.py \
  --results-dir outputs/sequences/4_pointcloud_exit_gate \
  --smpl \
  --smpl-cache \
    outputs/sequences/4_pointcloud_exit_gate/smpl_sequence_custom_shape.npz
```

如果省略 `--use-preset-scale`，程序只固定预设 betas，整体尺度仍根据旧序列
的三维骨长重新估计；如果显式提供 `--scale`，命令行数值优先级最高。

当前 `custom_shape.json` 已完成 35 帧拟合，其中前 33 帧生成网格，人物部分
离场和完全离场的最后两帧继续保持无网格。输出保存在独立的
`smpl_sequence_custom_shape.npz`，没有覆盖原始 `smpl_sequence.npz`。

### 29.2 SMPL 动画表面平滑

三维序列查看器会先对每帧 SMPL 网格执行默认 `2` 次 Taubin 几何平滑，
再重新计算顶点法线。该处理只改变 Open3D 中的显示网格，不修改
`smpl_sequence*.npz`、体型预设、拟合姿态或骨架关节坐标；Taubin 方法
相较普通拉普拉斯平滑也更不容易让人体整体收缩。

查看当前自定义体型和平滑后的原动作：

```bash
PYTHONPATH=src python scripts/view_pose3d_sequence.py \
  --results-dir outputs/sequences/4_pointcloud_exit_gate \
  --smpl \
  --smpl-cache \
    outputs/sequences/4_pointcloud_exit_gate/smpl_sequence_custom_shape.npz \
  --smpl-smooth-iterations 2
```

`0` 表示关闭几何平滑，以便和原始 SMPL 网格比较；如果轮廓仍显粗糙，
可尝试 `4`。不建议继续大幅增加次数，因为它会逐渐弱化脸部、手脚等局部
形状并增加逐帧更新开销。若粗糙感来自体型本身而不是三角面，应回到体型
编辑器降低达到 `-3` 或 `+3` 边界的 beta，而不是继续增加显示平滑次数。

### 29.3 手脚刚性末端约束

SMPL 拟合不再把观测到的指尖、脚趾坐标作为绝对位置逐点拉动模型。
手腕和脚踝仍由 Halpe26 决定米制位置；其余末端点只用于构造与长度无关的
单位方向：

- 手掌使用 Hand21 的拇指、中指和小指方向，共同约束掌面朝向。
- 脚掌使用 Halpe26 的脚跟到大脚趾、小脚趾方向，约束脚的前向和旋转。
- 检测到的手指长度、掌宽和脚长不会传入 SMPL，因此深度抖动不能拉伸
  假人的手脚形态。
- SMPL 的左右手掌和左右脚掌末端局部关节默认硬锁为零旋转；实际方向由
  上一级手腕和脚踝旋转承担，避免末端折叠。

方向损失默认权重为 `0.02`，可通过
`--end-effector-direction-weight` 调整。仅为对照旧行为时可以传入
`--allow-end-effector-articulation` 解除末端关节硬锁，但指尖、脚趾仍只
作为方向约束，不恢复绝对位置拉伸。

已使用当前体型预设重新拟合 4 号序列并保留旧缓存：

```text
outputs/sequences/4_pointcloud_exit_gate/
  smpl_sequence_custom_shape_rigid_endpoints.npz
```

查看结果：

```bash
PYTHONPATH=src python scripts/view_pose3d_sequence.py \
  --results-dir outputs/sequences/4_pointcloud_exit_gate \
  --smpl \
  --smpl-cache \
    outputs/sequences/4_pointcloud_exit_gate/smpl_sequence_custom_shape_rigid_endpoints.npz
```

该缓存 35 帧中拟合 33 帧，最后两帧继续遵守离场门控；33 帧的四个末端
局部关节旋转数值均严格为零，平均身体位置拟合误差约为 `0.033 m`。

### 29.4 Halpe26 → SMPL 语义重定向

直接将 Halpe26 的同名点拟合到 SMPL 内部关节并不严格成立。例如 Halpe
左右髋更接近图像中的人体表面位置，而 SMPL 左右髋是骨盆内部旋转中心；
当前数据中两者髋宽比例中位数达到 `1.64`。因此默认拟合模式已由直接
关节匹配改为 `semantic` 语义重定向：

1. Halpe `hip(19)` 只负责 SMPL `pelvis(0)` 的米制位置。
2. 髋轴、肩轴和躯干方向共同建立正交人体坐标系；SMPL 髋宽、肩宽和
   躯干比例始终取固定体型自身的静止骨架。
3. 大腿、小腿、上臂和前臂只传递单位方向；目标端点按固定 SMPL 骨长
   重新构造，不把 RGB-D 骨长抖动传给假人。
4. 每段骨骼先从整段序列估计中位长度和 MAD，并使用最多 `±25%` 的保守
   区间做三级门控：统计骨长异常但仍在解剖范围内时继续使用单位方向并
   降权；只有缺失、低置信度或物理长度不可能时才拒绝身体骨段。脚部仍
   使用严格门控，避免地面和图像边界深度把脚掌方向带偏。
5. 手掌方向使用 `wrist→middle_mcp`、`pinky_mcp→index_mcp` 和低权重
   `wrist→thumb_mcp`，不再使用会随握拳动作改变的指尖射线。
6. 体型 preset 只固定 betas；整体尺度默认从当前序列鲁棒估计。只有完成
   独立尺度标定后才应显式使用 `--use-preset-scale`。

实现位于 `src/rgbd_avatar/retargeting/halpe_smpl.py`。长度先验、逐帧拒绝
骨段和累计拒绝次数都会写入 SMPL cache 的 metadata，便于复现实验和定位
坏帧。旧的逐点方式仍可通过 `--retarget-mode legacy` 做对照，但不再推荐。

当前推荐拟合命令：

```bash
PYTHONPATH=src python scripts/fit_smpl_sequence.py \
  --results-dir outputs/sequences/4_pointcloud_exit_gate \
  --shape-preset assets/models/smpl/presets/custom_shape.json \
  --retarget-mode semantic \
  --device cpu \
  --output \
    outputs/sequences/4_pointcloud_exit_gate/smpl_sequence_custom_shape_semantic_retarget.npz
```

注意这里刻意不使用 `--use-preset-scale`。本序列与当前体型组合得到的鲁棒
尺度为 `1.04301`，而 preset 中的 `1.0` 偏小约 `4.3%`。

查看结果：

```bash
PYTHONPATH=src python scripts/view_pose3d_sequence.py \
  --results-dir outputs/sequences/4_pointcloud_exit_gate \
  --smpl \
  --smpl-cache \
    outputs/sequences/4_pointcloud_exit_gate/smpl_sequence_custom_shape_semantic_retarget.npz
```

35 帧中前 33 帧生成网格，最后两帧继续由离场门控抑制。与旧的刚性末端
逐点缓存相比，SMPL `spine1/2/3` 旋转 P95 从约
`5.9°/4.9°/7.8°` 降到约 `1.0°/0.7°/0.7°`；左腕从 `105.8°`
降到 `59.6°`，右脚踝从 `89.3°` 降到 `52.4°`。语义目标本身使用固定
SMPL 比例构造，平均拟合残差约 `0.001 m`，该数值不能与旧模式对原始
Halpe 点的残差直接等同解释。

首次语义门控曾把统计骨长异常直接当作整段无效，导致第 27、28 帧左上臂
目标被删除，腕部与可见 Halpe 骨架最大偏差达到 `0.736/0.751 m`。改为
三级门控后，这两帧不再丢失手臂目标；全序列最大可见手臂偏差降为
`0.153 m`。缓存 metadata 分别记录 `retarget_rejected_*` 和
`retarget_soft_*`，必须区分真正拒绝与保留方向但降权的观测。

## 29. 实时 RGB-D → 动态火柴人 → 3DGS 容器架构

实时版本不应在每帧运行当前 160 次迭代的 SMPL 拟合。火柴人直接消费
经过深度恢复、One Euro 滤波和骨长约束的 Halpe26 三维关节，流水线为：

```text
RGB-D camera adapter
  └─ RGB BGR8 + aligned depth float32(m) + runtime intrinsics + timestamp
       └─ RTMDet/RTMPose (2D Halpe26)
            └─ aligned-depth lifting (camera C, metres)
                 └─ temporal + bone constraints
                      └─ live placement (C → local L → 3DGS G)
                           └─ LivePosePacket (26 joints only)
                                └─ static gsplat scene + dynamic stick Gaussians
                                     └─ Viser browser
```

`src/rgbd_avatar/live/` 定义了硬件无关边界：

- `RGBDSource`：相机插件必须实现 `start/read/close`。
- `RGBDFrame`：要求深度已对齐到 RGB、单位已转为米，并携带运行时内参。
- `LiveSceneMapper`：把实时相机关节放到已有出生点和 3DGS 坐标轴。
- `LivePosePacket`：姿态服务发给渲染服务的版本化 Halpe26 小消息。

### 29.1 两种实时放置模式

`root_locked` 以每帧脚部中心为局部原点，始终把人放在保存的
`spawn_point_g`。它不依赖相机在现实房间中的绝对安装位置，换相机后只需
正确内参和深度对齐，最适合第一版和原地动作驱动。

`fixed_origin` 使用一次标定得到的固定相机原点与地面旋转，保留真人在
地面上的米制移动。换设备或移动相机后必须重新求
`rotation_l_from_c + origin_camera_m`，但不需要重新生成 3DGS，也不改变
`scene_alignment.json` 的场景尺度、轴和出生位置。

两者都使用：

```text
p_G = spawn_G + scale_g_per_m * R_G_from_L * p_L
```

而不是把新相机的绝对坐标直接代入旧离线序列的平移量。

### 29.2 为什么采用两个 GPU 容器

当前已验证环境存在不可直接合并的二进制依赖：

```text
live-pose       PyTorch 2.1.0 + CUDA 11.8 + MMCV 2.1.0 + MMPose 1.3.2
gsplat-viewer   PyTorch 2.7.1 + CUDA 12.8 + gsplat 1.5.3 + Viser 1.0.30
```

推荐 Compose 拆成：

```yaml
services:
  live-pose:
    image: rgbd-avatar/live-pose:cu118
    environment:
      CAMERA_BACKEND: realsense   # 或 orbbec/k4a/replay
      PLACEMENT_MODE: root_locked
      POSE_PUBLISH_ENDPOINT: tcp://0.0.0.0:5556
    devices:
      - /dev/bus/usb:/dev/bus/usb
    volumes:
      - ./assets/models:/app/assets/models:ro
      - ./deploy:/config:ro
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  gsplat-viewer:
    image: rgbd-avatar/gsplat-viewer:cu128
    environment:
      POSE_SUBSCRIBE_ENDPOINT: tcp://live-pose:5556
    ports:
      - "8080:8080"
    volumes:
      - ../data/3DGS:/data/3DGS:ro
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

相机 SDK 和宿主机 udev 规则无法由 Docker 完全抽象。设备更换策略应是
“相同 `RGBDSource` 接口，不同 adapter/image profile”，而不是让核心
算法识别任意厂商私有驱动。模型、3DGS 和标定均作为只读 volume 挂载，
不复制进镜像。

宿主机需要 NVIDIA 驱动和 NVIDIA Container Toolkit；Compose 的 GPU
reservation 必须包含 `capabilities: [gpu]`。USB 相机优先映射具体设备，
开发阶段才可映射整个 `/dev/bus/usb`，避免使用不必要的 `privileged`。

### 29.3 实时调度和性能边界

- 相机线程只保留最新一帧，队列长度固定为 1；推理慢时丢旧帧而不积压。
- RTMPose 模型只初始化一次，并在启动后执行 CUDA warm-up。
- 以设备时间戳驱动 One Euro filter，不能使用消息到达时间代替采集时间。
- 当前 CPU 基线约为：2D 推理 286 ms、深度恢复 66 ms、总计 358 ms。
  实时部署必须启用 CUDA，并首先以 `640×480/15 FPS` 或等价规格验收。
- 3DGS 服务只加载一次 132 万个场景 Gaussians；每条姿态消息只更新数百
  个火柴人 Gaussian means，不重新读取 PLY。
- 姿态消息很小，建议 ZeroMQ/WebSocket 只传关节、置信度、usable mask、
  frame number 和 capture timestamp，不在两个 GPU 容器间传 SMPL 网格。

### 29.4 换设备时的固定流程

1. 安装该相机的宿主机 udev 规则，并选择对应 adapter/profile。
2. 探测设备 serial、RGB/深度模式、运行时内参、深度单位和对齐能力。
3. 验证 `depth_m.shape == rgb.shape[:2]`，深度单位为米。
4. `root_locked` 可直接运行；`fixed_origin` 重新做地面/原点标定。
5. 运行 30 秒健康检查：采集 FPS、丢帧率、有效深度比例、有效关节数、
   端到端延迟和 GPU 显存。
6. 3DGS 场景、`scene_alignment.json` 和渲染容器保持不变。

下一阶段需要先确认相机品牌/型号以及目标主机是 x86 NVIDIA GPU 还是
Jetson；这两项决定厂商 SDK、基础镜像架构、USB 映射和可用 CUDA 版本。

## 30. 目录实时流与独立假人（2026-08-05）

新安装继续使用原相机、原分辨率、原内参、原深度单位和原文件命名，
只有相机在应用坐标系中的安装外参发生变化。相机软件持续写入：

```text
/home/fr1511b/下载/LxCameraViewer-ubuntu20-viewer/LxCameraViewer/data/FF6690772788
```

实时输入仍严格使用同一时间戳前缀的：

```text
YYYYMMDD_HHMMSSmmm_r.png
YYYYMMDD_HHMMSSmmm_d.pgm
YYYYMMDD_HHMMSSmmm_a.pgm
YYYYMMDD_HHMMSSmmm_t.pcd
```

实时姿态主链路只等待 RGB 和深度完整配对；强度图和相机导出 PCD 不
阻塞假人更新。RGB 和深度之间已经完成的像素对齐外参继续保持单位旋转
和零平移，不得用新的安装外参替换它。

新的应用安装外参为：

```text
Roll  =   91.51 degree
Pitch = -179.71 degree
Yaw   =   89.95 degree
x =   0.00 mm
y =   0.00 mm
z = 909.05 mm
```

继续采用右手坐标、列向量和 `ZYX / Yaw-Pitch-Roll`：

```text
p_application = R_application_from_camera @ p_camera + t_application
R_application_from_camera = Rz(yaw) @ Ry(pitch) @ Rx(roll)
t_application = [0.0, 0.0, 0.90905] m
```

数值旋转矩阵约为：

```text
[-0.00087265,  0.02634700,  0.99965248]
[-0.99998681, -0.00508267, -0.00073899]
[ 0.00506143, -0.99963994,  0.02635108]
```

这组参数单独保存在 `configs/live.yaml`，不覆盖 `configs/camera.yaml`
中的成像内参和对齐约定。

### 30.1 当前实现

实时实现包括：

- `src/rgbd_avatar/live/directory_source.py`：轮询目录，只接收精确配对且
  文件尺寸和修改时间已经稳定的 RGB-D 帧。
- `src/rgbd_avatar/live/extrinsics.py`：校验并执行新的 ZYX 应用外参。
- `src/rgbd_avatar/live/processor.py`：复用 RTMDet/RTMPose、三维恢复、
  出画门控、One Euro、个人骨长标定和骨长约束。
- `src/rgbd_avatar/visualization/live_mannequin.py`：同时维护原始 RGB、
  RGB 二维骨架和独立 Open3D 三维火柴人，不依赖离线 `poses.jsonl`、
  SMPL cache 或 3DGS。
- `scripts/view_live_mannequin.py`：薄命令入口。

目录源默认使用 `start_at: latest`：启动时处理最新一组已经存在的完整
帧，之后只处理更新的时间戳。如果模型推理期间已经到达多组完整帧，
直接选择最新一组并统计被替代的旧帧，不建立无限队列，因此延迟不会
随运行时间持续增长。可选模式为：

- `latest`：显示最新已有帧，然后追踪新帧，是目录模式默认值。
- `new`：忽略启动前已有文件，只等待启动后的新帧。
- `oldest`：从目录最早帧开始顺序处理，主要用于调试，不适合实时显示。

启动目录输入的实时三视图：

```bash
cd /home/fr1511b/program/workspace/humanpose
conda activate rgbd-avatar

PYTHONPATH=src python scripts/view_live_mannequin.py --source directory
```

程序默认自动选择 `cuda:0`（CUDA 可用时），模型只初始化一次，同一份
最新帧结果同时更新：

1. `Live RGB`：未经绘制的相机 RGB。
2. `Live RGB + Halpe26 Skeleton`：检测框、二维关节和二维骨架叠加图，
   底部同时显示帧状态、可用三维关节数和各阶段耗时。
3. `Live RGB-D 3D Stickman`：使用 constrained Halpe26 和新安装外参
   生成的米制三维火柴人。

默认三维输出已经改为经典黑色火柴人，不再显示程序化体积假人。RGB
两视图只做绘制，不会重复运行检测或姿态模型。Open3D 窗口中按 `M`
仍可在三维火柴人、来源三维骨架和两者叠加之间循环；在任意活动窗口
按 `Q` 退出。RGB 窗口默认缩放为原始 `816×612` 的 `0.75`，可在
`configs/live.yaml` 的 `viewer.rgb_view_scale` 中调整。

如果只为历史外观对照而临时恢复程序化假人，可显式使用：

```bash
PYTHONPATH=src python scripts/view_live_mannequin.py \
  --avatar-model procedural
```

无需桌面的完整数据链路验收：

```bash
PYTHONPATH=src python scripts/view_live_mannequin.py \
  --source directory \
  --device cpu \
  --validate-only \
  --max-frames 1 \
  --start-at latest
```

2026-08-05 对目标目录最新帧 `20260805_165550996` 的 CPU 验收结果为：

- 人体检测及三维恢复状态为 `ok`。
- constrained 输出 `25` 个可用 Halpe26 关节；同一结果可以直接构造
  三维火柴人。
- 新外参变换后关节高度范围约为 `0.048～1.585 m`，有效足部关键点
  高度中位数约为 `0.066 m`，与 `Z=0` 应用地面基本一致。
- CPU 单帧端到端约 `335 ms`；这是正确性验证，不是 CUDA 性能验收。
- 当时实时相关测试和完整项目回归为 `139 passed`；SDK 直连加入后更新为
  `141 passed`。

当前目录中的文件时间戳间隔仍约为 `0.5 s`，所以即使算法更快，目录
驱动的可见动作更新率也只有约 `2 FPS`。若需要真正的 15/30 FPS，必须
提高相机软件的落盘频率或改为相机 SDK 内存帧适配器。

2026-08-06 起，以优化后的 `pointcloud_cluster` 骨架为实时主输出。本地
三视图与无窗口 WebSocket 发布必须使用同一种恢复算法，避免前端骨架和
本机显示不一致。当前默认命令为：

```bash
PYTHONPATH=src python scripts/view_live_mannequin.py \
  --recovery-method pointcloud_cluster
```

如需临时优先降低延迟，仍可显式使用 `--recovery-method window_median`，
但它不再是默认和主展示结果。

本阶段直接用 constrained Halpe26 驱动三维火柴人，不执行程序化体积
假人或逐帧 SMPL 迭代拟合。这样能够先验收采集、坐标、动作方向和实时
调度；后续需要真实皮肤网格时，应增加实时 IK/动作重定向器，而不是在
实时循环中复用当前离线 160 次迭代的 SMPL 拟合。

## 31. MRDVS SDK 直连实时火柴人（2026-08-05）

为去掉 Viewer 落盘、目录轮询、文件稳定等待以及 PNG/PGM 解码，实时
输入默认改为兰鑫/MRDVS SDK 内存帧。目录输入仍保留，可用
`--source directory` 随时回退。

本机已确认的设备和运行环境为：

```text
device ID: FF6690772788
model:     camera_S10_192.168.1.125_3956
camera IP: 192.168.1.125:3956
host NIC:  192.168.1.188/24
GPU:       NVIDIA GeForce RTX 4070, 12 GB
```

实现文件为：

- `src/rgbd_avatar/live/lx_camera_source.py`：加载厂商 Python wheel 和
  `libLxCameraApi.so`，先枚举再按 ID 打开设备。
- `scripts/inspect_lx_camera.py`：只测 RGB-D 采集，不加载 RTMPose，便于
  将相机帧率问题和姿态推理问题分开定位。
- `scripts/view_live_mannequin.py`：通过同一 `RGBDSource` 接口直接复用
  原始 RGB、RGB 骨架和三维火柴人三个视图。

直连适配器在启动阶段执行：

1. 开启 RGB 和深度流，关闭不需要的强度流。
2. 设置 `DEPTH_TO_RGB`，使深度尺寸和像素坐标都以 RGB 为准。
3. 开启 `LX_BOOL_ENABLE_SYNC_FRAME`，只交付配对的 RGB-D 帧。
4. 对齐之后重新读取运行时分辨率和 RGB 内参，不能盲用静态尺寸。
5. SDK 的 RGB888 转为 OpenCV/RTMPose 使用的 BGR。
6. 深度按 `0.001` 从毫米转为米。
7. 立即复制 SDK 管理的 NumPy 内存视图，防止下一次 `getFrame()` 覆盖
   仍在推理的图像。
8. 使用微秒级 sensor timestamp 驱动时序滤波。

相机只能被一个进程独占。运行以下命令前必须完全退出
`LxCameraViewer`；正常退出或 `Ctrl+C` 时适配器会在 `finally` 中停止流
并关闭设备。

先执行无模型采集探测：

```bash
cd /home/fr1511b/program/workspace/humanpose
conda activate rgbd-avatar

PYTHONPATH=src python scripts/inspect_lx_camera.py --frames 30
```

2026-08-05 的真机 30 帧探测结果为：

- 对齐后的 RGB 和深度都是 `816×612`。
- 运行时内参为 `fx=390.6972351, fy=390.6013489,
  cx=408.2844543, cy=321.9712524`，与现有配置一致。
- SDK 深度/RGB 报告帧率均为 `15 FPS`，无瞬态帧错误。
- 主机收帧中位间隔 `66.77 ms`，sensor timestamp 中位间隔
  `66.63 ms`。
- 有效深度像素中位比例约 `82.5%`。

探测通过后启动三个实时视图：

```bash
PYTHONPATH=src python scripts/view_live_mannequin.py \
  --source sdk \
  --device cuda:0 \
  --recovery-method pointcloud_cluster
```

`configs/live.yaml` 已把 `source.type` 设为 `sdk`，因此也可以省略
`--source sdk`。显式写出 `--device cuda:0` 的好处是 CUDA 不可用时立即
报错，而不会无声退回 CPU。启动日志必须出现：

```text
RTMPose initialized on cuda:0
```

无 GUI 的完整链路验收命令为：

```bash
PYTHONPATH=src python scripts/view_live_mannequin.py \
  --source sdk \
  --device cuda:0 \
  --recovery-method pointcloud_cluster \
  --validate-only \
  --max-frames 10
```

以下 RTX 4070 性能数字是当时 `window_median` 的历史低延迟基线：第一帧
因 CUDA/CuDNN kernel 预热约 `380 ms`；随后
9 帧的检测、Halpe26、米制三维恢复、时序/骨长约束和火柴人构造总耗时
为 `23.9～35.8 ms/帧`。稳定阶段处理速度低于相机的 `66.7 ms` 帧周期，
因此可以跟满当前硬件的 `15 FPS`，不再受目录落盘 `2 FPS` 限制。

如果出现 `LX_E_DEVICE_NOT_FOUND`，依次检查相机供电、网线、主机
`192.168.1.x` 网卡以及 Viewer 是否完全退出；如果 Viewer 刚被强制结束，
等待相机心跳超时后再试。若出现 `LX_E_CTRL_PERMISS_ERROR`，通常表示仍有
另一个进程占用相机。

## 32. FastAPI Topic Hub 火柴人发布（2026-08-06）

根据 `实时WebSocket接口说明.md`，另一台电脑已经提供通用实时 Hub：

```text
ws://<FastAPI电脑IP>:8000/api/realtime/ws
```

本机不再开放 WebSocket 服务端，而是作为 Python 客户端主动连接 FastAPI。
这条链路只发布最终 `joints_application_m`，不发送 RGB、深度、二维骨架
或逐帧网格：

```text
MRDVS SDK → RTMPose/Halpe26 → depth lifting → temporal/bone
          → application-space 26 joints → FastAPI topic Hub → browser
```

实现位于 `src/rgbd_avatar/live/stickman_websocket.py`。发布器运行在独立
网络线程中，待发送槽长度固定为 1；FastAPI 不可用时继续采集和推理，
并以 `0.5～10 s` 指数退避重连，不让网络阻塞实时姿态管线。

固定协议为：

```text
client_id = rgbd-avatar-FF6690772788
event     = avatar.stickman.updated
topic     = avatar:stickman:FF6690772788
```

配置保存在 `configs/live.yaml` 的 `live.websocket_publish`。2026-08-06
确认 FastAPI 电脑地址为 `192.168.30.132:8000`，因此当前已启用：

```text
enabled = true
url = ws://192.168.30.132:8000/api/realtime/ws
```

有两种启动方式。

方式一：使用当前已经写入 YAML 的 FastAPI 地址，无本地窗口运行：

```bash
cd /home/fr1511b/program/workspace/humanpose
conda activate rgbd-avatar

PYTHONPATH=src python scripts/view_live_mannequin.py \
  --source sdk \
  --device cuda:0 \
  --recovery-method pointcloud_cluster \
  --headless
```

方式二：不修改 YAML，直接用命令行提供目标 URL；`--publish-url` 会自动
启用发布：

```bash
PYTHONPATH=src python scripts/view_live_mannequin.py \
  --source sdk \
  --device cuda:0 \
  --recovery-method pointcloud_cluster \
  --headless \
  --publish-url ws://<FastAPI电脑IP>:8000/api/realtime/ws
```

如果还需要本机三个调试窗口，只需删除 `--headless`。成功连接时日志应
出现：

```text
Stickman WebSocket connected: ...
topic=avatar:stickman:FF6690772788
```

前端连接同一个 Hub，并订阅：

```text
avatar:stickman:FF6690772788
```

FastAPI 转发的消息外层是 `type=event`，前端从
`message.payload.joints` 读取固定 26 项的 Halpe26 米制关节。不可用
关节为 JSON `null`，应用坐标为右手系、`+Z` 向上、地面 `Z=0`。

发布协议、latest-only 覆盖、真实 WebSocket hello/publish/ack 和管线结果
回调均有自动化测试。受限沙箱中的完整回归为 `145 passed, 1 skipped`；
被跳过的回环 socket 集成测试在宿主网络权限下单独运行结果为
`4 passed`。

2026-08-06 对实际服务 `192.168.30.132:8000` 完成联调：WebSocket
握手、`hello`、`ping/pong` 均成功；另发送一帧 `status=probe` 的合成
Halpe26 数据，发布器统计为 `sent=1, ack=1, last_error=None`，说明
FastAPI 已接受正式 `publish` 信封。

同日相机曾短暂出现 3D 流 `LX_E_NOT_RECEIVE_STREAM`；随后重新探测确认
RGB-D 已恢复为 `15.01 FPS`、深度有效率中位数 `84.1%`，30 帧无瞬时
错误。实际端到端测试继续处理了 5 帧，每帧均有 26 个可用关节，向远端
发送 5 帧，`last_error=None`。CUDA
预热后的代表性耗时为推理 `26 ms`、深度恢复 `1.5 ms`、总计约
`29 ms`。

首次短测试在所有帧和统计日志完成后触发了一次原生库退出段错误。原因
是相机 SDK 的 `start/read/close` 跨线程执行，导致 SDK 与 CUDA 在解释器
退出时清理顺序不稳定。`LatestPoseWorker` 现将相机完整生命周期固定在同
一个工作线程，`LxCameraRGBDSource.close()` 也会确定性释放 ctypes/CDLL
包装；相同 5 帧端到端测试随后以退出码 0 正常结束。

发布器关闭时还会用最多 `0.5 s` 排空在途 ACK。最终真实相机复测统计为
`submitted=5, sent=5, ack=5, last_error=None`，进程退出码为 0。

同日根据优化后的骨架展示结果，将实时主恢复方法从 `window_median` 切换
为 `pointcloud_cluster`。本地 GUI 和 `--headless` 发布现在读取同一个
`live.recovery_method`，只有显式命令行参数才会覆盖它。真实相机无窗口
复测日志确认 `Depth recovery method: pointcloud_cluster`；3 帧均为
`status=ok, usable=26`，稳定帧代表性耗时为推理 `24.3 ms`、点云恢复
`56.0 ms`、总计 `81.6 ms`，WebSocket 统计为
`submitted=3, sent=3, ack=3, last_error=None`。

## 33. 本地多人 RGB-D 火柴人实验（2026-08-06）

多人实验最初通过独立本地入口实现，不修改现有单人
`avatar.stickman.updated` 事件、topic 或 `payload.joints`。2026-08-10 的
正式可选发布集成见第 36 节；现有 `scripts/view_live_mannequin.py` 和单人
协议仍保持不变。

RTMPose 后端本来就会返回当前帧所有通过质量门控的 `Pose2D`；原实时路径
只取 `poses[0]`。新增实现位于：

- `src/rgbd_avatar/live/multi_person_processor.py`：每帧共享一次有组织点云，
  使用检测框 IoU、二维中心、Halpe26 关键点距离和三维躯干位置进行一对一
  匈牙利匹配，为每个本地 `track_id` 独立维护出画门控、One Euro、骨长
  标定和骨长约束。短时漏检保留预测，超过 `max_missing_s` 后删除轨迹，
  之后重新进入会分配新 ID。
- `src/rgbd_avatar/visualization/live_multi_person.py`：用不同颜色在 RGB、
  RGB 骨架和 Open3D 三维窗口显示每个本地 ID。
- `scripts/view_live_multi_person.py`：初版只启动本地处理和查看器；目前已
  增加第 36 节所述的可选 WebSocket 参数。

启动 SDK 本地多人视图：

```bash
cd /home/fr1511b/program/workspace/humanpose
conda activate rgbd-avatar

PYTHONPATH=src python scripts/view_live_multi_person.py \
  --source sdk \
  --device cuda:0 \
  --detector auto \
  --recovery-method hybrid \
  --max-persons 2 \
  --max-missing-s 0.35
```

不打开 GUI 的本地链路验证：

```bash
PYTHONPATH=src python scripts/view_live_multi_person.py \
  --source sdk \
  --device cuda:0 \
  --recovery-method hybrid \
  --validate-only \
  --max-frames 10
```

多人入口默认使用本地专用的 `hybrid` 深度恢复：26 个关节先用快速
`window_median` 恢复，再只对鼻、眼、耳、头、颈、肩、肘和腕这 13 个
高风险关节使用 `pointcloud_cluster`，并将点云结果以 fail-closed 方式覆盖
快速结果。这样保留头部分离和手臂异常延伸的主要防护，同时避免为每个人
重复聚类全部 26 个关节。它只影响 `view_live_multi_person.py`，不会改变
单人实时入口、WebSocket payload 或前端协议。

仍可显式传入 `--recovery-method pointcloud_cluster` 做最高稳健度对照，或
传入 `--recovery-method window_median` 做最低延迟对照。日志每帧输出
`inference`、`recovery` 和 `total`：若 `total` 高于相机 15 FPS 对应的
`66.7 ms`，显示帧率必然低于相机帧率；多人时应重点比较 `hybrid` 与完整
点云的 `recovery`。

本地入口的 `--max-persons` 默认也改为 2。真实 SDK/CUDA 两人复测中，模型
仍检测到 3 个候选框，但只对最高分 2 人执行 3D 恢复；预热后的代表性耗时
为推理 `22～24 ms`、深度恢复 `34～35 ms`、总计 `59～60 ms`，低于相机
约 16 FPS 对应的 `62.5 ms` 帧周期。相同画面允许 3～4 人时，总耗时约
`84～103 ms`。首帧约 `0.4～0.5 s` 是 CUDA/cuDNN 一次性预热，不代表
稳定延迟。若确实要测试 3 人或 4 人，可显式传入 `--max-persons 3/4`，但
当前 Python 点云聚类实现下会降低刷新率。

RGB 叠加窗口中的 `ID n` 和颜色只用于本机诊断。第一版没有外观 ReID 或
实例分割：两人分开、短时遮挡和检测结果顺序变化时可以保持几何 ID；两人
长时间贴身、同深度交叉时仍可能发生关键点串人或 ID 交换。发生明显重叠时
不应把这版轨迹用于最终个人身高或长期骨长身份，下一阶段需要逐人 mask、
轨迹预测深度门控和可选外观 ReID。

自动化测试覆盖两人检测顺序交换、短时漏检预测、轨迹超时后分配新 ID、
多人共享一次有组织点云以及彩色本地渲染；完整项目回归为
`159 passed, 1 skipped`。另用保存目录完成了一帧 `--validate-only` 真模型
冒烟，程序正常初始化、处理和退出。该目录最新帧没有检出人物，因此真实
双人检测质量仍需用 SDK 现场站入两人后检查。

## 34. Halpe26 直接驱动 Mixamo FBX（2026-08-07）

已将 `assets/models/mixamo/Ch09_nonPBR.fbx` 接入为第四种假人后端。该路径
不经过 SMPL 顶点或 SMPL 姿态优化：Halpe26 三维骨架直接生成 Mixamo
局部/全局骨骼旋转，Hand21 只在掌面质量门控通过时补充整只手的朝向。

### 34.1 FBX 资产导入

实现位于 `src/rgbd_avatar/avatar/mixamo_asset.py`。项目直接解析 Mixamo
导出的二进制 FBX 7700 子集，不要求安装 Blender、Assimp 或 FBX SDK。
导入器会严格验证：

- 单一 Mesh 与 Skin 关系、骨架拓扑无环且只有一个根骨骼；
- 65 个 Cluster 与 LimbNode 一一连接；
- Cluster `Transform/TransformLink` 与绑定/逆绑定矩阵互逆；
- 所有顶点都有有效权重，只保留并重新归一化权重最大的 4 个影响骨骼；
- 四边形按原多边形顺序三角化，`ByPolygonVertex/IndexToDirect` UV 与三角形
  角点严格同步；
- DiffuseColor 连接的内嵌 PNG 纹理存在且格式正确。

当前人物包含 `15716` 个蒙皮控制点、`31292` 个三角形、`65` 根骨骼以及
一张 `4096×4096` 漫反射纹理。绑定姿态执行一次 LBS 后与原始控制点的
最大数值差为 `3.33e-16 m`，说明矩阵约定、厘米到米换算和权重方向一致。

### 34.2 固定比例解析 IK

实现位于 `src/rgbd_avatar/retargeting/halpe_mixamo.py`：

1. `Hips` 使用 Halpe `hip(19)` 的米制位置；髋轴、肩轴和
   `hip→neck` 建立躯干坐标系。
2. 左右臂和腿使用肩—肘—腕、髋—膝—踝方向解析对齐；模型骨骼平移始终
   来自 FBX 绑定姿态，因此检测骨长不会拉伸网格。
3. `Spine/Spine1/Spine2` 保持作者绑定的直躯干局部关系，由 Hips 的完整
   躯干朝向带动；这避免 Halpe 缺少三节脊柱时产生无观测 S 形自由度。
4. Foot 使用 heel→双 toe 中心，Hand 使用 `wrist→middle_mcp` 与
   `pinky_mcp→index_mcp` 建立完整朝向。手指本身保持 FBX 原始局部姿态。
5. 身体骨段复用全序列 median/MAD 和解剖范围三级门控；无效段保持上一帧
   相对父骨骼的局部旋转。有效旋转使用最大角速度和响应系数限制，避免单帧
   180° 翻转。
6. 该角色是风格化大头短腿比例，因此统一尺度使用头顶到双足的鲁棒身高，
   而不是强行匹配每段人体骨长。当前序列尺度为 `1.09679`。
7. 有地面标定时默认对每帧执行刚性竖直平移，使网格最低点精确接触
   `z=0`；可用 `--no-ground-contact` 做不接地对照。

逐帧蒙皮和自包含缓存位于
`src/rgbd_avatar/avatar/mixamo_sequence.py`。缓存包含逐帧顶点、骨骼矩阵、
拓扑、三角 UV、内嵌 PNG 和全部输入哈希，查看器会拒绝 pose、Hand21、
pose layer 或地面对齐方式不一致的旧缓存。

### 34.3 生成与查看

生成缓存：

```bash
cd /home/fr1511b/program/workspace/humanpose
conda activate rgbd-avatar

PYTHONPATH=src python scripts/fit_mixamo_sequence.py \
  --results-dir outputs/sequences/4_pointcloud_exit_gate \
  --model assets/models/mixamo/Ch09_nonPBR.fbx \
  --output outputs/sequences/4_pointcloud_exit_gate/mixamo_sequence.npz \
  --overwrite
```

查看带纹理人物：

```bash
PYTHONPATH=src python scripts/view_pose3d_sequence.py \
  --results-dir outputs/sequences/4_pointcloud_exit_gate \
  --mixamo \
  --mixamo-cache outputs/sequences/4_pointcloud_exit_gate/mixamo_sequence.npz \
  --no-show-hands \
  --start-paused
```

去掉 `--no-show-hands` 可以叠加左右 Hand21 做诊断；按 `M` 在来源骨架、
Mixamo 网格和二者叠加之间切换，按 `F` 恢复胸部高度的轻微俯视相机。

当前 `mixamo_sequence.npz` 约 `16 MB`。35 帧中前 33 帧有效，最后两帧
继续由部分出画/无人门控抑制；每帧 `15716` 个顶点均有限，最低点恒为
`z=0`，网格高度 P5–P95 为 `1.753–1.796 m`。Open3D 已验证能同时装载
`31292` 个三角形、`93876` 个 UV 角点、4K 纹理和动态顶点法线。完整项目
回归为 `171 passed, 1 skipped`。

## 35. 多人身份遮挡 shadow test（2026-08-10）

本节记录正式集成前的隔离评估；测试入口仍可用于 A/B 对照，正式集成结果
见第 36 节。为验证运动、外观、RGB-D 深度和遮挡歧义门控，新增身份对照
入口：

- `src/rgbd_avatar/tracking/shadow_identity.py`：零新增依赖的测试跟踪器，
  使用匀速预测、HSV 上半身外观、三维躯干位置、重叠时冻结外观、歧义时
  拒绝强制匹配，以及遮挡期间约 1 秒的轨迹保留。
- `scripts/test_multi_person_identity.py`：先运行现有本地多人处理器，再把同一
  帧观测旁路送入 shadow tracker。现有 `track_id` 只作为 `current` 对照，
  shadow tracker 独立产生 `shadow_id`，其结果不会回写姿态、WebSocket、
  topic 或前端 payload。

现场测试时让两个人先分开站立，再相向交叉、短时完全遮挡并重新分开：

```bash
cd /home/fr1511b/program/workspace/humanpose
conda activate rgbd-avatar

PYTHONPATH=src python scripts/test_multi_person_identity.py \
  --source sdk \
  --device cuda:0 \
  --detector auto \
  --recovery-method hybrid \
  --max-persons 2 \
  --save-video outputs/multi_identity_shadow.mp4
```

窗口标签格式为 `current:n shadow:m`。`frozen` 表示两框重叠时已冻结外观
模板；红色 `shadow:?` 表示关联过于歧义，测试跟踪器选择保留预测而不是
强行换 ID。日志中的 `predicted`、`ambiguous`、`removed` 分别表示预测
轨迹、拒绝匹配的当前观测和已超时轨迹；`remaps` 表示同一个 `shadow_id`
对应的现有 `current` ID 发生变化，是需要回看片段的候选事件。按 `Q` 或
`Esc` 退出。

无窗口冒烟命令：

```bash
PYTHONPATH=src python scripts/test_multi_person_identity.py \
  --source sdk \
  --device cuda:0 \
  --recovery-method hybrid \
  --max-persons 2 \
  --validate-only \
  --max-frames 100
```

合成测试覆盖两人交叉且检测顺序变化、完全重叠时拒绝强制分配、遮挡轨迹
保留/过期以及不同衣服颜色的 HSV 描述子。真实 SDK/CUDA 10 帧冒烟中，
现有两个 ID 与两个 shadow ID 均保持一致，shadow 关联耗时稳定为
`0.37～0.43 ms/帧`，没有构造或调用 WebSocket 发布器。这一版本只验证
身份关联机制，不是完整 BoT-SORT 或深度学习 ReID；真实交叉录制通过后，
再决定是否在独立依赖环境中评估正式 BoT-SORT-ReID。

新增 4 项 shadow 身份测试全部通过；与原单人 WebSocket 测试组合运行结果为
`8 passed, 1 skipped`。排除既有 `tests/test_viser_mixamo_viewer.py` 后，项目
其余回归为 `187 passed, 1 skipped`。该既有 viewer 测试单独运行也会在
`viser_mixamo_viewer.py::_resize` 的原生调用中段错误，与本次未被生产路径
导入的 shadow 模块无关。

## 36. 多人身份与 WebSocket 正式集成（2026-08-10）

上一节的 shadow tracker 现已作为第二方案接入
`scripts/view_live_multi_person.py`，旁路测试入口继续保留。正式多人入口有
两种身份后端：

- `geometry`：原有 IoU、二维关键点和三维根节点匈牙利匹配，仍是默认和
  保底方案。
- `shadow`：加入运动预测、HSV 躯干外观、RGB-D 根位置、重叠冻结外观、
  歧义拒配和遮挡延长保留。若该模块在运行中抛出异常，处理器会记录异常，
  并在当前进程后续帧自动使用 `geometry`。

只在本机使用原方法：

```bash
PYTHONPATH=src python scripts/view_live_multi_person.py \
  --source sdk --device cuda:0 --detector auto \
  --identity-tracker geometry --max-persons 2
```

使用新方法，同时向现有 FastAPI Hub 发送多人结果：

```bash
PYTHONPATH=src python scripts/view_live_multi_person.py \
  --source sdk --device cuda:0 --detector auto \
  --identity-tracker shadow --max-persons 2 \
  --publish-stickmen
```

服务器部署时不打开本机 GUI：

```bash
PYTHONPATH=src python scripts/view_live_multi_person.py \
  --source sdk --device cuda:0 --detector auto \
  --identity-tracker shadow --max-persons 2 \
  --publish-stickmen --headless
```

`configs/live.yaml` 中 `live.multi_person.identity_tracker` 可设置常用默认值，
`live.multi_person.websocket_publish.enabled` 控制多人发布。命令行显式选项
优先；`--no-publish-stickmen` 可以临时关闭发送，`--publish-url ws://...`
会覆盖地址并启用发送。

兼容性上，原单人事件 `avatar.stickman.updated`、原 topic 和
`payload.joints[26]` 完全不变。多人使用新增事件
`avatar.stickmen.updated`，仍可走同一个 topic；payload 增加
`stream_id`、`identity_method`、`identity_fallback` 和 `persons[]`。每个
person 包含 `track_id`、`status`、`observed_in_frame` 和 26 个 joints。
前端交付包新增 `StickmenWebSocketClient` 与 `MultiAvatarController`，可用
`setDisplayLimit(1)`、`setDisplayLimit(2)` 或 `setDisplayLimit("all")`
控制显示数量，而不要求后端为不同人数建立不同接口。

## 37. 二维关键点引导的局部深度分簇（2026-08-11）

为降低 4 人以上时 `hybrid` 对每个人执行点云关节恢复的开销，多人入口新增
实验模式 `guided_window`。这里不是只在 RGB 图上对 26 个二维坐标聚类，
而是把每个二维关键点作为深度图中的局部 ROI，在小深度窗内做一维深度
分簇：

1. 第一帧或新建轨迹仍使用原 `window_median` 最近有效簇，建立初始三维
   关节深度。
2. 之后先用原快速路径完成检测与 `track_id` 关联，再用该轨迹上一帧经过
   时序/骨长处理后的每个关节深度作为期望值。
3. 局部窗口同时出现前后两个人的深度层时，选择最接近期望深度的受支持
   深度簇，不再固定选择离相机最近的一层。
4. 如果窗口只有遮挡者表面，且与轨迹期望深度相差超过 `0.45 m`，该关节
   本帧直接判为无效，由现有短时预测承接；不会把错误前景深度发送成新的
   实测关节。

该模式只读取二维关节点周围的小窗口，不生成有组织点云；前端仍收到同一个
`avatar.stickmen.updated`、相同的 `persons[]` 和每人 26 个关节，无需修改
协议或前端代码。默认 `hybrid` 和保底 `window_median` 均保持不变。

本地观察并对比恢复耗时：

```bash
cd /home/fr1511b/program/workspace/humanpose
conda activate rgbd-avatar

PYTHONPATH=src python scripts/view_live_multi_person.py \
  --source sdk --device cuda:0 --detector auto \
  --identity-tracker shadow \
  --recovery-method guided_window \
  --max-persons 4 \
  --view-mode all
```

需要同时发送给前端时只需追加 `--publish-stickmen`；切回当前默认方案则把
参数改为 `--recovery-method hybrid`。建议现场测试先让每个人分开出现一小段
时间，使各 `track_id` 建立自己的深度历史，再进行交叉和遮挡。若一开始就
完全重叠，系统没有可用历史，`guided_window` 无法凭单目二维坐标判断哪一层
属于哪个人。

合成回归覆盖了三种关键行为：旧调用仍选择最近簇；有 2 m 轨迹历史时在
1 m/2 m 混合窗口中选择 2 m；完全只剩 1 m 遮挡面时拒绝错误观测并保持
2 m 的短时预测。该方法主要减少深度恢复开销和跨人的深度串扰，身份交换
仍由 `shadow` 跟踪器处理，二维模型把两个人合并成一个检测框时也无法仅靠
本方法恢复出两套骨架。

主测试使用 `--view-mode all`，会同时打开原始 RGB、二维 Halpe26 骨架叠加和
三维骨架三个窗口。`--view-mode 2d` 则只打开独立的
`Local Multi-Person 2D Detection` 窗口，
显示每个人的检测框、Halpe26 连接线、关节点、`track_id`、检测数量以及各阶段
耗时，不会初始化 Open3D。`--view-mode 3d` 只显示三维。`--headless` 和
`--validate-only` 按定义不会打开
任何本地可视化窗口，因此查看二维结果时不要同时传入这两个参数。

## 38. 实时相机应用外参更新（2026-08-11）

相机安装位姿更新后，`configs/live.yaml` 的
`live.application_extrinsics` 调整为：

```yaml
transform: "application_from_camera"
euler_order: "ZYX"
roll_deg: 90.38
pitch_deg: -179.83
yaw_deg: 89.95
translation_m: [0.0, 0.0, 0.78795]
```

定义仍为列向量约定
`p_application = Rz(yaw) @ Ry(pitch) @ Rx(roll) @ p_camera + t`。标定输入的
`x=0, y=0, z=787.95` 单位为毫米，写入实时配置前转换为
`[0, 0, 0.78795] m`。本次只更新相机到应用坐标系的软件外参，不修改 SDK
设备内参、RGB/深度工厂外参或 `DEPTH_TO_RGB` 对齐方式。

## 39. 多人深度图空间连通域恢复（2026-08-11）

针对 `guided_window` 在单人轮廓关节仍可能选中背景、并将首次错误写入历史
的问题，多人入口新增实验模式 `depth_connected`。默认正式模式仍为
`hybrid`，`guided_window` 仅保留为历史性能对照。

`depth_connected` 不生成 `HxWx3` 有组织点云。其处理顺序为：

1. 先用二维 bbox、关键点以及 `shadow` 的外观描述完成身份关联；该模式不再
   为关联额外执行一次 26 点快速深度恢复。
2. 把同一人物 26 个二维关节点的扩展局部窗口合并成一个 bbox 内联合 ROI，
   只保留有效深度范围和历史躯干深度带内的像素。
3. 对联合 ROI 只执行一次八邻域、自适应深度边缘连通标记。各关节随后在自己
   的基础/扩展半径内查询共享组件，并计算局部中心距离、支持像素数、深度
   MAD、簇占比和躯干深度分数，不再各自重复 BFS 聚类。
4. 同一 `track_id` 的历史关节深度只作为软评分项，不再像
   `guided_window` 一样决定性地锁定某个深度层。当前空间表面证据更强时可以
   从错误历史重新捕获。
5. 选择结果还需通过躯干—四肢父子最大长度、脸部联合候选、耳—眼距离、
   脚部紧凑性以及左右肩髋异常检查；不可信关节 fail-closed，交给现有时序
   预测承接。
6. 某关节连续 3 帧没有可靠观测后，暂停使用其旧历史深度；重新获得有效
   空间表面后恢复历史引导。
7. 只对最终选中的关节深度执行反投影，WebSocket、`persons[]` 和每人
   Halpe26 数据结构完全不变。

本地使用方式：

```bash
cd /home/fr1511b/program/workspace/humanpose
conda activate rgbd-avatar

PYTHONPATH=src python scripts/view_live_multi_person.py \
  --source sdk --device cuda:0 --detector auto \
  --identity-tracker shadow \
  --recovery-method depth_connected \
  --max-persons 4
```

不传 `--view-mode` 时仍打开原始 RGB、二维 Halpe26 和三维骨架三个窗口。
需要向前端发布时追加 `--publish-stickmen`。切回当前正式方案使用
`--recovery-method hybrid`。

自动化测试覆盖局部中心人体表面与大面积背景分离、错误历史为背景时仍恢复
当前人体表面、不可能的前臂长度被拒绝、多人处理不构造点云、第二帧开始使用
轨迹历史、快速窗口恢复调用数为 0、每人只执行一次连通标记，以及连续 3 帧
失败后停用旧深度引导。当前环境没有可读取的历史
RGB-D 原始帧目录，因此真实 SDK 的 `recovery` 和 `total` 提升幅度必须在
相同人数、站位和预热帧数下与 `hybrid` 做现场 A/B 测量，不能由合成测试
推断绝对毫秒数。

## 40. 按需点云恢复的 Adaptive Hybrid（2026-08-11）

为验证“正常关节走快速路径、只有疑难关节进入鲁棒恢复”，多人入口新增
实验模式 `adaptive_hybrid`。原 `hybrid` 的行为和默认选择保持不变，前端
WebSocket schema、`persons[]` 和每人 26 个 Halpe 关节也保持不变。

处理顺序为：

1. 每个人先执行一次原有 26 点快速窗口深度恢复。
2. 只在二维分数有效时检查深度置信度、相对躯干深度以及头颈、肩颈、
   上臂和前臂的三维长度。
3. 触发单位不是孤立关节，而是脸部、左臂和右臂三个解剖组。这样点云
   恢复仍能同时获得 nose/eye/ear 或 shoulder/elbow/wrist 的组内上下文。
4. 所有组均通过快速质量门控时，本帧不创建 `HxWx3` 有组织点云。
5. 至少一个组可疑时才创建一次共享点云，并只恢复被触发组；鲁棒结果仍以
   fail-closed 方式覆盖该组，其他快速关节保持不变。

默认试验门限位于 `configs/camera.yaml` 的
`camera.depth_recovery.adaptive_hybrid`：快速深度置信度 `0.55`、相对躯干
深度差 `0.45 m`、头颈 `0.40 m`、上臂 `0.50 m`、前臂 `0.45 m`。这些值
仅用于决定是否进入原点云鲁棒恢复，不直接接受异常关节。

现场 A/B 命令：

```bash
cd /home/fr1511b/program/workspace/humanpose
conda activate rgbd-avatar

PYTHONPATH=src python scripts/view_live_multi_person.py \
  --source sdk --device cuda:0 --detector auto \
  --identity-tracker shadow \
  --recovery-method adaptive_hybrid \
  --max-persons 4
```

控制台现会将恢复耗时拆成：

```text
recovery=... (fast=... cloud=... robust=... refine=... joints=...)
```

- `fast`：26 点快速窗口恢复总耗时。
- `cloud`：有组织点云创建耗时；无疑难组时应为 `0.0`。
- `robust`：被触发组的点云表面恢复耗时。
- `refine`：`guided_window/depth_connected` 的关联后第二阶段耗时；
  `adaptive_hybrid` 中应为 `0.0`。
- `joints`：本帧实际请求点云鲁棒恢复的关节总数，而不是最终有效点数。

建议先让 1、2、4 人分别稳定站立，再做手臂贴近身体、头部靠近背景和两人
交叉。若 `joints` 经常接近 `13 × 人数`，说明当前门限过于敏感，性能会接近
原 `hybrid`；若 `joints=0` 时仍出现头分离或手臂延伸，说明快速质量门控漏判，
需要收紧对应门限。任何时候都可以切回 `--recovery-method hybrid`。

## 41. 人物退出后的幽灵骨架修复（2026-08-11）

现场画面中 `detected=1` 但仍有 `tracks=2/3`，并出现蹲折或拉长骨架。根因
不是 `adaptive_hybrid` 的点云恢复本身，而是多人公共链路把“短时保留轨迹
ID”和“继续输出逐关节速度预测”混为一体：整个人消失后，26 个关节各自按
历史速度外推，速度差会迅速破坏骨长；同时只有二维检测、没有任何可靠深度
关节的观测还会错误地给旧轨迹续命。

修复后的规则为：

1. 深度恢复结果必须满足多人三维质量门控，才允许参与
   `geometry/shadow` 关联、创建轨迹或刷新轨迹存活时间；全关节无效会直接
   被拒绝。
2. `guided_window/depth_connected` 在关联后细化为全关节无效时，会撤销该次
   匹配，同样不能创建或续命。
3. 整个人未观测时，时序滤波器内部状态仍短时保留，供同一 `track_id` 重新
   进入时平滑恢复；对外的 `usable/observed/predicted` 全部清零。
4. 本地 Open3D/程序化火柴人跳过 `observed_in_frame=false` 的轨迹；多人
   WebSocket 仍可保留该 `track_id`，但其 26 个关节全部发送为 `null`。
5. 仅个别关节缺失、人物其余关节仍有可靠三维观测时，原有逐关节短时预测
   保持不变，不会因为本修复造成手腕或脚部立即闪断。
6. 连续缺失仍累计历史深度门控的失败次数，达到 3 帧后允许从新深度重新
   捕获，避免隐藏输出反过来阻碍远距离重新进入。

该保护位于多人处理、渲染和发布的公共层，因此同时覆盖 `hybrid`、
`adaptive_hybrid`、`guided_window`、`depth_connected`、
`window_median` 和 `pointcloud_cluster`，无需增加新的运行参数。

## 42. 有数值但几何错误的多人骨架门控（2026-08-11）

第二次现场复测中，`hybrid` 和 `adaptive_hybrid` 仍能在人物经过门板或离开
画面时产生弯折骨架。与第 41 节的全人缺失不同，这些帧日志仍显示
`status=ok`、`usable=26`：深度恢复确实返回了有限数值，但多个关节分别落在
人体、门板、背景或地面表面，原逻辑把“数值有效”误当成“人体几何有效”。

多人公共链路新增 `pose3d_quality`，位于 `configs/tracking.yaml`，并在所有
深度恢复模式之后执行轻量级 O(26) 检查：

1. 至少保留 8 个关节且躯干 6 点中至少有 3 点有效。
2. 躯干有效点的深度跨度不得超过 `0.55 m`。
3. 检查髋、腿、躯干、头颈、肩和手臂 14 条核心连接的宽松人体绝对上限。
4. 同一轨迹骨长先验完成标定后，连接长度还必须位于历史中值的
   `1 / 1.65` 到 `1.65` 倍内。
5. 根据真实帧时间计算关节允许的深度移动；5 个以上关节同时跳到另一深度
   层，或 neck/hip 根节点发生异常深度跳变时，拒绝整个人。
6. 仅一条非躯干连接异常时，不拒绝其余人体，而是使对应手臂或腿部分支
   fail-closed，由已有时序状态短时承接；两条以上核心连接异常则整人隐藏。

二维出画面门控也同步收紧：bbox 接触顶部、左侧或右侧时，即便二维模型补出
26 个高分关节，也标记为 `partial_person_out_of_frame` 并隐藏，直到 bbox 完整
回到图像内部。底部接触仍允许“脚部关键点完整”的正常人物，避免站在图像
下沿时误隐藏。

控制台和二维窗口新增三个现场诊断值：

- `quality=... ms` / `quality=...ms`：三维质量门控耗时。
- `reject` / `qreject`：本帧被整人拒绝的检测数量。
- `invalid` / `qinvalid`：本帧被屏蔽的关节数量。

发生截图中的错误时，应看到 `qreject>0`，或 `qinvalid>0` 且异常肢体不再
绘制。正常稳定人物应保持二者为 0。该检查不构造点云，也不改变启动命令和
WebSocket schema。

## 43. Depth Connected 单阶段与共享连通域优化（2026-08-11）

此前多人 `depth_connected` 的实际链路为：

```text
26点 window_median -> 身份关联 -> 26点 depth_connected
```

而第二阶段中每个关节先尝试基础窗口、失败后再尝试扩展窗口，两次都可能独立
执行 BFS 连通域，因此单人一帧最坏接近 52 次局部聚类，4 人时会放大为约
208 次。

优化后链路为：

```text
二维/外观关联 -> 每人一次联合ROI连通标记 -> 26个关节查询共享组件
```

- `depth_connected` 在关联前使用空三维占位，不调用 `recover_pose3d`；关联
  依据二维 bbox、关键点和可选 `shadow` 外观描述。
- 已有轨迹的躯干历史中值作为当前人物深度软提示，每关节历史仍独立参与候选
  评分；新人物没有历史时直接使用当前空间证据。
- 联合 ROI 是 26 个扩展关节圆窗的并集，不是整个 bbox 或全幅深度图，避免
  对大面积无关背景做连通标记。
- 八邻域的深度边缘关系通过 NumPy 批量生成稀疏邻接图，再由 SciPy
  `csgraph.connected_components` 在底层实现中求组件，不再用 Python 队列
  逐像素执行大区域 BFS。
- 共享标签只负责空间连通性；每个关节仍在自己的圆窗交集中重新计算局部
  深度中值、支持度和中心距离，因此不会把整个人强制压成同一个深度。
- 基础半径有可靠候选时直接返回；只有缺少候选才查询扩展半径，但不再重新
  聚类。

现场日志中该模式应满足 `fast=0.0`，主要耗时集中在 `refine`；前端事件、
`track_id`、`persons[]` 和每人 Halpe26 数据结构均不变。自动回归会统计共享
连通函数的实际调用次数，要求 26 个关节合计为 1 次，而不是仅比较输出结果。

在 `816×612`、26 点分布于 `220×480` 人体框、均匀 3 m 深度的保守合成基准
中，旧逐关节候选聚类约 `10.36 ms/人`，新共享方案包含完整拓扑选择约
`8.14 ms/人`，该阶段约为 `1.27×`。这不是现场相机的最终帧率结论；实际总
收益还包含取消一次 26 点快速恢复，需继续以相同人数和站位比较日志中的
`recovery/total`。

## 44. RTMO 单阶段多人二维姿态实验后端（2026-08-11）

当前 RTMPose 多人链路属于 top-down：先运行人体检测器，再对每个 bbox 分别
执行 RTMPose，因此 `inference` 会随人物数量明显增长。为验证单阶段多人模型
是否能消除这段人数斜率，`view_live_multi_person.py` 新增显式启用的 RTMO
实验后端。默认配置仍为 `pose.backend: rtmpose`，不传新参数时现有链路完全
不变。

实验配置使用 MMPose 1.3.2 自带模型索引中的
`rtmo-t_8xb32-600e_body7-416x416`。它直接从整幅图像同时输出多人 COCO17，
不初始化额外人体检测器。第一次运行且本地没有缓存时，MMPose 会下载官方
checkpoint 到 `assets/models/cache`；也可以通过 `--rtmo-checkpoint` 指定已经
下载的本地权重。

后续深度、跟踪和前端固定使用 Halpe26，因此适配层执行以下转换：

1. COCO17 与 Halpe26 相同顺序的前 17 个身体点直接复制。
2. `head` 使用 nose 的稳定二维位置，`neck` 取左右肩中点，`hip` 取左右髋
   中点；合成点置信度采用对应源点的保守最小值。
3. RTMO 不输出的六个细分足点保持置信度 0，深度恢复会将它们视为不可用；
   左右 ankle 仍然存在。
4. 输出仍然是每人 26 点的 `Pose2D`，所以 `depth_connected`、身份跟踪、
   Open3D、本地二维窗口和多人 WebSocket schema 都不需要分叉。

实时试运行命令：

```bash
cd /home/fr1511b/program/workspace/humanpose
conda activate rgbd-avatar

PYTHONPATH=src python scripts/view_live_multi_person.py \
  --source sdk --device cuda:0 \
  --pose-backend rtmo \
  --identity-tracker shadow \
  --recovery-method depth_connected \
  --max-persons 4
```

立即切回原正式方法只需删除 `--pose-backend rtmo`，或显式设置：

```bash
--pose-backend rtmpose --detector auto
```

迁移前应使用同一段录制 RGB-D 数据分别运行两个后端，保持
`--recovery-method`、`--max-persons` 和所有阈值一致。建议丢弃首次模型预热的
30 帧，然后分别统计 1、2、4 人的 `inference/total` 中位数和 P95，同时记录
检测人数、每人有效三维关节数、骨架拒绝数以及交叉时 ID 交换次数。RTMO 的
目标不是只让某一帧更快，而是使 `inference` 对人数的增长斜率明显低于
RTMPose；若四人总耗时没有稳定下降，或者漏检、远距离骨架和遮挡质量明显
变差，则继续保留为实验选项，不修改默认后端。
