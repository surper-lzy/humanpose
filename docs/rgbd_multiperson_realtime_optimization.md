# RGB-D 多人实时人体骨架系统性能优化方案

## 1. 项目目标

当前系统目标：

- 单台 RGB-D 相机；
- 支持尽可能多的人体；
- 实时提取多人二维/三维骨架；
- 保持稳定多人 ID；
- 将三维骨架发送给仿生人驱动模块；
- 目标 30 FPS；
- 更核心的指标是：**人数增加时，端到端延迟增长尽可能缓慢。**

当前相机输出：

- RGB 图
- 深度图
- 强度图
- 点云

当前 Pose Backend：

- RTMPose
- RTMO
- 后续可扩展其他多人 Pose 模型

---

# 2. 当前系统链路

## GPU

```text
GPU
├── 人体检测器
└── RTMPose / RTMO 推理
```

## CPU

```text
CPU
├── RGB和深度图预处理
├── depth_connected
├── 点云恢复与聚类
├── 多人ID关联
├── Shadow外观特征
├── 三维质量检查
├── 时序和骨长约束
├── 可视化
└── WebSocket发送
```

当前最明显的问题：

> GPU 主要只做神经网络推理，Pose 结束后大部分数据处理重新回到 CPU。

随着人数增长，CPU 端的 `depth_connected`、点云恢复、聚类、Shadow、多人后处理、可视化等模块会快速放大，并且产生频繁的 GPU→CPU→GPU 数据搬运。

---

# 3. 优化总原则

目标架构应从：

```text
GPU做Pose
↓
CPU做一切
```

改成：

```text
GPU完成：
RGB/Depth预处理
→ Pose
→ 深度恢复
→ 3D骨架
→ 质量检查
→ 时序滤波
→ 骨长约束
→ Retarget

CPU只负责：
Track生命周期
Hungarian
参数管理
Visualization Worker
WebSocket Worker
```

核心原则：

1. **整幅点云退出 30 FPS 骨架主链。**
2. **避免 per-person / per-joint Python 循环。**
3. **Pose 后尽量保持 CUDA Tensor，不落回 NumPy。**
4. **可视化和 WebSocket 不进入关键路径。**
5. **多人后处理统一 batch 化。**
6. **昂贵模块按需或低频执行。**

---

# 4. 推荐的新架构

```text
                        RGB-D Camera
                  ┌─────────┴─────────┐
                 RGB                Depth
                  │                   │
                  ▼                   ▼
        ┌─────────────────┐   ┌──────────────────┐
GPU     │ RGB preprocess  │   │ Depth preprocess │
        └────────┬────────┘   └─────────┬────────┘
                 │                      │
                 ▼                      │
        ┌─────────────────┐             │
GPU     │ RTMO / RTMPose  │             │
        │ TensorRT FP16   │             │
        └────────┬────────┘             │
                 │                      │
          keypoints [N,K,2]             │
                 └────────────┬─────────┘
                              ▼
                 ┌────────────────────────┐
GPU              │ Batched Depth Recovery │
                 │ [N,K,S,S]              │
                 └────────────┬───────────┘
                              ▼
                        XYZ [N,K,3]
                              │
                              ▼
                 ┌────────────────────────┐
GPU              │ 3D Quality Check       │
                 │ Temporal Filter        │
                 │ Bone Constraints       │
                 └────────────┬───────────┘
                              ▼
                      Stable 3D Skeleton
                              │
                 ┌────────────┼─────────────┐
                 ▼            ▼             ▼
CPU          Tracking     Visualization   WebSocket
             Hungarian       Worker         Worker
```

---

# 5. P0：重写 Depth Recovery

## 5.1 当前问题

旧链路：

```text
Depth
↓
生成整幅 816×612×3 XYZ 点云
↓
每个人建立候选人体区域
↓
每个关节执行连通域
↓
扩大窗口
↓
再次聚类
↓
拓扑筛选
↓
输出 XYZ
```

816×612 深度图约为：

```text
499,392 pixels
```

若 20 人、每人 17 个关节：

```text
20 × 17 = 340 joints
```

最终只需要 340 个三维点，却先处理约 50 万个像素并按人、按关节做 CPU 聚类。

## 5.2 推荐方案：Sparse Batched Depth Recovery

统一接口：

```text
输入：
depth      [H,W]
keypoints  [N,K,2]
confidence [N,K]

输出：
xyz        [N,K,3]
valid      [N,K]
```

主路径：

```text
所有2D joints
    +
Depth
    ↓
GPU一次局部采样
    ↓
[N,K,S,S]
    ↓
invalid depth过滤
    ↓
person depth gate
    ↓
temporal depth gate
    ↓
weighted median / robust depth
    ↓
Z [N,K]
    ↓
批量反投影
    ↓
XYZ [N,K,3]
```

例如：

```text
N=20
K=17
S=11
```

局部采样数量：

```text
20 × 17 × 11 × 11 = 41,140
```

远小于整图约 50 万点。

---

# 6. depth_connected 改造成 fallback

不要让 `depth_connected` 默认作用于所有人所有关节。

推荐三级恢复：

## Level 1：Fast

```text
7×7 patch
+
valid depth
+
median / weighted median
```

## Level 2：Robust

```text
15×15 patch
+
person torso depth prior
+
previous joint depth
+
confidence
```

## Level 3：Fallback

仅对失败关节：

```text
local ROI
↓
connected component
↓
best component
```

即：

```text
旧：
N × K × connected component

新：
绝大多数 joint → GPU fast recovery
少数 failed joint → local cluster fallback
```

---

# 7. 多人深度串线处理

多人遮挡时不能简单：

```text
Z = depth[v,u]
```

因为该像素可能对应另一个前景人物。

为每个 track 维护躯干深度：

```text
Z_root
```

可由左右肩、左右髋附近深度的鲁棒中值获得。

第 n 个人的候选深度应满足：

```text
abs(Z - Z_root[n]) < person_depth_threshold
```

并结合上一帧：

```text
abs(Z - Z_prev[n,k]) < temporal_threshold
```

推荐每个关节维护状态：

```text
VALID
LOW_CONFIDENCE
OCCLUDED
PREDICTED
```

完全遮挡时，不继续扩大窗口“硬找深度”，而使用：

- 上一帧位置；
- 速度预测；
- 父关节；
- 骨长约束。

---

# 8. 整幅点云退出实时主链

## 30 FPS 主链

```text
RGB
↓
Pose
+
Depth
↓
Batched Joint Recovery
↓
3D Skeleton
```

## 低频链

```text
PointCloud
├── Ground Plane       0.5~2 Hz
├── Height Estimation  按Track偶尔执行
└── Scene Geometry     按需
```

身高估计成功后缓存，不需要每帧重新计算。

---

# 9. RGB / Depth 预处理迁移 GPU

当前 CPU 中：

```text
resize
normalize
mask
depth filtering
crop
```

应尽量迁移 GPU。

推荐：

```text
Camera Buffer
↓
Pinned Host Memory
↓ non_blocking H2D
CUDA
↓
GPU preprocess
```

避免：

```text
NumPy
→ OpenCV
→ NumPy
→ Torch
→ CUDA
```

的多次转换。

---

# 10. Pose 模块

## RTMPose

若保留 RTMPose：

```text
Detector
↓
所有bbox
↓
batch crop
↓
[N,3,H,W]
↓
一次RTMPose batch
```

不要逐人调用模型。

部署方向：

```text
PyTorch
↓
ONNX
↓
TensorRT FP16
```

但 RTMPose 属于 top-down，人数增长仍会增加 Pose 工作量。

## RTMO

若采用 RTMO：

```text
RGB
↓
RTMO
↓
所有人Pose
```

可取消独立人体检测器。

建议实验定位：

```text
RTMPose = top-down baseline
RTMO     = 多人实时主候选
```

---

# 11. 多人 ID 关联

不需要把整个 tracking 全部搬 GPU。

## GPU 负责代价矩阵

计算：

```text
bbox distance
pose distance
3D root distance
depth distance
appearance cosine distance
```

生成：

```text
cost_matrix [N_current,N_previous]
```

## CPU 保留

```text
Hungarian
Track lifecycle
ID create/delete
```

CPU 只处理小矩阵和元数据。

---

# 12. Shadow 外观特征

若 Shadow 是外观 / ReID embedding，不要每个人每帧都运行。

推荐：

```text
稳定Track
→ 不更新

新Track
→ 计算

两人交叉
→ 计算

ID歧义
→ 计算

每隔若干帧
→ 可选刷新
```

形成：

```text
cheap tracking
bbox + pose + depth + motion
          ↓
是否歧义？
          ↓ yes
Shadow appearance
```

若 Shadow 是神经网络：

```text
所有需要更新的人
↓
batch crop
↓
一次GPU embedding
```

---

# 13. 三维质量检查 GPU batch 化

统一输入：

```text
joints     [N,K,3]
confidence [N,K]
```

批量执行：

```text
depth range check
velocity check
confidence gate
bone length check
outlier rejection
```

输出：

```text
valid_mask [N,K]
```

禁止主路径 Python 双层循环。

---

# 14. 时序与骨长约束 Tensor 化

建议维护：

```text
position      [M,K,3]
velocity      [M,K,3]
confidence    [M,K]
bone_length   [M,B]
valid         [M,K]
track_id      [M]
```

批量执行：

```text
EMA / One Euro
velocity prediction
occlusion recovery
bone-length projection
```

而不是大量：

```text
track[id].joint[j].history
```

对象级循环。

---

# 15. Visualization 独立 Worker

禁止：

```text
Compute
↓
cv2.circle
↓
cv2.line
↓
cv2.putText
↓
imshow
↓
Next frame
```

改成：

```text
Compute Thread
↓
latest_frame
↓
Visualization Worker
```

如果 GUI 跟不上，直接丢弃旧帧。

例如：

```text
Pose       30 FPS
Skeleton   30 FPS
WebSocket  30 FPS
GUI        10~20 FPS
```

GUI 不得反向阻塞主链。

---

# 16. WebSocket 独立 Worker

禁止：

```text
compute
↓
json.dumps
↓
send
↓
wait
↓
next frame
```

改成：

```text
Skeleton Producer
↓
Latest-State Queue
↓
WebSocket Worker
```

队列只保存最新状态。

网络慢时：

```text
Frame100
Frame101
Frame102
Frame103
```

只需要发送最新的 Frame103。

实时动作系统优先：

```text
低延迟
```

而不是：

```text
每帧可靠回放
```

---

# 17. 推荐 CPU / GPU 职责

## GPU

```text
GPU
├── RGB preprocess
├── Depth preprocess
├── Person detector（RTMPose时）
├── RTMPose / RTMO
├── Batched depth recovery
├── Person-aware depth filtering
├── XYZ back-projection
├── 3D quality check
├── Temporal filtering
├── Bone constraints
├── Appearance embedding（按需）
└── Retarget
```

## CPU

```text
CPU
├── Camera SDK control
├── Track lifecycle
├── Hungarian assignment
├── Parameter management
├── Visualization worker
└── WebSocket worker
```

---

# 18. CUDA Stream

可在主架构稳定后尝试：

```text
Stream 0:
RGB preprocess → Pose

Stream 1:
Depth upload → Depth preprocess

Stream 2:
previous frame filter / retarget
```

但必须 profile。

如果 Pose 已经占满 GPU，增加 stream 不一定提高性能。

---

# 19. CUDA Graph

当每帧执行结构稳定后：

```text
preprocess
↓
pose
↓
depth recovery
↓
filter
```

可以考虑 CUDA Graph 降低 kernel launch overhead。

这是后期优化，不应早于 Depth Recovery 重构。

---

# 20. 改造优先级

## P0：Depth Recovery

删除主路径：

```text
full pointcloud
per-person cluster
per-joint Python BFS
```

实现：

```text
DepthPatchRecovery
```

---

## P1：IO 解耦

拆出：

```text
Visualization Worker
WebSocket Worker
```

使用 latest-only queue。

---

## P2：GPU 预处理

实现：

```text
Pinned Memory
+
non_blocking H2D
+
GPU resize/normalize/filter
```

---

## P3：3D Pipeline Batch 化

将：

```text
3D QC
Temporal
Bone Constraints
```

统一改成 `[N,K,3]` Tensor 运算。

---

## P4：Tracking 优化

实现：

```text
GPU cost matrix
+
CPU Hungarian
+
RGB-D depth gating
```

---

## P5：Shadow 优化

改为：

```text
ambiguity-triggered appearance
```

---

## P6：Pose TensorRT

```text
ONNX
↓
TensorRT FP16
```

---

## P7：Streams / CUDA Graph

只在 profiler 显示 GPU 调度或 idle 时间成为主要问题后再做。

---

# 21. 30 FPS 性能预算

总预算：

```text
33.3 ms / frame
```

建议关键路径目标：

```text
Pose                     10~18 ms
GPU Depth Recovery        2~4 ms
Tracking                   1~2 ms
3D QC / Filter / Bone      1~3 ms
Retarget                   1~2 ms
----------------------------------
Critical Compute          15~29 ms
```

以下模块不进入关键路径：

```text
Visualization
WebSocket
Height
Ground
Full PointCloud
Appearance refresh
```

这些数值是工程目标，不是当前硬件性能保证。

---

# 22. Profiler 必须记录

建议每帧输出：

```text
capture_ms
rgb_h2d_ms
depth_h2d_ms
preprocess_ms

detector_ms
pose_ms
pose_postprocess_ms

depth_patch_ms
depth_filter_ms
xyz_ms
fallback_cluster_ms

tracking_cost_ms
hungarian_ms
shadow_ms

qc_ms
temporal_ms
bone_ms
retarget_ms

visualization_ms
websocket_ms

critical_path_ms
total_latency_ms
fps
```

同时记录：

```text
detected_persons
active_tracks
valid_3d_joints
fallback_joint_count
gpu_utilization
cpu_utilization
```

---

# 23. 多人性能实验

测试人数：

```text
1
2
4
8
12
16
20
24
32
```

记录：

| 人数 | Pose | Depth | Tracking | 3D Filter | Total | FPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | | | | | | |
| 2 | | | | | | |
| 4 | | | | | | |
| 8 | | | | | | |
| 12 | | | | | | |
| 16 | | | | | | |
| 20 | | | | | | |
| 24 | | | | | | |
| 32 | | | | | | |

核心指标：

> 人数增加时，Latency 曲线增长尽可能缓慢。

---

# 24. Codex 第一阶段任务

## Task 1：实现 BatchedDepthRecovery

要求：

```text
输入：
depth [H,W]
keypoints [N,K,2]
confidence [N,K]

输出：
xyz [N,K,3]
valid [N,K]
```

必须满足：

- 不生成 full pointcloud；
- 主路径无 Python per-joint BFS；
- 支持任意 N；
- GPU batch；
- 支持 7×7 / 15×15 patch；
- 支持 torso depth prior；
- 支持 previous depth prior；
- 失败返回 invalid；
- fallback cluster 为独立接口。

---

## Task 2：拆分 Visualization / WebSocket

实现：

```text
latest frame queue
latest skeleton queue
```

要求：

- 队列最大长度 1 或极小；
- 新数据覆盖旧数据；
- 不阻塞实时主线程；
- GUI / 网络慢时主链 FPS 不下降。

---

## Task 3：增加完整 Profiler

输出例如：

```text
persons=12
pose=12.8ms
depth=3.4ms
track=1.2ms
filter=0.9ms
critical=19.1ms
fps=52.3
fallback=8
```

---

# 25. 最终核心链路

```text
RGB-D Camera
      │
      ├──────── RGB ────────┐
      │                     │
      └──────── Depth ──────┤
                            ▼
                     GPU Preprocess
                            │
                            ▼
                      Multi-Person Pose
                            │
                            ▼
                     [N,K,2] joints
                            │
Depth ──────────────────────┤
                            ▼
                 Batched Depth Recovery
                            │
                            ▼
                      [N,K,3] joints
                            │
                            ▼
              Quality / Temporal / Bone
                            │
                            ▼
                         Tracking
                            │
                            ▼
                        Retarget
                            │
                    ┌───────┴────────┐
                    ▼                ▼
              Visualization       WebSocket
                Worker              Worker
```

最终设计原则：

> **从 Pose 开始，数据尽量保持在 GPU。**

> **整幅点云不进入实时骨架主链。**

> **所有 per-person / per-joint Python 循环尽可能改成 Tensor Batch。**

> **Visualization、WebSocket、身高、地面等模块从 30 FPS 关键路径中移除。**

> **最终优化目标不是某一个模块最快，而是人数增加时整条系统的延迟增长尽可能缓慢。**
