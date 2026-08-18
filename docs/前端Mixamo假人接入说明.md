# 前端 Mixamo 假人接入说明

本文面向前端开发人员，说明如何复用现有的 Halpe26 实时骨架消息，
在页面中二选一地显示三维火柴人或带蒙皮的 Mixamo 假人。

本文主体描述原单人 `avatar.stickman.updated` 路径。可选多人路径现使用
`avatar.stickmen.updated`，但每个人内部仍是相同的 Halpe26 数据；多人
消息和显示数量控制见本文末尾“多人扩展”。

通用 WebSocket 控制协议见
[`./实时WebSocket接口说明.md`](实时WebSocket接口说明.md)，Halpe26 业务消息、
关节顺序和状态处理见
[`./前端实时火柴人数据接入说明.md`](前端实时火柴人数据接入说明.md)。
本文不新增第二条实时数据协议。

可直接交付给前端的 GLB、TypeScript 模块、fixtures 和 README 已生成在
[`../frontend/mixamo-avatar-delivery/`](../frontend/mixamo-avatar-delivery/README.md)。

## 1. 接入结论

后端始终只发布一种业务消息：

```text
event = avatar.stickman.updated
topic = avatar:stickman:FF6690772788
data  = payload.joints（固定 26 项 Halpe26 三维关节）
```

前端使用同一个 `payload` 选择一种显示方式：

```text
avatar.stickman.updated
          │
          └─ payload.joints
                 ├─ displayMode = stickman
                 │      └─ 关节球 + 骨架连线
                 │
                 └─ displayMode = mixamo
                        └─ Halpe26 重定向
                              → Mixamo 骨骼旋转
                              → Three.js GPU 蒙皮
```

切换角色或显示方式时，不重连 WebSocket，不更换 topic，也不要请求后端
发送骨骼矩阵、蒙皮后顶点或 `mixamo_sequence.npz`。

## 2. 交付给前端的内容

交付内容分为“实时数据”和“静态前端资源”。静态资源通过前端工程或 HTTP
部署，不通过实时 WebSocket 重复发送。

| 交付项 | 形式 | 当前状态 | 前端用途 |
| --- | --- | --- | --- |
| WebSocket 接入协议 | 现有两份 Markdown | 已存在 | 连接、订阅、重连和消息过滤 |
| Halpe26 关节和连接表 | TypeScript 常量 | 文档中已给出 | 火柴人绘制和重定向输入 |
| Mixamo 蒙皮模型 | 每个角色一个 `.glb` | 已生成 `character-a.glb` | 网格、纹理、骨架和蒙皮 |
| Mixamo 重定向器 | `mixamoRetargeter.ts` | 已生成初版 | 26 关节转换为 Mixamo 骨骼旋转 |
| 模型注册表 | `avatarRegistry.ts` 及 manifest | 已生成 | 角色列表、GLB URL 和模型级覆盖参数 |
| 统一显示控制器 | `avatarController.ts` | 已生成初版 | 在火柴人与假人之间切换 |
| 固定姿态样例 | JSON 测试夹具 | 已生成 4 帧 | 验证左右、坐标系和骨骼映射 |

建议在前端工程中按以下结构交付，具体目录可根据前端项目调整：

```text
public/
└─ avatars/
   ├─ character-a.glb
   └─ character-b.glb

src/avatar/
├─ types.ts
├─ halpe26.ts
├─ stickmanRenderer.ts
├─ mixamoRetargeter.ts
├─ avatarRegistry.ts
└─ avatarController.ts
```

### 2.1 当前模型源文件

当前后端使用：

```text
assets/models/mixamo/Ch09_nonPBR.fbx
```

当前解析结果为 15,716 个顶点、31,292 个三角形、65 根骨骼，包含纹理和
每顶点最多 4 个蒙皮权重。该 FBX 是导出 GLB 的源资产，不是最终前端实时
数据。

## 3. 实时消息保持不变

浏览器仍然按火柴人文档订阅：

```text
ws://192.168.30.132:8000/api/realtime/ws?client_type=browser&client_id=avatar-ui&topics=avatar%3Astickman%3AFF6690772788
```

只处理同时符合以下条件的消息：

```text
message.type  = event
message.event = avatar.stickman.updated
message.topics 包含 avatar:stickman:FF6690772788
```

两种显示模式都读取 `message.payload`。不要只保留一个裸的 `26×3` 数组，以下
字段也是前端状态管理所必需的：

| 字段 | 用途 |
| --- | --- |
| `frame_number` | 识别帧、latest-only 更新和联调 |
| `timestamp_ms` | 计算 IK 时间间隔，不依赖渲染帧率 |
| `status` | 控制隐藏、淡出和重置跟踪状态 |
| `coordinate_system` | 约束单位、手性、向上轴和地面 |
| `joints` | 固定 26 项，每项为 `[x,y,z]` 或 `null` |

当前协议故意不包含置信度、`predicted` 标记、骨骼矩阵和手部关节。

## 4. 前端显示模式

前端只维护一份最新姿态：

```ts
type DisplayMode = "stickman" | "mixamo";

let displayMode: DisplayMode = "stickman";
let latestPose: StickmanPayloadV1 | null = null;
let latestArrivalMs = 0;
```

收到通过协议检查的消息时，替换 `latestPose`，不建立待播放的无限队列。
渲染循环根据当前模式只显示一种对象：

```ts
function renderLatestPose(payload: StickmanPayloadV1) {
  if (displayMode === "stickman") {
    mixamoController.hide();
    stickmanRenderer.show(payload.joints);
    return;
  }

  stickmanRenderer.hide();
  mixamoController.show(payload);
}
```

切换到 Mixamo 模式或切换角色时：

1. 隐藏之前的显示对象。
2. 延迟加载并缓存选中的 GLB。
3. 从 GLB `Skeleton` 读取骨骼、父子关系和 bind pose。
4. 为新模型创建独立的重定向器状态。
5. 重置上一帧旋转，再使用最新有效姿态开始驱动。

## 5. GLB 模型要求

每个可选角色的 GLB 必须是完整蒙皮模型，而不是普通静态 Mesh。

必须包含：

- `SkinnedMesh`、`Skeleton` 和有名称的骨骼节点。
- bind pose 和 inverse bind matrices。
- 每顶点骨骼索引及归一化权重。
- UV、材质和纹理。
- 明确的米制比例和正确的初始朝向。

通用 Mixamo 重定向器至少需要找到以下骨骼：

```text
Hips
Spine Spine1 Spine2 Neck Head
LeftShoulder LeftArm LeftForeArm LeftHand
RightShoulder RightArm RightForeArm RightHand
LeftUpLeg LeftLeg LeftFoot LeftToeBase
RightUpLeg RightLeg RightFoot RightToeBase
```

如果新模型使用 `mixamorig:Hips` 这类带前缀名称，加载器应在匹配时去掉
`mixamorig:` 前缀，或在模型注册表里提供别名映射。

## 6. Halpe26 到 Mixamo 的重定向

### 6.1 输入限制

火柴人直接经过 26 个测量点；Mixamo 假人则必须将关节方向变成固定
骨长骨架的旋转。26 个三维点不能唯一决定 65 根骨骼的所有轴向旋转，
因此前端需要 bind pose、躯干参考平面和上一帧状态来补齐约束。

当前 Python 实现可作为数学参考：

```text
src/rgbd_avatar/retargeting/halpe_mixamo.py
```

不能原样逐行移植：Python 实现还使用了未在现有 WebSocket 协议中发送的
`confidence`、`usable`、`predicted` 和可选 Hand21 数据。前端版本应将
`joint !== null` 作为有效性判断，对缺失段保持上一帧或回到 bind pose。

### 6.2 关节与骨骼驱动关系

| Mixamo 部位 | Halpe26 参考 | 用途 |
| --- | --- | --- |
| `Hips` | `hip(19)`、左右髋 `(11,12)`、颈 `(18)` | 根位置和人体朝向 |
| `Spine*` | `hip(19) → neck(18)` | 躯干方向，旋转可在脊柱骨之间分配 |
| `LeftShoulder` | `neck(18) → left_shoulder(5)` | 左肩带 |
| `LeftArm` | `5 → 7` | 左上臂 |
| `LeftForeArm` | `7 → 9` | 左前臂 |
| `RightShoulder` | `neck(18) → right_shoulder(6)` | 右肩带 |
| `RightArm` | `6 → 8` | 右上臂 |
| `RightForeArm` | `8 → 10` | 右前臂 |
| `LeftUpLeg` | `11 → 13` | 左大腿 |
| `LeftLeg` | `13 → 15` | 左小腿 |
| `RightUpLeg` | `12 → 14` | 右大腿 |
| `RightLeg` | `14 → 16` | 右小腿 |
| `LeftFoot` | `heel(24) → mean(big_toe(20), small_toe(22))` | 左脚朝向 |
| `RightFoot` | `heel(25) → mean(big_toe(21), small_toe(23))` | 右脚朝向 |
| `Neck` | `neck(18) → head(17)` | 头颈方向 |
| 手、手指 | Halpe26 不足 | 暂时保持 bind pose；后续接入 Hand21 |

`left` 和 `right` 仍指被拍摄人物自身的解剖学左右，不得根据屏幕观感
交换索引。

Mixamo 的 `Foot` 节点位于脚踝，`Foot → ToeBase` 在绑定骨架中同时向前、
向下，并不等同于水平的脚底方向。求解脚掌旋转时，应先把绑定姿势的
`Foot → ToeBase` 投影到绑定地面，再与 Halpe26 的 `heel → toe center`
对齐；直接对齐两条原始向量会使站立人物的脚掌向上翘。

### 6.3 建议求解顺序

1. 检查 `status`、`joints.length === 26` 及每个关节的有效性。
2. 将所有有效点统一从应用 Z-up 坐标转为 Three.js Y-up 坐标。
3. 以髋部中心、颈部和左右髋/肩建立人体正交基。
4. 从每个可用的关节段方向建立目标旋转基。
5. 结合该模型的 bind 骨骼基计算目标全局旋转。
6. 用 `inverse(parentGlobal) * childGlobal` 转为 Three.js `Bone` 需要的局部旋转。
7. 使用 `timestamp_ms` 计算 `deltaTime`，对旋转速度限制和插值。
8. 更新骨骼 quaternion，保留 GLB 中初始骨长和局部偏移。
9. 求解完成后调用 `skeleton.update()`。

重定向应在每个新的 WebSocket 输入帧上最多执行一次，不要在 60 FPS 的
`requestAnimationFrame` 中对同一输入帧重复推进有状态的 IK。渲染循环只显示最新
完成的姿态。

## 7. 坐标系、根节点和比例

后端发送的是右手系、米制、`+Z` 向上坐标。Three.js 默认使用 `+Y` 向上。
所有关节只转换一次：

```ts
function applicationToThree(joint: [number, number, number]): THREE.Vector3 {
  return new THREE.Vector3(joint[0], joint[2], -joint[1]);
}
```

不要同时旋转角色根 Group 又对关节做上述转换，否则会二次转轴。

根位置优先使用 `hip(19)`；该点缺失时，可使用左右髋 `(11,12)` 的中点。
根节点缺失时不得将角色移到原点。

角色比例建议在首批稳定姿态中估计一次，并在当前跟踪期间保持不变。
可使用 `head(17)` 到左右踝 `(15,16)` 中点的中位高度与 GLB bind pose 身高
之比。不要每帧根据带噪声的关节距离改变模型 scale 或骨长。

## 8. 缺失关节和时序状态

该部分遵循火柴人协议中的现有规则：

- `null` 必须保持为无效值，不能变成 `[0,0,0]`。
- 单个身体段缺失时，假人对应骨骼短时保持上一帧。
- `partial_person_out_of_frame` 和 `awaiting_full_reentry` 时立即隐藏当前显示对象并重置 IK。
- `no_person` 可短时保留协议中仍非 `null` 的预测点，但数据超过 300～500 ms
  未更新时必须隐藏。
- `frame_number` 允许跳号，程序重启后也可从 0 重新开始。
- 断线重连、切换人物或跟踪重置后，清空上一帧骨骼旋转。

火柴人和假人共用同一个过期计时器和可见性状态，不要分别维护两套
“人是否在场”判断。

## 9. 多个 Mixamo 角色的注册

新增角色应当只增加 GLB 和一条注册信息，不新增 WebSocket 事件。

```ts
export interface AvatarDefinition {
  id: string;
  label: string;
  modelUrl: string;
  rigType: "mixamo";
  boneAliases?: Record<string, string>;
  scaleMultiplier?: number;
  rotationOffsetEuler?: [number, number, number];
}

export const AVATARS: AvatarDefinition[] = [
  {
    id: "character-a",
    label: "Character A",
    modelUrl: "/avatars/character-a.glb",
    rigType: "mixamo"
  }
];
```

`scaleMultiplier` 和 `rotationOffsetEuler` 只用于修正资产导出差异。标准化的 GLB
应优先在导出阶段修正单位和朝向，避免每个模型依赖大量手工偏移。

## 10. 不交付给前端的后端缓存

### 10.1 `mixamo_sequence.npz`

`mixamo_sequence.npz` 是 Python 离线拟合和查看器的缓存，包含多帧已蒙皮顶点、
多帧骨骼全局矩阵、纹理和后端 IK 元数据。它不是前端角色协议，也不包含
前端从任意 26 关节重新求解动作所需的完整蒙皮契约。

不向前端交付它的原因：

- 它绑定一段预先求解的历史序列，不是实时输入。
- 浏览器不原生解析 NumPy NPZ。
- 它重复存储每帧顶点，不符合前端 GPU 蒙皮方式。
- 它的元数据包含后端本地 FBX 路径，前端无法使用。

如果以后需要网页离线回放，另行将动作导出为 glTF animation clip，不改变
当前实时协议。

### 10.2 Python 查看器与 Filament 渲染器

`viser_live_mixamo.py`、`live_mixamo_process.py` 和 Filament 离屏渲染链路只用于
后端对照、调试和验证。前端使用 Three.js 直接渲染 GLB，不需要移植
Viser、Open3D、Filament 或 JPEG 回传逻辑。

## 11. 与当前 Python 实现的对照

| 后端文件 | 可供前端参考的内容 | 是否直接打包给前端 |
| --- | --- | --- |
| `src/rgbd_avatar/live/stickman_websocket.py` | 实时 payload 真实生成逻辑 | 否，以协议文档为准 |
| `src/rgbd_avatar/retargeting/halpe_mixamo.py` | 人体基、肢体目标基、bind 修正和时序旋转限制 | 否，需整理为 TypeScript |
| `src/rgbd_avatar/avatar/mixamo_asset.py` | 骨架、bind pose、inverse bind 和权重结构 | 否，最终封装在 GLB |
| `assets/models/mixamo/Ch09_nonPBR.fbx` | 当前模型源资产 | 否，先导出和验证 GLB |
| `mixamo_sequence.npz` | 后端离线结果对照 | 否 |

## 12. 建议的前端实现边界

### `StickmanRenderer`

- 输入 `Joint3[26]`。
- 直接更新关节球和连线。
- 不处理 GLB 和骨骼。

### `MixamoRetargeter`

- 一个 GLB `Skeleton` 对应一个实例。
- 在初始化时缓存 bind pose 和骨骼索引。
- 每个新输入帧求解最多一次。
- 输出/应用 Hips 位置、根旋转和各骨骼局部 quaternion。

### `AvatarController`

- 拥有唯一的 `latestPose`。
- 处理显示模式、角色选择、加载和隐藏。
- 统一执行过期检查、状态重置和坐标转换。
- 不创建第二条 WebSocket 连接。

## 13. 联调和验收清单

### 数据接入

- [ ] 页面只建立一条实时骨架 WebSocket 连接。
- [ ] 浏览器处理 `type=event`，而不是等待 `type=publish`。
- [ ] 只保留最新帧，允许 `frame_number` 跳号。
- [ ] `null` 关节没有被转换为原点。

### 火柴人

- [ ] 关节数和骨架连接与 Halpe26 文档完全一致。
- [ ] 左右关节未因屏幕观感被手工交换。

### Mixamo 假人

- [ ] GLB 中存在 `SkinnedMesh` 和所有必需骨骼。
- [ ] 模型使用米制比例，站立时脚底位于 Three.js 地面 `Y=0`附近。
- [ ] 举左手、踢左腿、转身和前后移动方向正确。
- [ ] 缺失手腕/踝部时不跳到原点。
- [ ] 每个输入帧只求解一次，没有因 60 FPS 渲染重复推进滤波器。

### 模式和角色切换

- [ ] 火柴人和假人任何时刻只显示一个。
- [ ] 切换显示模式不需要重连 WebSocket。
- [ ] 切换 Mixamo 角色后重置 IK，不复用上一模型的 bind pose。
- [ ] 新增角色只需 GLB 和注册信息，不修改后端协议。

## 14. 当前范围与后续工作

当前仓库已生成可交付的初版：

- `frontend/mixamo-avatar-delivery/public/avatars/character-a.glb`
- `frontend/mixamo-avatar-delivery/src/` 下的 Three.js/TypeScript 模块
- `frontend/mixamo-avatar-delivery/fixtures/halpe26-pose-samples.json`
- `frontend/mixamo-avatar-delivery/README.md`

GLB 已验证网格、纹理、65 根骨骼、层级、inverse bind 和四骨骼权重；原有
单人 TypeScript 模块已通过 strict 类型检查，并用实际 GLB 骨架和前 3 帧
fixture 完成了有限变换运行验证。

剩余工作是将该目录复制或合并进实际前端工程，再用真实 WebSocket 完成
左右、朝向、材质、缺失点和 UI 切换验收。

## 15. 多人扩展

多人模式不再读取顶层 `payload.joints`，而是读取
`payload.persons[].joints`，并以 `stream_id + track_id` 区分人物。交付包
新增：

- `StickmenWebSocketClient`：只接受 `avatar.stickmen.updated`。
- `MultiAvatarController`：为每个 `track_id` 建立独立
  `AvatarController`，并在 `stream_id` 变化时清理旧人物。
- `setDisplayLimit(1 | 2 | "all")`：由前端决定当前显示人数，后端仍可检测
  和发送 `--max-persons` 范围内的全部人物。

多人和单人可以使用相同 topic，但事件名不同；一个页面根据所选模式创建
对应客户端即可，不需要为了每个 `track_id` 建立独立 WebSocket。完整示例见
`frontend/mixamo-avatar-delivery/README.md`，消息字段见
`实时WebSocket接口说明.md` 第 9.5 节。

整个人暂时消失时，后端为了重识别可以继续保留 `track_id`，但会发送
`observed_in_frame=false` 且该人物的 26 个关节全部为 `null`。前端必须隐藏
该人物，不能保持最后一帧姿态；交付包中的 `MultiAvatarController`、
`StickmanRenderer` 和 `MixamoRetargeter` 已兼容这一语义。
