# Mixamo Avatar 前端交付包

这个目录可以复制到使用 Three.js 的前端项目中。原有单人路径继续消费
`avatar.stickman.updated`；可选多人路径消费新增的
`avatar.stickmen.updated`。两种消息中的每个人都仍是 Halpe26 的 26 个点，
Mixamo 显示不需要专用 WebSocket 事件。

完整设计和状态处理见：

- [`../../docs/前端Mixamo假人接入说明.md`](../../docs/前端Mixamo假人接入说明.md)
- [`../../docs/前端实时火柴人数据接入说明.md`](../../docs/前端实时火柴人数据接入说明.md)
- [`../../docs/实时WebSocket接口说明.md`](../../docs/实时WebSocket接口说明.md)

## 包含内容

```text
public/avatars/character-a.glb       自包含蒙皮 GLB
public/avatars/character-a.manifest.json
src/types.ts                         WebSocket 类型和运行时校验
src/halpe26.ts                       26 关节名称和连线
src/coordinates.ts                   Z-up 到 Three.js Y-up
src/stickmanWebSocket.ts             latest-only WebSocket 客户端
src/stickmenWebSocket.ts             多人 latest-only WebSocket 客户端
src/stickmanRenderer.ts              Three.js 火柴人
src/mixamoRetargeter.ts              Halpe26 到 Mixamo 重定向
src/avatarRegistry.ts                可选角色注册表
src/avatarController.ts              火柴人/假人二选一控制器
src/multiAvatarController.ts         按 track_id 管理多人和显示数量
fixtures/halpe26-pose-samples.json    无相机测试数据
```

`character-a.glb` 来自 `assets/models/mixamo/Ch09_nonPBR.fbx`，包含 17,071
个导出顶点、31,292 个三角形、65 根骨骼、四骨骼蒙皮权重和内嵌 PNG
纹理。模型大小约 10.74 MiB。

## 集成

前端项目需要 Three.js 0.160 或更高版本。将 `public/avatars` 复制到前端
静态资源目录，将 `src` 中模块复制到项目源码目录，然后按实际服务地址创建
控制器和 WebSocket 客户端：

```ts
import * as THREE from "three";
import {
  AvatarController,
  StickmanWebSocketClient
} from "./avatar/index.js";

const avatarController = new AvatarController(scene);

const poseClient = new StickmanWebSocketClient({
  url: "ws://192.168.30.132:8000/api/realtime/ws",
  topic: "avatar:stickman:FF6690772788",
  clientId: "avatar-ui",
  onPose: (payload) => avatarController.acceptPose(payload),
  onConnectionChange: (connected) => {
    console.info("avatar websocket", connected ? "connected" : "disconnected");
  }
});

poseClient.start();

// UI：只显示火柴人。
await avatarController.setDisplayMode("stickman");

// UI：切换到当前选中的 Mixamo 假人。
await avatarController.setDisplayMode("mixamo");

// UI：以后增加模型后切换角色，不改变 WebSocket。
await avatarController.selectAvatar("character-a");

function animate() {
  avatarController.tick();
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}
requestAnimationFrame(animate);
```

## 多人接入和显示数量

后端多人模式发送一个帧级 payload，`persons[]` 中每项包含独立
`track_id`、状态和 26 个 `joints`。前端仍订阅原 topic，但按新的 event
过滤：

```ts
import {
  MultiAvatarController,
  StickmenWebSocketClient
} from "./avatar/index.js";

const avatars = new MultiAvatarController(scene, { displayLimit: 2 });
const multiClient = new StickmenWebSocketClient({
  url: "ws://192.168.30.132:8000/api/realtime/ws",
  topic: "avatar:stickman:FF6690772788",
  clientId: "multi-avatar-ui",
  onPoses: (payload) => avatars.acceptPoses(payload)
});

multiClient.start();

// UI 可在运行中切换显示人数；已有 ID 会优先保留，减少人物闪烁。
avatars.setDisplayLimit(1);
avatars.setDisplayLimit(2);
avatars.setDisplayLimit("all");

function animateMulti() {
  avatars.tick();
  renderer.render(scene, camera);
  requestAnimationFrame(animateMulti);
}
requestAnimationFrame(animateMulti);
```

`stream_id` 在后端进程重启时变化，控制器会清理旧的 `track_id`，避免把新
进程的 ID 误认为旧人物。若继续运行旧单人命令，原
`StickmanWebSocketClient + AvatarController` 代码无需修改。

当某个 `persons[]` 项的 `observed_in_frame=false` 时，后端仍可能短时保留
该项和 `track_id` 用于重新关联，但会把 26 个关节全部置为 `null`。交付包
中的火柴人和 Mixamo 求解器会立即隐藏该人物；重新获得可靠三维观测后，
相同 `track_id` 可以直接恢复显示。

如果前端已经有 WebSocket 客户端和火柴人，可以只复制：

```text
public/avatars/
src/coordinates.ts
src/mixamoRetargeter.ts
src/avatarRegistry.ts
src/avatarController.ts
src/types.ts
```

并将现有客户端通过校验后的 `message.payload` 传给
`avatarController.acceptPose()`。

## 使用 fixtures 联调

`fixtures/halpe26-pose-samples.json` 包含站立、T pose、抬左臂和缺失关节
四帧数据。无需连接相机即可逐帧调用：

```ts
avatarController.acceptPose(fixture.frames[index], performance.now());
```

先检查以下动作：

1. T pose 左右方向正确。
2. 抬左臂时驱动人物自身的左臂。
3. 缺失手腕时手臂保持上一帧，不跳到原点。
4. 火柴人和假人模式切换时没有重复建立 WebSocket。

## 增加其他 Mixamo 模型

1. 将完整蒙皮 GLB 放入 `public/avatars/`。
2. 在 `src/avatarRegistry.ts` 的 `AVATARS` 中增加一项。
3. 保证模型至少包含 `mixamoRetargeter.ts` 中的 `REQUIRED_BONES`。
4. 骨骼带 `mixamorig:` 前缀时会自动按去前缀名称匹配。
5. 非标准骨骼名通过 `boneAliases` 提供“标准名到实际名”的映射。

不要为新模型增加新的 WebSocket topic 或实时消息。

## 重新导出当前 GLB

在 humanpose 项目根目录运行：

```bash
conda activate rgbd-avatar
PYTHONPATH=src python scripts/export_mixamo_glb.py --overwrite
```

导出器位于 `src/rgbd_avatar/avatar/mixamo_gltf.py`，不依赖 Blender 或
Assimp。重新导出后需要同步更新 `character-a.manifest.json` 中的文件哈希和
大小。

## 当前限制

- 实时协议没有置信度和 `predicted` 标记，前端以 `joint !== null` 判断有效性。
- Halpe26 没有手指关键点，手和手指暂时保持 bind pose。
- 26 个位置不能唯一恢复全部骨骼扭转，重定向器使用人体参考平面、bind
  pose 和上一帧旋转补齐约束。
- TypeScript 重定向器与后端 Python IK 的输入不同，目标是稳定前端显示，
  不承诺与后端调试查看器逐矩阵完全一致。
