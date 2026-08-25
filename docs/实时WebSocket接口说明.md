# 实时 WebSocket 接口说明

## 1. 概述

这个接口用于浏览器前端、Python 后端或其他系统接入同一条实时通道。

- 接口默认不需要登录鉴权，适合测试阶段和内部联调
- 姿态、任务等业务消息使用 JSON 文本帧；RGB 骨架预览使用独立连接上的 JPEG 二进制帧
- 服务端会按 topic 做订阅分发
- 当前实现是进程内 hub，同一个后端实例里的连接可以互通

## 2. 连接地址

```
ws://<host>/api/realtime/ws
```

可选 query 参数：

- `client_type`：客户端类型，默认 `generic`
- `client_id`：客户端标识，默认自动生成
- `topics`：初始订阅 topic，逗号分隔

示例：

```
ws://127.0.0.1:8000/api/realtime/ws?client_type=browser&client_id=scene-ui&topics=tasks,tasks:task-001
```

## 3. 消息信封

所有收发消息都遵循同一个 JSON 结构：

```json
{
  "type": "event",
  "event": "task.updated",
  "topic": "tasks:task-001",
  "topics": ["tasks", "tasks:task-001"],
  "source_type": "backend",
  "source_id": "task-service",
  "target_type": null,
  "target_id": null,
  "client_type": "browser",
  "client_id": "scene-ui",
  "connection_id": "b7c5c1f7a0bc4ce1a8c6b0d7d6b1b1c0",
  "message_id": "msg-001",
  "reply_to": "msg-000",
  "payload": {},
  "timestamp": "2026-08-06T05:15:00.000Z"
}
```

字段说明：

- `type`：消息类型，控制消息通常是 `hello`、`subscribe`、`publish`，业务消息通常是 `event`
- `event`：事件名，例如 `task.updated`
- `topic`：主 topic，单 topic 时会带上
- `topics`：当前消息覆盖的 topic 列表
- `source_type` / `source_id`：消息来源
- `target_type` / `target_id`：消息目标
- `client_type` / `client_id`：连接的客户端身份
- `connection_id`：服务端为这条连接生成的 ID
- `message_id`：消息 ID，建议客户端每条消息都带上
- `reply_to`：回复哪一条消息
- `payload`：业务数据
- `timestamp`：服务端时间

## 4. 控制消息

### 4.1 `hello` / `identify`

用于声明客户端身份，或在连接后更新身份信息。

请求：

```json
{
  "type": "hello",
  "message_id": "hello-001",
  "client_type": "python",
  "client_id": "worker-a",
  "topics": ["tasks"]
}
```

返回：

```json
{
  "type": "system.ack",
  "event": "hello",
  "reply_to": "hello-001",
  "payload": {
    "connection_id": "...",
    "client_id": "worker-a",
    "client_type": "python",
    "topics": ["tasks"]
  }
}
```

### 4.2 `subscribe`

订阅一个或多个 topic。

请求：

```json
{
  "type": "subscribe",
  "message_id": "sub-001",
  "topics": ["tasks:task-001", "scene:demo"]
}
```

返回：

```json
{
  "type": "system.ack",
  "event": "subscribe",
  "reply_to": "sub-001",
  "payload": {
    "subscribed": ["tasks", "tasks:task-001", "scene:demo"]
  }
}
```

### 4.3 `unsubscribe`

取消订阅一个或多个 topic。

### 4.4 `publish`

客户端向服务端发布业务消息。服务端会把消息转发到对应 topic 的订阅者。

请求：

```json
{
  "type": "publish",
  "message_id": "pub-001",
  "event": "task.updated",
  "topics": ["tasks:task-001"],
  "payload": {
    "task_id": "task-001",
    "status": "running"
  }
}
```

返回：

```json
{
  "type": "system.ack",
  "event": "publish",
  "reply_to": "pub-001",
  "payload": {
    "topics": ["tasks:task-001"],
    "published_to": 1
  }
}
```

### 4.5 `ping`

用于心跳探测。

请求：

```json
{
  "type": "ping",
  "message_id": "ping-001"
}
```

返回：

```json
{
  "type": "system.pong",
  "event": "pong",
  "reply_to": "ping-001"
}
```

## 5. 服务端事件

当前后端已经接入任务事件广播：

- `task.created`
- `task.updated`

任务事件会同时发往两个 topic：

- `tasks`
- `tasks:{task_id}`

这样既可以订阅全局任务流，也可以只盯某一个任务。

任务事件示例：

```json
{
  "type": "event",
  "event": "task.updated",
  "topics": ["tasks", "tasks:task-001"],
  "source_type": "backend",
  "source_id": "task-service",
  "payload": {
    "task": {
      "task_id": "task-001",
      "task_name": "场景训练-demo",
      "status": "running",
      "result": null,
      "error": null,
      "created_by": 1,
      "created_at": "2026-08-06T05:15:00Z",
      "updated_at": "2026-08-06T05:16:00Z"
    },
    "changes": {
      "status": "running"
    }
  }
}
```

## 6. Topic 约定

建议统一用小写和冒号分层：

- `tasks`：任务总线
- `tasks:{task_id}`：单个任务
- `scene:{scene_id}`：场景相关消息
- `user:{user_id}`：用户相关消息

这样前端、Python worker、其他系统都能按同一套规则接入。

## 7. 示例

### 浏览器

```js
const ws = new WebSocket("ws://127.0.0.1:8000/api/realtime/ws?client_type=browser&client_id=scene-ui&topics=tasks");

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  console.log(msg);
};

ws.addEventListener("open", () => {
  ws.send(JSON.stringify({
    type: "subscribe",
    message_id: crypto.randomUUID(),
    topics: ["tasks:task-001"]
  }));
});
```

### Python

```python
import asyncio
import json
import websockets

async def main():
    uri = "ws://127.0.0.1:8000/api/realtime/ws?client_type=python&client_id=worker-a&topics=tasks"
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({
            "type": "ping",
            "message_id": "ping-001",
        }))
        print(await ws.recv())

asyncio.run(main())
```

## 8. 注意事项

- 当前实现是单进程 hub，同一台后端实例内可实时互通
- 如果后续要多 worker、多实例广播，需要把 broker 换成 Redis、Kafka 或类似的共享消息层
- 客户端最好给每条消息带 `message_id`，方便排查和做 ack 关联
- 文本帧里请直接发 JSON，不要发表单格式

## 9. RGB-D 实时火柴人接入约定

前端字段、Halpe26 关节顺序、骨骼连接和 Three.js 示例详见
[`前端实时火柴人数据接入说明.md`](前端实时火柴人数据接入说明.md)。

相机/GPU 电脑作为客户端主动连接本接口；FastAPI 继续作为通用 Hub，
不需要新增专用 ingest 路由。该客户端只发布最终三维关节，不发送 RGB、
深度图、二维骨架或火柴人网格。

### 9.1 连接身份

```text
client_type = python
client_id   = rgbd-avatar-FF6690772788
```

当前部署地址：

```text
ws://192.168.30.132:8000/api/realtime/ws?client_type=python&client_id=rgbd-avatar-FF6690772788
```

连接建立后，相机客户端发送 `hello`，但不订阅自己的输出 topic：

```json
{
  "type": "hello",
  "message_id": "hello-<session-id>",
  "client_type": "python",
  "client_id": "rgbd-avatar-FF6690772788",
  "topics": []
}
```

### 9.2 事件与 topic

```text
event = avatar.stickman.updated
topic = avatar:stickman:FF6690772788
```

前端订阅：

```text
topics=avatar:stickman:FF6690772788
```

### 9.3 相机发布消息

```json
{
  "type": "publish",
  "message_id": "pose-<session-id>-310",
  "event": "avatar.stickman.updated",
  "topics": ["avatar:stickman:FF6690772788"],
  "payload": {
    "schema_version": 1,
    "keypoint_format": "halpe26",
    "coordinate_system": {
      "name": "application",
      "handedness": "right",
      "unit": "meter",
      "up_axis": "+z",
      "ground_z_m": 0.0
    },
    "source_id": "FF6690772788",
    "frame_number": 310,
    "timestamp_ms": 1785927947534,
    "status": "ok",
    "joints": [
      [0.12, 0.35, 1.67],
      [0.10, 0.34, 1.71],
      null
    ]
  }
}
```

`joints` 固定为 26 项并遵循 Halpe26 顺序；可用关节是米制
`[x,y,z]`，不可用关节必须为 `null`，不能发送 `NaN`。坐标已经应用
相机安装外参，应用地面是 `z=0`。

FastAPI Hub 转发给前端时，最外层会变为 `type=event` 信封；前端读取
`message.payload.joints`，而不是把整条 `publish` 请求当作业务数据。

### 9.4 实时策略

- 相机端待发送队列长度固定为 1；网络变慢时覆盖未发送的旧姿态。
- FastAPI 断开时相机和 GPU 推理继续运行，客户端指数退避重连。
- 每次重连后重新发送 `hello` 和最新一帧。
- 相机端消费 `system.ack`，用于统计发布是否被 Hub 接收。
- 当前 Hub 为单进程内存实现，FastAPI 暂时使用单 worker；多 worker 时
  需要 Redis/Kafka 等共享 broker。

2026-08-06 已用相机实时产生的 Halpe26 数据验证本地址：发布 5 帧，
`sent_frame_count=5`、`publish_ack_count=5`，连接和发布均无错误。

### 9.5 可选多人骨架事件

原单人事件保持不变。`view_live_multi_person.py --publish-stickmen` 在相同
topic 上发布新增事件：

```text
event = avatar.stickmen.updated
topic = avatar:stickman:FF6690772788
```

发布信封的 payload 结构如下；每个 `joints` 都固定为 Halpe26 的 26 项：

```json
{
  "schema_version": 1,
  "keypoint_format": "halpe26",
  "coordinate_system": {
    "name": "application",
    "handedness": "right",
    "unit": "meter",
    "up_axis": "+z",
    "ground_z_m": 0.0
  },
  "source_id": "FF6690772788",
  "stream_id": "a1b2c3d4e5f6",
  "frame_number": 310,
  "timestamp_ms": 1785927947534,
  "status": "ok",
  "identity_method": "shadow",
  "identity_fallback": false,
  "detected_person_count": 2,
  "published_person_count": 2,
  "persons": [
    {
      "track_id": 1,
      "status": "ok",
      "observed_in_frame": true,
      "joints": [[0.12, 0.35, 1.67], null]
    },
    {
      "track_id": 2,
      "status": "temporarily_missing",
      "observed_in_frame": false,
      "joints": [null, null]
    }
  ]
}
```

- `track_id` 只在同一个 `stream_id` 内有意义；后端进程重启会产生新的
  `stream_id`。
- `identity_method` 是本帧实际采用的 `geometry` 或 `shadow`。
- 请求新算法但其发生异常时，`identity_method=geometry` 且
  `identity_fallback=true`。
- 前端按 `persons[].track_id` 维护独立假人；显示 1 人、2 人或全部只是前端
  选择，不需要改变后端检测数量和 WebSocket 接口。
- `observed_in_frame=false` 表示整个人本帧没有可靠三维观测。后端可暂时保留
  该 `track_id` 以便重新关联，但其 26 个 `joints` 全部发送为 `null`，前端
  应立即隐藏人物，不能继续显示上一帧姿态。
- 人物仍被可靠观测、仅个别关节短时缺失时，`observed_in_frame=true`；这些
  关节仍可由后端的局部时序预测承接，不受整人隐藏规则影响。

## 10. RGB 实拍 + Halpe26 骨架预览

RGB 诊断画面与三维姿态使用两条独立 WebSocket 连接，但都通过现有
`/api/realtime/ws` 路径接入。这样 Nano 上已有的 Nginx WebSocket 反代无须新增
端口或 location，JPEG 传输也不会进入 JSON topic hub。

生产者连接：

```text
ws://127.0.0.1:8000/api/realtime/ws?preview_role=producer&preview_source_id=FF6690772788
```

浏览器连接（同源部署时会自动使用 `ws://当前地址` 或 `wss://当前地址`）：

```text
ws://<host>/api/realtime/ws?preview_role=browser&preview_source_id=FF6690772788
```

约定如下：

- 生产者只发送完整 JPEG 二进制帧，不发送 JSON、Base64 或深度图。
- 后端按 `preview_source_id` 隔离相机来源。
- 每个浏览器只有一个待发送槽；新帧会覆盖尚未发出的旧帧，不形成延迟队列。
- 新连接会立即收到服务端保存的最新一帧。
- humanpose 默认以 5 FPS、JPEG quality 75、原图 0.75 倍尺寸发布；三维姿态仍逐帧发布。
- 帧上已经包含多人彩色 Halpe26 骨架、检测框、稳定 track ID 和诊断信息。
- 当前接口与原实时 hub 一样没有鉴权，只应在可信局域网使用。

humanpose 开关：

```bash
--publish-rgb-preview
--no-publish-rgb-preview
--rgb-preview-url ws://127.0.0.1:8000/api/realtime/ws
```

Nano 的 `humanpose-tensorrt.service` 已使用 `--publish-stickmen` 时，新版本会同步
启用配套 RGB 预览；需要节省带宽或 JPEG 编码开销时可显式追加
`--no-publish-rgb-preview`。
