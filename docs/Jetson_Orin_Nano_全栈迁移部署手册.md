# Jetson Orin Nano：相机 + humanpose + GaussFlow 前后端全栈迁移部署手册

> 文档版本：2026-08-14（已加入本次 Nano 实机迁移复盘）  
> 面向设备：Jetson Orin Nano（优先 8 GB、NVMe、主动散热）  
> 当前源主机：humanpose=`/home/fr1511b/program/workspace/humanpose`，平台=`/home/fr1511b/program/3DGSAlgPlatform`  
> 已验证 Nano：用户=`nvidia`，项目根=`/home/nvidia/program`，JetPack 6.2.2 / L4T R36.5  
> 目标：即使 Nano 上没有 Codex，也能按本手册逐项执行、验证、排错和回滚。

> **阅读顺序说明：**第 0～26 章保留了最初的通用全栈方案，其中部分
> `/home/jetson/gaussflow`、JetPack 6.1、MySQL/RustFS 示例没有在本次
> Nano 上采用。**第 27 章是 2026-08-14 已实际跑通的权威记录；发生
> 路径、服务名、推理入口或后端入口冲突时，以第 27 章为准。**

---

## 0. 先读这一页：最终部署形态和关键结论

### 0.1 推荐的第一阶段架构

```text
蓝芯 S10 RGB-D 相机（192.168.1.125）
                 │ 独立有线相机网段 192.168.1.0/24
                 ▼
┌──────────────── Jetson Orin Nano ────────────────┐
│ humanpose（TensorRT 独立入口）                    │
│   RTMDet-M FP16 + RTMPose-M Halpe26 FP16         │
│   geometry + depth_connected，最多2人             │
│   └─ 发布 ws://127.0.0.1:8000/api/realtime/ws    │
│                                                  │
│ FastAPI realtime-only（127.0.0.1:8000，单worker） │
│   └─ 进程内多人 WebSocket Hub                    │
│                                                  │
│ Nginx（监听局域网 80）                            │
│   ├─ React 静态 dist                             │
│   ├─ /api/ → FastAPI                             │
│   └─ /api/realtime/ws → FastAPI WebSocket        │
└───────────────────────┬──────────────────────────┘
                        │ Wi-Fi/第二网口：管理网
                        ▼
                 局域网电脑浏览器

本次边缘实时模式不加载：MySQL、RustFS、3DGS训练任务
```

这样拆分的理由：

- 相机、推理、实时 Hub 和网页入口已经全部在 Nano，主实时链路不绕远路。
- React 在普通电脑浏览器里渲染，避免 Nano 本地浏览器的 WebGL 与 CUDA 推理争用统一内存。
- Nano 使用 `app.realtime_app:app`，因此实时模式不依赖 MySQL、RustFS
  和训练环境；完整平台业务后端仍可作为后续独立阶段处理。
- RTMDet/RTMPose 已转为 Nano 本机 TensorRT FP16 Engine；原 PyTorch
  入口仍保留用于回退和对照。

### 0.2 五条硬规则

1. **不要把现有 Conda 环境或 `.venv` 复制到 Nano。** 当前 humanpose 环境是 x86_64、PyTorch 2.1、CUDA 11.8；Nano 是 aarch64，CUDA 随 JetPack。
2. **不要只在 Nano 上 `git clone`。** 当前平台领先远端且含未提交的多人/Mixamo代码；`.pth` 权重和模型缓存也被 Git 忽略。
3. **FastAPI 必须只有一个 Uvicorn worker。** 当前 WebSocket Hub 在进程内，多 worker 会让发布端和浏览器落到不同进程。
4. **相机只能由一个进程独占。** 切到 Nano 前，旧主机的 Viewer 和 humanpose 必须完全退出。
5. **严格逐层验收。** PyTorch CUDA 没通过，就不要编 MMCV；相机 30 帧没通过，就不要加载姿态模型；本地姿态没通过，就不要接网页。

### 0.3 本次不承诺的范围

- 不把 3DGS 训练作为 Nano 首轮验收项。当前训练代码仍依赖 `conda run -n gaussian`，配置里的训练脚本路径也不完整。
- 原生 PyTorch + OpenMMLab仅作为正确性基线；生产入口已经改为独立
  TensorRT FP16链路。最终长时间FPS仍以第27章的实机验收方法为准。
- 不用 `RTMO tiny` 替换当前模型作为第一版。它输出 COCO17，Halpe26 的 20–25 足趾/脚跟点缺失，不适合当前 Mixamo 脚部映射。
- 不在第一天启用 `shadow`、`hybrid`、Open3D GUI 或四人推理。先用已验证的 `geometry + depth_connected + 1→2人`。

---

## 1. 本手册的约定

第0～26章的原始示例曾假定 `jetson/gaussflow` 目录。当前实机应使用
下面这一组值；复制旧章节命令时必须替换路径，或者直接执行第27章：

| 项目 | 示例值 | 你要做的事 |
|---|---:|---|
| Nano 用户名 | `nvidia` | systemd `User=`/`Group=`均使用该值 |
| Nano 当前管理网 IP | `192.168.8.119` | 这是DHCP现值；日常优先用 `nvidia-desktop.local` |
| Nano 部署根目录 | `/home/nvidia/program` | 本次实机已经按此目录完成 |
| 相机 IP | `192.168.1.125` | 当前已确认，不要随意修改 |
| 相机设备 ID | `FF6690772788` | 当前已确认，不要随意改成长 ID |
| Nano 相机口 IP | `192.168.1.188/24` | 仅在旧主机不再使用该地址时采用 |
| 实机记录日期 | `20260814` | TensorRT Engine与该Nano环境绑定 |

在**当前 x86 主机**终端先定义：

```bash
export NANO_USER=nvidia
export NANO_MGMT_IP=192.168.8.119
export RELEASE_ID=20260814
```

检查后再继续：

```bash
printf 'user=%s\nip=%s\nrelease=%s\n' "$NANO_USER" "$NANO_MGMT_IP" "$RELEASE_ID"
```

文中命令标签含义：

- **[主机]**：在当前 x86 主机执行。
- **[Nano]**：SSH 登录 Nano 后执行。
- **[MySQL主机]**：在当前保存 MySQL 的主机执行。
- 命令成功后才勾选对应检查项；失败不要跳到下一阶段。

---

## 2. 完整验收关卡

| 关卡 | 验收内容 | 通过条件 |
|---|---|---|
| G0 | 旧主机基线和备份 | 原链路仍能运行，配置/模型/数据有备份 |
| G1 | Nano OS/硬件 | `aarch64`、JetPack/CUDA版本已记录、磁盘和供电正常 |
| G2 | 双网段 | 管理默认路由正常，Nano 能 ping 相机 |
| G3 | Jetson PyTorch | CUDA可用，GPU张量运算成功 |
| G4 | MMCV/OpenMMLab | `from mmcv.ops import nms` 成功，版本全部匹配 |
| G5 | 相机SDK | 30帧约15 FPS、816×612、frame ID mismatch不增长 |
| G6 | 单图姿态 | RTMDet和RTMPose均从本地权重加载并在CUDA运行 |
| G7 | 实时本地姿态 | 1人100帧，再2人100帧，无崩溃/明显错误 |
| G8 | FastAPI | `/api/health` 成功，单worker运行 |
| G9 | WebSocket | publisher sent/ack增长，浏览器收到多人事件 |
| G10 | Nginx/React | 页面刷新不404，火柴人和Mixamo均实时运动 |
| G11 | 自启动 | Nano整机重启后所有服务自动恢复 |
| G12 | 稳定性 | 连续运行、温度、内存、日志体积均可接受 |

如果某一关失败，只处理该层，不同时改网络、模型、坐标轴和滤波参数。

---

## 3. G0：迁移前在旧主机做基线和备份

### 3.1 保留旧链路

Nano 验收完成前：

- 不删除 `/home/fr1511b/program/workspace/humanpose`。
- 不删除 `/home/fr1511b/program/3DGSAlgPlatform`。
- 不删除相机 SDK 包。
- 不升级旧主机的 PyTorch/MMCV。
- 不清空旧数据库和 RustFS 对象。

### 3.2 记录源文件状态

**[主机]**：

```bash
mkdir -p /home/fr1511b/program/nano-migration-records/20260812-01

git -C /home/fr1511b/program/3DGSAlgPlatform status --short --branch \
  > /home/fr1511b/program/nano-migration-records/20260812-01/platform-git-status.txt

git -C /home/fr1511b/program/3DGSAlgPlatform log -1 --oneline \
  > /home/fr1511b/program/nano-migration-records/20260812-01/platform-git-head.txt
```

humanpose 当前目录不一定是独立 Git 仓库，因此不要以“Git是干净的”作为迁移依据。

### 3.3 核对关键资产

**[主机]**：

```bash
sha256sum \
  /home/fr1511b/program/workspace/humanpose/assets/models/rtmpose/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.pth \
  /home/fr1511b/program/workspace/humanpose/assets/models/cache/hub/checkpoints/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
  '/home/fr1511b/program/3DGSAlgPlatform/fronted/fronted-react/public/假人/mixamo-avatar-delivery/public/avatars/character-a.glb'
```

预期 SHA256：

```text
RTMPose:
4d3e73ddd31222b7b0db36caeda396af1d7630c3b5a60451bdfa99a79e8dbb90

RTMDet:
35b0c7406499e0d141dd6a0235db07c10d2bee8f891f8f4e353c16a009de30e8

character-a.glb:
45bb2e7d3471cd6033b248d80c37898dda2e26ea216425bcc1dcf99e118e72ce
```

若 SHA 不同，先确认文件是不是同名的新版；不要把未知文件当成传输损坏直接删除。

### 3.4 记录旧主机性能基线

在旧主机关闭前记录一段：

- 相机约 15 FPS。
- 典型 `total` 每帧耗时。
- 单人/双人时 GPU、CPU、内存。
- WebSocket `sent`、`ack` 是否一致。
- 当前实际使用 `depth_connected + geometry`。

建议把终端日志和一段屏幕录像存进：

```text
/home/fr1511b/program/nano-migration-records/20260812-01/
```

### 3.5 数据库与对象存储备份

仓库里的 `fronted/sql/gaussflow_full_20260803_133802.sql` 不是可靠的现网完整备份：它体积很小、表不全且没有业务数据。

如果要保留登录、场景、文件记录，必须从正在使用的 MySQL 再导出一次。例如：

```bash
mysqldump --single-transaction --routines --triggers \
  -u root -p gaussflow \
  > /安全备份目录/gaussflow-before-nano-20260812.sql
```

然后验证文件非空：

```bash
ls -lh /安全备份目录/gaussflow-before-nano-20260812.sql
head -n 20 /安全备份目录/gaussflow-before-nano-20260812.sql
```

RustFS 需要另外确认实际数据根目录或使用对象存储客户端导出。数据库记录和 RustFS 对象必须成对保留，否则数据库会指向不存在的文件。

**G0 通过检查：**

- [ ] 旧主机代码未删除。
- [ ] 平台工作树状态已保存。
- [ ] 三个关键资产 SHA 已核对。
- [ ] 当前运行性能和画面已记录。
- [ ] MySQL 有新备份。
- [ ] RustFS 数据位置已确认，或明确第一阶段不迁移。

---

## 4. G1：准备 Nano、存储和 JetPack

### 4.1 硬件建议

最低建议：

- Jetson Orin Nano 8 GB。
- 官方稳定电源；不要用功率不明的 USB-C 供电代替。
- 主动散热风扇。
- NVMe SSD 作为系统/工作目录；不建议在 microSD 上长期编译 MMCV 和写高频日志。
- 相机有线网口 + 管理用 Wi-Fi；或者相机网口 + 第二个 USB 网卡。
- 至少预留 20 GB 空闲空间，若使用容器建议 35 GB 以上。

### 4.2 JetPack 版本选择，不要盲目追新

截至 2026-08-12：

- 官方 [JetPack Archive](https://developer.nvidia.com/embedded/jetpack-archive) 显示 JetPack 7.2 已支持 Orin 系列。
- [JetPack 6.2.2](https://developer.nvidia.com/embedded/jetpack-sdk-622) 是 JetPack 6 的最新生产版，Jetson Linux 36.5、Ubuntu 22.04、CUDA 12.6。
- 但本项目的 OpenMMLab 组合是 `mmcv 2.1.0 / mmdet 3.2.0 / mmpose 1.3.2`，属于较旧栈；JetPack 越新，PyTorch/MMCV ABI 风险越大。

本手册的选择原则：

| Nano 当前状态 | 建议 |
|---|---|
| 已经稳定安装 JetPack 6.1 | 优先走本手册“原生环境路线”，不为追新升级 |
| 全新设备、重现当前项目优先 | 可选 JetPack 6.1；它是 Ubuntu 22.04/Python 3.10，且官方矩阵曾提供 24.09 PyTorch wheel |
| 已安装 JetPack 6.2/6.2.2 | 先保留；只能使用与 6.2 对应的 NVIDIA构建/容器，不能套用 6.1 wheel |
| JetPack 7.x | 不建议作为本项目第一次迁移目标；Python/CUDA跨度更大 |
| JetPack 5.x | 不建议；系统 Python 3.8 与本项目 `Python >=3.10` 冲突 |

**最重要：先看设备上已经是什么，不要立刻刷机。**

### 4.3 首次启动 JetPack 6.x 的 QSPI 注意事项

NVIDIA 的 [Orin Nano Getting Started Guide](https://developer.nvidia.com/embedded/learn/get-started-jetson-orin-nano-devkit) 明确说明：部分原厂设备的旧固件不兼容 JetPack 6.x，需要先用 JetPack 5.1.3 SD 卡流程更新固件/QSPI，再启动 JetPack 6 镜像。

因此：

- JetPack 6 SD 卡不启动时，不要立刻判定 SD 卡或 Nano 损坏。
- 刷 QSPI/bootloader 前备份数据、确保供电稳定。
- 不要只通过远程 SSH 做一次没有本地恢复手段的 bootloader 升级。
- 使用官方 SD 镜像或 SDK Manager 流程；不要套用普通 Ubuntu PC 的安装教程。
- JetPack 6.2.2 的官方 SD 流程是先用 6.2.1/36.4.4，再按官方说明升级到 6.2.2/36.5。

### 4.4 首次登录后记录环境

**[Nano]**：

```bash
uname -a
uname -m
cat /etc/nv_tegra_release
dpkg-query -W nvidia-jetpack 2>&1 || true
python3 --version
nvcc --version
lsblk
df -h
free -h
ip -br address
ip route
sudo nvpmodel -q --verbose
sudo jetson_clocks --show
```

把输出保存：

```bash
mkdir -p /home/jetson/gaussflow/diagnostics

{
  uname -a
  uname -m
  cat /etc/nv_tegra_release
  dpkg-query -W nvidia-jetpack 2>&1 || true
  python3 --version
  nvcc --version
  free -h
  df -h
} > /home/jetson/gaussflow/diagnostics/g1-platform.txt
```

`uname -m` 必须输出：

```text
aarch64
```

若 `nvcc` 不存在，先确认是否只装了 Jetson Linux runtime、没有完整 JetPack开发组件。MMCV CUDA ops源码编译需要 CUDA toolkit/nvcc。

### 4.5 基础系统配置

**[Nano]**：

```bash
sudo timedatectl set-timezone Asia/Shanghai
timedatectl

sudo apt update
sudo apt install -y \
  git rsync curl wget ca-certificates \
  build-essential cmake ninja-build pkg-config \
  python3-pip python3-venv python3-dev \
  libopenblas-dev libjpeg-dev libpng-dev zlib1g-dev \
  libglib2.0-0 libgl1 libgomp1 \
  nginx iputils-ping iputils-arping ethtool
```

不要执行 `do-release-upgrade`。Jetson 的 Ubuntu、驱动、CUDA、bootloader 是一个配套系统，不按普通 PC 升级大版本。

先查看常见设备组；把服务用户加入系统实际存在的组：

```bash
getent group video
getent group render || true

sudo usermod -aG video jetson

if getent group render >/dev/null; then
  sudo usermod -aG render jetson
fi
```

重新登录后检查：

```bash
id jetson
```

### 4.6 创建目录

**[Nano]**：

```bash
mkdir -p \
  /home/jetson/gaussflow/humanpose \
  /home/jetson/gaussflow/platform/backend-fastapi \
  /home/jetson/gaussflow/vendor \
  /home/jetson/gaussflow/venvs \
  /home/jetson/gaussflow/wheelhouse \
  /home/jetson/gaussflow/diagnostics \
  /home/jetson/gaussflow/releases
```

**G1 通过检查：**

- [ ] `uname -m` 是 `aarch64`。
- [ ] JetPack/L4T/CUDA/Python版本已保存。
- [ ] NVMe或目标盘空间足够。
- [ ] 供电和散热可靠。
- [ ] `nvcc --version` 正常。
- [ ] 基础编译包已安装。

---

## 5. G2：配置相机网与管理网

### 5.1 推荐网卡分工

```text
Nano Wi-Fi/第二网卡：192.168.8.x（管理、SSH、网页、访问旧主机数据库）
Nano 有线相机网卡：192.168.1.188/24（只连接相机，不设网关）
相机：192.168.1.125/24
```

相机口绝不能抢默认路由。默认路由应指向管理网的路由器。

### 5.2 找出连接名和设备名

**[Nano]**：

```bash
nmcli -t -f NAME,TYPE,DEVICE connection show --active
ip -br link
ip -br address
ip route
```

注意区分：

- `enP...`/`eth0` 是设备名。
- `Wired connection 1` 是 NetworkManager 连接名。
- `nmcli connection modify` 后面需要连接名，不是设备名。

下面假设：

```bash
export CAMERA_IFACE=enP1p1s0
export CAMERA_CONNECTION='Wired connection 1'
```

请先把两个值替换成 Nano 的实际输出。

### 5.3 检查 `192.168.1.188` 是否冲突

如果旧主机还连接着相机网，并且仍使用 `192.168.1.188`，Nano 不能复用这个地址。先关闭旧主机相机网口，或给 Nano 选一个未占用的 `192.168.1.x` 地址，例如 `.189`。

可做重复地址探测：

```bash
sudo arping -D -I "$CAMERA_IFACE" -c 3 192.168.1.188
```

若收到其他主机对该地址的回应，不要使用 `.188`。

### 5.4 设置静态相机地址

**[Nano]**：

```bash
sudo nmcli connection modify "$CAMERA_CONNECTION" \
  ipv4.method manual \
  ipv4.addresses 192.168.1.188/24 \
  ipv4.gateway '' \
  ipv4.dns '' \
  ipv4.never-default yes \
  ipv6.method disabled

sudo nmcli connection up "$CAMERA_CONNECTION"
```

检查：

```bash
ip -br address show "$CAMERA_IFACE"
ip route
ping -c 4 192.168.1.125
sudo ethtool "$CAMERA_IFACE" | grep -E 'Speed|Duplex|Link detected'
```

预期：

- 相机口有 `192.168.1.188/24`。
- `192.168.1.0/24` 走相机口。
- `default via ...` 仍走 Wi-Fi/管理网。
- 相机能 ping 通。
- 链路最好为 `1000Mb/s`、`Full`、`Link detected: yes`。

先保持 MTU 1500。只有相机、Nano、中间交换机全部确认支持时才考虑 jumbo frame。

### 5.5 防火墙

先看状态：

```bash
sudo ufw status verbose
```

如果 UFW 是 inactive，不需要为了相机专门启用它。如果 UFW 已启用，可只允许来自相机 IP、进入相机口的流量：

```bash
sudo ufw allow in on "$CAMERA_IFACE" proto udp from 192.168.1.125
sudo ufw allow in on "$CAMERA_IFACE" proto tcp from 192.168.1.125
sudo ufw reload
```

不要盲跑 SDK 里的 `set_firewall.sh`；当前厂商包中的脚本引用和命令存在不确定性。厂商资料涉及的端口包括 UDP 9700、9800、3956、3959、31900、32000、39560，以及 TCP 9900，但按相机 IP + 专用接口放行更不容易误开放其他网络。

**G2 通过检查：**

- [ ] 管理网 SSH 不会因插上相机而断开。
- [ ] 默认路由走管理网。
- [ ] 相机口地址没有冲突。
- [ ] `ping 192.168.1.125` 成功。
- [ ] 有线链路为千兆全双工。

---

## 6. 从当前工作目录复制代码、模型和 SDK

### 6.1 为什么不能只用 Git

当前状态有三个特殊点：

- `3DGSAlgPlatform` 工作树领先远端约 83 个提交，并有大量 modified/untracked 文件。
- 多人/Mixamo集成代码和 `public/假人/` 中有未跟踪文件。
- humanpose 的 `.pth` 和 `assets/models/cache` 被忽略，Git 不会带走权重。

所以首次迁移使用 `rsync` 当前实际工作目录；等 Nano 跑通后，再整理为可复现提交或发布包。

### 6.2 测试 SSH

**[主机]**：

```bash
ssh "$NANO_USER@$NANO_MGMT_IP" 'uname -m && hostname && df -h /home'
```

必须看到 `aarch64`，并确认连接的是目标 Nano。

### 6.3 复制 humanpose，包含模型，不包含输出和 x86 环境

**[主机]**：

```bash
rsync -aH --info=progress2 \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude 'outputs/' \
  --exclude 'logs/' \
  /home/fr1511b/program/workspace/humanpose/ \
  "$NANO_USER@$NANO_MGMT_IP:/home/jetson/gaussflow/humanpose/"
```

这里**没有排除** `assets/models/cache`，这是有意的。

### 6.4 复制蓝芯 SDK 包

**[主机]**：

```bash
rsync -aH --info=progress2 \
  '/home/fr1511b/下载/MRDVS-2.4.60.260126-ubuntu-sdk/MRDVS/' \
  "$NANO_USER@$NANO_MGMT_IP:/home/jetson/gaussflow/vendor/MRDVS/"
```

### 6.5 在 x86 主机上构建 React

静态 `dist/` 与 CPU 架构无关。推荐在当前 x86 主机构建，Nano 不需要 Node/npm，也不需要复制约 296 MiB 的 `node_modules`。

**[主机]**：

```bash
cd /home/fr1511b/program/3DGSAlgPlatform/fronted/fronted-react
npm ci
VITE_API_BASE_URL=/api npm run build
```

验证：

```bash
test -s dist/index.html
test -s 'dist/假人/mixamo-avatar-delivery/public/avatars/character-a.glb'
sha256sum 'dist/假人/mixamo-avatar-delivery/public/avatars/character-a.glb'
du -sh dist
```

GLB 预期 SHA256：

```text
45bb2e7d3471cd6033b248d80c37898dda2e26ea216425bcc1dcf99e118e72ce
```

如果构建失败，不要把旧 `dist` 当成新版本上传。先处理 TypeScript/Vite 构建错误。

### 6.6 复制 FastAPI 源码

**[主机]**：

```bash
rsync -aH --info=progress2 \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '.env' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude 'logs/' \
  /home/fr1511b/program/3DGSAlgPlatform/fronted/backend-fastapi/ \
  "$NANO_USER@$NANO_MGMT_IP:/home/jetson/gaussflow/platform/backend-fastapi/"
```

`.env` 被故意排除：它含数据库密码、RustFS密钥和 JWT，不应该随源码散播。

### 6.7 复制前端为带版本的发布目录

**[Nano]** 先准备目录：

```bash
mkdir -p /home/jetson/gaussflow/releases/frontend-20260812-01
```

**[主机]**：

```bash
rsync -aH --info=progress2 \
  /home/fr1511b/program/3DGSAlgPlatform/fronted/fronted-react/dist/ \
  "$NANO_USER@$NANO_MGMT_IP:/home/jetson/gaussflow/releases/frontend-20260812-01/"
```

### 6.8 Nano 端校验

**[Nano]**：

```bash
sha256sum \
  /home/jetson/gaussflow/humanpose/assets/models/rtmpose/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.pth \
  /home/jetson/gaussflow/humanpose/assets/models/cache/hub/checkpoints/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
  '/home/jetson/gaussflow/releases/frontend-20260812-01/假人/mixamo-avatar-delivery/public/avatars/character-a.glb'

du -sh \
  /home/jetson/gaussflow/humanpose \
  /home/jetson/gaussflow/vendor/MRDVS \
  /home/jetson/gaussflow/releases/frontend-20260812-01
```

三项 SHA 必须与 G0 相同。若不一致，重新传那个文件，不要继续。

---

## 7. 安装蓝芯相机 SDK（ARM64）

### 7.1 已确认的 ARM64 制品

厂商包内已有：

```text
SDK/lib/linux_aarch64/libLxCameraApi.so
SDK/lib/linux_aarch64/libLxDataProcess.so
Sample/python/lx_camera_py-1.3.3-py3-none-any.whl
```

两个 `.so` 是 aarch64 ELF；Python wheel 是 `py3-none-any`，可以在 ARM64 使用。

### 7.2 不推荐直接运行厂商 `install.sh`

当前 `install.sh` 会删除并重建 `/opt/MRDVS`、删除 `/opt/Lanxin-MRDVS`、修改 shell rc 和系统 socket 参数。为便于审计和回滚，本手册使用手工安装。

如果 Nano 上已经有 `/opt/MRDVS`，先查看并备份，不要直接覆盖：

```bash
sudo ls -la /opt/MRDVS 2>/dev/null || true

if test -d /opt/MRDVS && \
   test ! -e /opt/MRDVS.backup-before-gaussflow; then
  sudo cp -a /opt/MRDVS /opt/MRDVS.backup-before-gaussflow
fi
```

### 7.3 安装动态库和头文件

**[Nano]**：

```bash
sudo install -d -m 0755 /opt/MRDVS/lib /opt/MRDVS/include

sudo install -m 0755 \
  /home/jetson/gaussflow/vendor/MRDVS/SDK/lib/linux_aarch64/libLxCameraApi.so \
  /opt/MRDVS/lib/libLxCameraApi.so

sudo install -m 0755 \
  /home/jetson/gaussflow/vendor/MRDVS/SDK/lib/linux_aarch64/libLxDataProcess.so \
  /opt/MRDVS/lib/libLxDataProcess.so

sudo install -m 0644 \
  /home/jetson/gaussflow/vendor/MRDVS/SDK/include/*.h \
  /opt/MRDVS/include/
```

建立系统动态库搜索路径：

```bash
printf '%s\n' '/opt/MRDVS/lib' | \
  sudo tee /etc/ld.so.conf.d/mrdvs.conf
sudo ldconfig
```

### 7.4 设置网络接收/发送缓冲

```bash
printf '%s\n' \
  'net.core.rmem_max=10485760' \
  'net.core.wmem_max=10485760' | \
  sudo tee /etc/sysctl.d/90-lanxin-rgbd.conf

sudo sysctl --system
sysctl net.core.rmem_max net.core.wmem_max
```

预期两个值都至少为 `10485760`。

### 7.5 校验架构、依赖和 SHA

```bash
file /opt/MRDVS/lib/libLxCameraApi.so
file /opt/MRDVS/lib/libLxDataProcess.so
ldd /opt/MRDVS/lib/libLxCameraApi.so
ldd /opt/MRDVS/lib/libLxDataProcess.so

sha256sum \
  /opt/MRDVS/lib/libLxCameraApi.so \
  /opt/MRDVS/lib/libLxDataProcess.so \
  /home/jetson/gaussflow/vendor/MRDVS/Sample/python/lx_camera_py-1.3.3-py3-none-any.whl
```

预期：

```text
libLxCameraApi.so:
0da4cece44018324689b23141e1b22dac61d30d0333118a426a669f97cf9abb2

libLxDataProcess.so:
246e6f46d34ec81d9da72cbad7103177b28be2acefb4692a45744484a5f68b83

lx_camera_py wheel:
a24032b9d31ac4c6dd543993f539aef6374ecc1588bba6996c16c23e869c2369
```

`ldd` 不能出现 `not found`。架构必须包含 `ARM aarch64`，不能是 `x86-64`。

---

## 8. G3：创建 humanpose 环境并安装 Jetson PyTorch

### 8.1 当前 x86 环境只用于参考，不能照搬

当前旧主机实测组合：

```text
Python       3.10.20
PyTorch      2.1.0 + CUDA 11.8
torchvision  0.16.0
NumPy        1.26.4
SciPy        1.11.4
OpenCV       4.10.0
MMCV         2.1.0（x86_64 CUDA扩展）
MMEngine     0.10.7
MMDetection  3.2.0
MMPose       1.3.2
```

以下内容都不能复制到 Nano：

- Conda 环境目录。
- `site-packages`。
- `.venv`。
- `mmcv/_ext.cpython-310-x86_64-linux-gnu.so`。
- `pytorch-cuda=11.8`。

### 8.2 先根据 JetPack 选路线

运行：

```bash
cat /etc/nv_tegra_release
dpkg-query -W nvidia-jetpack 2>&1 || true
python3 --version
nvcc --version
```

然后只选一条：

#### 路线 A：JetPack 6.1，原生 venv（本手册完整覆盖）

这是迁移旧 OpenMMLab 项目的首选可复现基线：Ubuntu 22.04、Python 3.10、CUDA 12.6，NVIDIA 的 v61 目录提供 CPython 3.10 aarch64 PyTorch 2.5 wheel。

#### 路线 B：JetPack 6.2/6.2.2

> **当前实机例外：**下面是迁移前的保守通用建议。本次具体Nano后来已
> 用NVIDIA PyTorch 2.5 wheel完成CUDA、torchvision、MMCV和真实姿态
> 验收，并进一步切换到TensorRT生产入口。重建当前设备时不要继续猜
> 新组合，直接复用第27.9节记录的wheel、SHA和MMEngine补丁。

截至本文日期，NVIDIA [PyTorch/JetPack兼容表](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html) 对 JetPack 6.2列出的是 25.02–25.06 framework container（PyTorch 2.7/2.8），没有对应 standalone framework wheel。不要把 6.1 wheel 强装进 6.2环境，也不要直接从普通 PyPI 安装一个 CPU版 `torch`。

如果 Nano 已经是 6.2.x，有三种选择：

1. 保留 6.2.x，先用官方 NGC PyTorch容器做 `torch.cuda` 探针，再在容器内尝试编译 MMCV 2.1；**G4 没通过前不要继续全栈迁移**。
2. 把模型改成 ONNX/TensorRT 推理，绕开旧 MMCV运行时；这是后续更合理的生产方向，但超出本次“原样迁移”范围。
3. 若设备无重要数据、项目复现优先，可按官方归档流程刷 JetPack 6.1，再走路线 A。

官方容器 CUDA 探针示例（只验证，不代表 MMCV 已兼容）：

```bash
sudo docker run --rm --runtime nvidia \
  nvcr.io/nvidia/pytorch:25.06-py3 \
  python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

25.06 容器是 Python 3.12/PyTorch 2.8，和旧 MMPose/MMCV跨度较大。若选择路线 B，必须单独完成并保存一份通过测试的镜像，不能把路线 A 的 pip 命令混进宿主环境。

### 8.3 路线 A：创建独立 venv

humanpose 与 FastAPI 必须使用两个独立 venv。否则后端升级 NumPy/websockets 时，可能破坏 Jetson PyTorch栈。

**[Nano，JetPack 6.1]**：

```bash
python3 -m venv /home/jetson/gaussflow/venvs/humanpose
source /home/jetson/gaussflow/venvs/humanpose/bin/activate

python -m pip install --upgrade \
  pip==24.3.1 \
  setuptools==69.5.1 \
  wheel
```

确认当前 Python 来自 venv：

```bash
which python
python --version
python -c 'import platform; print(platform.machine())'
```

预期：

```text
/home/jetson/gaussflow/venvs/humanpose/bin/python
Python 3.10.x
aarch64
```

### 8.4 先安装 NumPy 和基础库

```bash
python -m pip install \
  numpy==1.26.4 \
  scipy==1.11.4 \
  Pillow==10.4.0 \
  PyYAML==6.0.2 \
  tqdm==4.67.1 \
  rich==13.9.4 \
  opencv-python-headless==4.10.0.84 \
  'websockets>=15,<18' \
  ninja psutil pytest
```

headless 服务不需要先装：

```text
open3d、viser、nerfview、splines、jaxtyping、smplx、SMPL模型、手部模型
```

这些属于本机 GUI、离线 SMPL 或浏览器开发链，不是实时 Halpe26发布的必要条件。

### 8.5 安装 cuSPARSELt

NVIDIA 的 [Jetson PyTorch安装指南](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html) 说明 24.06 及之后的 PyTorch需要 cuSPARSELt。

先检查：

```bash
ldconfig -p | grep -i cusparseLt || true
```

如果没有 `libcusparseLt.so.0`，按 NVIDIA 官方指南安装，并确保使用与当前 `nvcc --version` 对应的 CUDA版本。不要照抄官方页面里仅作为示例的 `CUDA_VERSION=12.1`。

可选的官方 Tegra 本地安装包路线：进入 [NVIDIA cuSPARSELt Downloads](https://developer.nvidia.com/cusparselt-downloads)，选择：

```text
Linux → aarch64/Jetson(Tegra) → Ubuntu 22.04 → 与 CUDA 12.x 对应版本
```

安装本地 repo `.deb` 后，终端会打印“复制 keyring”的准确命令；先执行那条命令，再：

```bash
sudo apt update
sudo apt install -y libcusparselt0 libcusparselt-dev
sudo ldconfig
ldconfig -p | grep -i cusparseLt
```

在 `libcusparseLt.so.0` 出现前不要安装/导入 PyTorch，否则常见报错就是：

```text
ImportError: libcusparseLt.so.0: cannot open shared object file
```

### 8.6 下载并安装与 JetPack 6.1 对应的 NVIDIA PyTorch wheel

官方 v61 目录：

```text
https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/
```

当前目录中的文件为：

```text
torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
```

**[Nano，venv已激活]**：

```bash
cd /home/jetson/gaussflow/wheelhouse

curl -fL \
  'https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl' \
  -o torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl

python -m pip install --no-cache-dir \
  /home/jetson/gaussflow/wheelhouse/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
```

不要执行：

```bash
pip install torch
```

这可能用普通 PyPI包覆盖 NVIDIA Jetson GPU构建。

### 8.7 构建匹配的 torchvision

PyTorch 2.5 对应 torchvision 0.20。为避免下载错误架构/ABI的 wheel，在 Nano 上从官方 torchvision源码构建：

```bash
cd /home/jetson/gaussflow/wheelhouse
git clone --depth 1 --branch v0.20.0 \
  https://github.com/pytorch/vision.git torchvision-0.20.0-src

cd /home/jetson/gaussflow/wheelhouse/torchvision-0.20.0-src
export BUILD_VERSION=0.20.0
export TORCH_CUDA_ARCH_LIST=8.7
export MAX_JOBS=2

python -m pip wheel . \
  --no-deps \
  --no-build-isolation \
  -w /home/jetson/gaussflow/wheelhouse/built

python -m pip install \
  /home/jetson/gaussflow/wheelhouse/built/torchvision-0.20.0-*.whl
```

如果 8 GB 内存构建时被杀死，把 `MAX_JOBS=2` 改成 `MAX_JOBS=1` 后重试。

### 8.8 G3 CUDA验收脚本

```bash
python - <<'PY'
import platform
import torch
import torchvision

print("machine:", platform.machine())
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("torch CUDA build:", torch.version.cuda)
print("cuDNN:", torch.backends.cudnn.version())
print("CUDA available:", torch.cuda.is_available())

assert platform.machine() == "aarch64"
assert torch.cuda.is_available()

print("GPU:", torch.cuda.get_device_name(0))
x = torch.randn((1024, 1024), device="cuda")
y = x @ x
torch.cuda.synchronize()
print("GPU tensor OK:", y.shape, float(y[0, 0]))

from torchvision.ops import nms
boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 9, 9]], dtype=torch.float32, device="cuda")
scores = torch.tensor([0.9, 0.8], dtype=torch.float32, device="cuda")
print("torchvision CUDA NMS:", nms(boxes, scores, 0.5))
PY
```

以下条件缺一不可：

- `machine: aarch64`
- `CUDA available: True`
- 能打印 Orin GPU名称。
- GPU矩阵运算成功。
- torchvision NMS成功。

保存当前状态：

```bash
python -m torch.utils.collect_env \
  > /home/jetson/gaussflow/diagnostics/g3-torch-environment.txt

python -m pip freeze \
  > /home/jetson/gaussflow/diagnostics/g3-humanpose-before-mmcv-freeze.txt
```

**G3 通过前禁止继续。**

---

## 9. G4：在 Nano 上编译 MMCV CUDA ops 并安装 OpenMMLab

### 9.1 项目要求的精确组合

```text
mmengine==0.10.7
mmcv==2.1.0
mmdet==3.2.0
mmpose==1.3.2
```

MMDetection 3.2.0要求 MMCV `>=2.0.0,<2.2.0`；MMPose 1.3.2要求 MMDetection `<3.3.0`。不要只升级其中一个包。

OpenMMLab 通常没有适配“Jetson aarch64 + 当前 PyTorch + CUDA”的 MMCV预编译 wheel，因此要按 [MMCV官方源码编译指南](https://mmcv.readthedocs.io/en/2.x/get_started/build.html) 本机编译。

### 9.2 编译前门禁

**[Nano，humanpose venv已激活]**：

```bash
which python
which nvcc
nvcc --version
gcc --version

python - <<'PY'
import torch
from torch.utils.cpp_extension import CUDA_HOME
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA_HOME:", CUDA_HOME)
print("CUDA available:", torch.cuda.is_available())
PY
```

预期 `CUDA_HOME` 通常为 `/usr/local/cuda`，且 CUDA available为 True。

先安装 MMEngine 和构建辅助依赖：

```bash
python -m pip install \
  mmengine==0.10.7 \
  ninja psutil packaging addict yapf==0.40.2
```

### 9.3 内存不足时的临时 swap（只放 NVMe）

先尝试 `MAX_JOBS=1`。只有编译确实因 OOM 被杀死，并且 `/home` 位于 NVMe 时，才创建临时 swap：

```bash
findmnt /home
free -h
test ! -e /home/jetson/gaussflow/mmcv-build.swap

sudo fallocate -l 8G /home/jetson/gaussflow/mmcv-build.swap
sudo chmod 600 /home/jetson/gaussflow/mmcv-build.swap
sudo mkswap /home/jetson/gaussflow/mmcv-build.swap
sudo swapon /home/jetson/gaussflow/mmcv-build.swap
free -h
```

不要在 microSD 上长期高频换页。编译完成并验证后关闭：

```bash
sudo swapoff /home/jetson/gaussflow/mmcv-build.swap
```

文件可暂时保留，确认稳定后再人工归档或删除。

### 9.4 拉取固定版本并构建 wheel

```bash
cd /home/jetson/gaussflow/wheelhouse
git clone --depth 1 --branch v2.1.0 \
  https://github.com/open-mmlab/mmcv.git mmcv-2.1.0-src

cd /home/jetson/gaussflow/wheelhouse/mmcv-2.1.0-src

export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=8.7
export MMCV_WITH_OPS=1
export FORCE_CUDA=1
export MAX_JOBS=1
set -o pipefail

python -m pip wheel . \
  --no-deps \
  --no-build-isolation \
  -v \
  -w /home/jetson/gaussflow/wheelhouse/built \
  2>&1 | tee /home/jetson/gaussflow/diagnostics/g4-mmcv-build.log
```

编译可能需要较长时间。SSH断开风险高时先进入 `tmux`：

```bash
sudo apt install -y tmux
tmux new -s mmcv-build
```

重新连接后：

```bash
tmux attach -t mmcv-build
```

### 9.5 安装编好的 MMCV wheel

先查看实际文件名：

```bash
ls -lh /home/jetson/gaussflow/wheelhouse/built/mmcv-2.1.0-*.whl
```

再安装：

```bash
python -m pip install \
  /home/jetson/gaussflow/wheelhouse/built/mmcv-2.1.0-*.whl
```

务必保存这个 wheel；它是为当前 Nano/Python/PyTorch/CUDA组合编出来的可回滚制品。

### 9.6 安装 MMDetection 和 MMPose

```bash
python -m pip install \
  mmdet==3.2.0 \
  mmpose==1.3.2
```

安装后立刻检查 pip 是否偷偷替换了 Torch/NumPy：

```bash
python -m pip check
python -m pip show torch torchvision numpy mmcv mmengine mmdet mmpose
```

### 9.7 安装 humanpose 本身和相机 Python wheel

```bash
cd /home/jetson/gaussflow/humanpose

python -m pip install --no-deps -e .

python -m pip install --no-deps \
  /home/jetson/gaussflow/vendor/MRDVS/Sample/python/lx_camera_py-1.3.3-py3-none-any.whl
```

### 9.8 G4 必须验证真实 CUDA ops

只运行 `import mmcv` 不够。执行：

```bash
python - <<'PY'
import platform
import torch
import mmcv
import mmengine
import mmdet
import mmpose
from mmcv.ops import nms

print("machine:", platform.machine())
print("torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("mmcv:", mmcv.__version__)
print("mmengine:", mmengine.__version__)
print("mmdet:", mmdet.__version__)
print("mmpose:", mmpose.__version__)

boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 9, 9]], dtype=torch.float32, device="cuda")
scores = torch.tensor([0.9, 0.8], dtype=torch.float32, device="cuda")
kept, indices = nms(boxes, scores, 0.5)
torch.cuda.synchronize()
print("MMCV CUDA NMS OK:", kept, indices)
PY
```

查找扩展并确认架构：

```bash
MMCV_EXT=$(find /home/jetson/gaussflow/venvs/humanpose/lib \
  -path '*/mmcv/_ext*.so' -print -quit)
printf '%s\n' "$MMCV_EXT"
file "$MMCV_EXT"
ldd "$MMCV_EXT" | grep 'not found' || true
```

预期扩展为 ARM aarch64，且没有 `not found`。

冻结最终版本：

```bash
python -m pip freeze \
  > /home/jetson/gaussflow/diagnostics/g4-humanpose-working-freeze.txt

sha256sum /home/jetson/gaussflow/wheelhouse/built/mmcv-2.1.0-*.whl \
  > /home/jetson/gaussflow/diagnostics/g4-mmcv-wheel.sha256
```

### 9.9 G4 失败时怎么判断

| 错误 | 含义和处理 |
|---|---|
| `Killed` | 编译 OOM；`MAX_JOBS=1`，确认 NVMe swap |
| `No module named mmcv._ext` | 装成 mmcv-lite、wheel未包含ops或编译失败 |
| `undefined symbol` | MMCV 与当前 Torch ABI不一致；不要复制x86 `.so`，清理该次构建后用当前Torch重编 |
| `CUDA_HOME is None` | CUDA toolkit/nvcc未完整安装或环境路径错误 |
| `no kernel image...` | 没包含 Orin 的 `sm_87`；确认 `TORCH_CUDA_ARCH_LIST=8.7` |
| torchvision缺少 `nms` | torchvision与NVIDIA torch不配套，重编 torchvision |
| pip准备下载另一个 torch | 立即取消；使用现有NVIDIA torch，避免依赖解析覆盖 |
| `GLIBCXX... not found` | 编译器/运行库混用；确认全部在Nano同一系统构建，不要混入Conda x86制品 |

G4 如果无法通过，不要靠连续升级/降级单个 OpenMMLab包碰运气。保留 `g4-mmcv-build.log`，选择整组兼容版本或转 TensorRT。

**G3/G4 通过检查：**

- [ ] PyTorch CUDA 为 True，GPU矩阵运算成功。
- [ ] torchvision CUDA NMS成功。
- [ ] MMCV 2.1.0 wheel在Nano本机生成并保存。
- [ ] MMCV CUDA NMS成功。
- [ ] mmdet/mmpose版本正确。
- [ ] 最终 `pip freeze` 和 wheel SHA 已保存。

---

## 10. 创建 Nano 专用 humanpose 配置

不要直接覆盖主机配置。创建 Nano 副本：

```bash
cd /home/jetson/gaussflow/humanpose
cp configs/live.yaml configs/live.nano.yaml
nano configs/live.nano.yaml
```

至少修改下面两处：

```yaml
live:
  source:
    type: "sdk"
    directory: "/home/jetson/gaussflow/camera-data/FF6690772788"

    sdk:
      python_wheel: "/home/jetson/gaussflow/vendor/MRDVS/Sample/python/lx_camera_py-1.3.3-py3-none-any.whl"
      library_path: "/opt/MRDVS/lib/libLxCameraApi.so"
```

并逐项确认以下值仍在：

```yaml
open_mode: "id"
open_param: "FF6690772788"
align_depth_to_rgb: true
sync_frame: true
require_matching_frame_ids: true
enable_amplitude: false
disable_builtin_algorithm: true
rgb_order: "rgb"
timestamp_source: "sensor_us"
```

多人部分保持：

```yaml
multi_person:
  recovery_method: "depth_connected"
  identity_tracker: "geometry"
```

相机到应用坐标系的现有标定也必须原样保留：

```yaml
application_extrinsics:
  transform: "application_from_camera"
  euler_order: "ZYX"
  roll_deg: 90.38
  pitch_deg: -179.83
  yaw_deg: 89.95
  translation_m: [0.0, 0.0, 0.78795]
```

不要在迁移过程中重新交换坐标轴、左右关节或足趾/脚跟索引。先证明相同输入在主机和 Nano 输出一致，再单独做新的标定。

全部服务部署在 Nano 时，发布地址保持回环地址：

```yaml
websocket_publish:
  enabled: true
  url: "ws://127.0.0.1:8000/api/realtime/ws"
  client_id: "rgbd-avatar-FF6690772788"
  source_id: "FF6690772788"
  topic: "avatar:stickman:FF6690772788"
```

多人事件应为：

```yaml
multi_person:
  websocket_publish:
    enabled: true
    event: "avatar.stickmen.updated"
```

扫描残留的旧绝对路径：

```bash
rg -n '/home/fr1511b' configs/live.nano.yaml configs/pose.yaml configs/camera.yaml
```

`configs/pose.yaml` 的模型路径本来就是项目相对路径，不要改成旧主机绝对路径。

---

## 11. G5：只验证相机，不加载姿态模型

### 11.1 先释放旧主机的相机

**[主机]**：

```bash
pgrep -af 'LxCamera|view_live_multi_person|inspect_lx_camera|LxCameraViewer' || true
```

正常关闭列出的 Viewer/脚本。不要同时从旧主机和 Nano 打开相机。

**[Nano]** 同样检查：

```bash
pgrep -af 'view_live_multi_person|inspect_lx_camera' || true
```

### 11.2 验证 Python SDK导入

```bash
source /home/jetson/gaussflow/venvs/humanpose/bin/activate
export LD_LIBRARY_PATH=/opt/MRDVS/lib:/usr/local/cuda/lib64

python - <<'PY'
import LxCameraSDK
print("LxCameraSDK import: OK")
print(LxCameraSDK)
PY
```

### 11.3 采集30帧

```bash
cd /home/jetson/gaussflow/humanpose
set -o pipefail

env \
  PYTHONPATH=src \
  LD_LIBRARY_PATH=/opt/MRDVS/lib:/usr/local/cuda/lib64 \
  python scripts/inspect_lx_camera.py \
    --live-config configs/live.nano.yaml \
    --camera-config configs/camera.yaml \
    --frames 30 \
  2>&1 | tee /home/jetson/gaussflow/diagnostics/g5-camera-30frames.log
```

历史基线：

```text
RGB/Depth分辨率：816×612
相机频率：约15 FPS
帧间隔：约66.7 ms
有效深度比例：约82.5%–84.1%（随场景变化，不是硬阈值）
frame_id_mismatch：0
```

关注日志末尾 `SDK source statistics`：

- RGB/Depth frame ID mismatch不能持续增长。
- RGB与深度时间戳差不能持续异常扩大。
- 深度有效率不能接近0。
- 30帧期间不能反复超时。

### 11.4 相机常见错误

| 现象/错误 | 优先检查 |
|---|---|
| `LX_E_DEVICE_NOT_FOUND` | 供电、线缆、192.168.1.x地址、千兆链路、防火墙、二层广播 |
| `LX_E_CTRL_PERMISS_ERROR` | Viewer、旧主机或另一个服务仍占用相机 |
| 能 ping，按 ID 找不到 | 二层枚举/防火墙；复制一份诊断配置临时用 IP打开 |
| 动态库找不到 | `ldconfig`、配置的library_path、`LD_LIBRARY_PATH` |
| frame mismatch增长 | 保持门控开启；查丢包、socket buffer、网线/交换机，不要先调骨架 |
| FPS明显低于15但CPU空闲 | 网卡协商、丢包、SDK同步或相机流配置 |

仅用于诊断的 IP打开方式：

```bash
cp configs/live.nano.yaml configs/live.nano-ip-diagnostic.yaml
nano configs/live.nano-ip-diagnostic.yaml
```

改成：

```yaml
open_mode: "ip"
open_param: "192.168.1.125"
```

诊断完成后生产配置仍优先用设备 ID。

**G5 通过检查：**

- [ ] SDK Python包导入成功。
- [ ] 30帧采集完成。
- [ ] 分辨率816×612、频率接近15 FPS。
- [ ] frame-ID mismatch不增长。
- [ ] 深度有效率合理。

---

## 12. G6：用单张图片验证 RTMDet + RTMPose

### 12.1 复制已知测试图片

**[Nano]**：

```bash
mkdir -p /home/jetson/gaussflow/testdata
```

**[主机]**：

```bash
rsync -a --info=progress2 \
  /home/fr1511b/program/workspace/data/1/20260730_145911656_r.png \
  "$NANO_USER@$NANO_MGMT_IP:/home/jetson/gaussflow/testdata/"
```

### 12.2 明确使用 CUDA 和多人检测器

**[Nano]**：

```bash
source /home/jetson/gaussflow/venvs/humanpose/bin/activate
cd /home/jetson/gaussflow/humanpose
set -o pipefail

env \
  PYTHONPATH=src \
  XDG_CACHE_HOME=/home/jetson/gaussflow/humanpose/assets/models/cache \
  python scripts/test_rtmpose_single.py \
    --image /home/jetson/gaussflow/testdata/20260730_145911656_r.png \
    --device cuda:0 \
    --detector auto \
    --output-dir outputs/nano-single-smoke \
  2>&1 | tee /home/jetson/gaussflow/diagnostics/g6-single-image.log
```

必须在日志中确认：

- resolved device是 `cuda:0`，不是 CPU。
- RTMDet checkpoint从本地 cache读取。
- RTMPose checkpoint从本地 assets读取。
- 输出 JSON 和 overlay PNG生成。
- 有 26个 Halpe关键点定义。

查看输出：

```bash
ls -lh outputs/nano-single-smoke
```

如果第一次运行还在下载 RTMDet，检查此路径层级是否完全一致：

```text
assets/models/cache/hub/checkpoints/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth
```

**G6 通过检查：**

- [ ] 两个模型均离线命中本地权重。
- [ ] 推理明确运行在CUDA。
- [ ] 输出图和JSON均生成。
- [ ] 图片中的人体检测结果合理。

---

## 13. G7：实时 humanpose 从1人逐步升到2人

### 13.1 先运行核心单元测试

```bash
source /home/jetson/gaussflow/venvs/humanpose/bin/activate
cd /home/jetson/gaussflow/humanpose

env PYTHONPATH=src python -m pytest -q \
  tests/test_lx_camera_source.py \
  tests/test_depth_connected.py \
  tests/test_local_multi_person.py \
  tests/test_live_multi_profile.py \
  tests/test_shadow_identity.py \
  tests/test_stickman_websocket.py
```

如果某项测试依赖当前 Nano 未安装的非生产 GUI包，先记录准确测试名和错误；不要为了一个无关 GUI测试安装整套 Open3D。

### 13.2 单人、100帧、禁止发布、无GUI

```bash
set -o pipefail

env \
  PYTHONPATH=src \
  LD_LIBRARY_PATH=/opt/MRDVS/lib:/usr/local/cuda/lib64 \
  python scripts/view_live_multi_person.py \
    --live-config configs/live.nano.yaml \
    --source sdk \
    --device cuda:0 \
    --detector auto \
    --identity-tracker geometry \
    --recovery-method depth_connected \
    --max-persons 1 \
    --validate-only \
    --max-frames 100 \
    --no-publish-stickmen \
  2>&1 | tee /home/jetson/gaussflow/diagnostics/g7-one-person-100.log
```

这里显式写 `cuda:0`，目的是 CUDA失败时立即暴露，避免 `auto` 静默退回CPU。

### 13.3 双人、100帧、禁止发布、无GUI

```bash
set -o pipefail

env \
  PYTHONPATH=src \
  LD_LIBRARY_PATH=/opt/MRDVS/lib:/usr/local/cuda/lib64 \
  python scripts/view_live_multi_person.py \
    --live-config configs/live.nano.yaml \
    --source sdk \
    --device cuda:0 \
    --detector auto \
    --identity-tracker geometry \
    --recovery-method depth_connected \
    --max-persons 2 \
    --validate-only \
    --max-frames 100 \
    --no-publish-stickmen \
  2>&1 | tee /home/jetson/gaussflow/diagnostics/g7-two-person-100.log
```

不要用 `--detector whole_image` 测多人；它绕过人体检测，不能得到真正的多人框。

### 13.4 同时监控 Nano

另开一个 SSH终端：

```bash
tegrastats --interval 1000
```

再开一个终端：

```bash
watch -n 2 'free -h; df -h /home; journalctl --disk-usage'
```

判断：

- 推理 `total` 若稳定不高于约 66.7 ms，才能跟满15 FPS相机。
- 若更慢但结果正确，先记录为性能问题，不要混成坐标轴/骨架问题。
- 看是否发生热降频、内存持续上涨或 OOM。
- 不要在 Nano 上同时打开 Open3D、本地浏览器和推理服务做首次性能测试。

### 13.5 为什么第一版固定用这组参数

```text
detector=auto
identity_tracker=geometry
recovery_method=depth_connected
max_persons=1 → 2
```

- `auto` 才有真实多人检测。
- `geometry` 是当前成熟基线；`shadow` 留到稳定后 A/B。
- `depth_connected` 比 `hybrid/pointcloud_cluster` 更适合 Nano 首轮CPU预算。
- 人数从1到2可以分辨“单模型太慢”和“多人开销太高”。

**G7 通过检查：**

- [ ] 核心测试通过或已解释非生产依赖失败。
- [ ] 单人100帧完成，无CUDA/相机错误。
- [ ] 双人100帧完成，track ID独立。
- [ ] 没有持续OOM、热降频或内存泄漏。
- [ ] 已记录单人/双人的平均FPS和每阶段耗时。

---

## 14. G8：部署 FastAPI（数据库/对象存储先留旧主机）

> 本章描述完整业务后端。当前Nano实时模式没有采用它，而是使用
> 第27.13节的 `app.realtime_app:app`，因此不依赖MySQL和RustFS。

### 14.1 当前后端的真实约束

- 后端启动时会连接 MySQL，并执行 `Base.metadata.create_all`；MySQL不可达可能直接导致启动失败。
- `create_all` 只能创建缺失表，不是 Alembic迁移工具，不会安全修改已有表结构。
- RustFS初始化失败通常只记录 warning，`/api/health` 仍可能成功，但上传和场景资源会失败。
- WebSocket Hub是进程内对象，Uvicorn必须 `--workers 1`。
- `start.py` 使用 `reload=True`，只适合开发，不用于 systemd生产服务。
- 当前后端每帧 INFO日志较多，稳定后应改为 WARNING或将逐帧日志降为 DEBUG。

### 14.2 旧主机 MySQL创建 Nano 专用账号

先确认旧主机管理网 IP。下面假设：

```text
旧主机：192.168.8.104
Nano：  192.168.8.120
```

如果实际地址不同，替换所有示例。

**[MySQL主机]** 登录 MySQL：

```bash
mysql -u root -p
```

执行：

```sql
CREATE DATABASE IF NOT EXISTS gaussflow
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

CREATE USER IF NOT EXISTS
  'gaussflow_nano'@'192.168.8.120'
  IDENTIFIED BY '在这里填写一个新的高强度数据库密码';

GRANT ALL PRIVILEGES ON gaussflow.*
  TO 'gaussflow_nano'@'192.168.8.120';

SHOW GRANTS FOR 'gaussflow_nano'@'192.168.8.120';
```

不要让 Nano 使用远程 root账号。

检查 MySQL监听地址：

```bash
sudo ss -ltnp | grep ':3306'
```

如果只监听 `127.0.0.1:3306`，编辑 MySQL配置，把 `bind-address` 限制到旧主机的管理网 IP `192.168.8.104`；不要为了省事暴露到公网接口。修改后：

```bash
sudo systemctl restart mysql
sudo systemctl status mysql --no-pager
```

若旧主机启用 UFW，只允许 Nano：

```bash
sudo ufw allow from 192.168.8.120 to any port 3306 proto tcp
```

### 14.3 数据库密码含特殊字符时先URL编码

如果密码包含 `@`、`:`、`/`、`#`、`?` 等字符，不能原样塞进 SQLAlchemy URL。可在 Nano执行：

```bash
python3 - <<'PY'
from urllib.parse import quote_plus
print(quote_plus("把真实数据库密码临时粘贴在这里"))
PY
```

把输出的编码结果放入连接 URL。执行完清理终端历史中可能留下的敏感命令；更稳妥的方法是在离线编辑器里编码，不把密码写进共享终端记录。

### 14.4 创建后端独立 venv

**[Nano]**：

```bash
python3 -m venv /home/jetson/gaussflow/venvs/backend
source /home/jetson/gaussflow/venvs/backend/bin/activate

python -m pip install --upgrade pip setuptools wheel

cd /home/jetson/gaussflow/platform/backend-fastapi
python -m pip install -r requirements.txt
python -m pip check
```

当前 `requirements.txt` 主要是下限而非完整 lock。首次成功后立即冻结：

```bash
python -m pip freeze \
  > /home/jetson/gaussflow/diagnostics/g8-backend-working-freeze.txt
```

不要让这个 venv 安装/升级 humanpose 的 torch、numpy或mmcv。

### 14.5 创建 `.env`

```bash
cd /home/jetson/gaussflow/platform/backend-fastapi
cp .env.example .env.nano.example
nano .env
```

推荐第一阶段内容如下；把所有示例密码和 IP替换成真实值：

```dotenv
APP_ENV=prod
LOG_DIR=/home/jetson/gaussflow/platform/backend-fastapi/logs
LOG_LEVEL=INFO

GAUSSFLOW_DATASOURCE_URL=mysql+aiomysql://gaussflow_nano:URL编码后的密码@192.168.8.104:3306/gaussflow?charset=utf8mb4
GAUSSFLOW_DATASOURCE_USERNAME=gaussflow_nano
GAUSSFLOW_DATASOURCE_PASSWORD=真实密码

GAUSSFLOW_MINIO_ENDPOINT=http://192.168.8.104:9000
GAUSSFLOW_MINIO_CONSOLE_URL=http://192.168.8.104:9001
GAUSSFLOW_MINIO_ACCESS_KEY=替换为真实访问密钥
GAUSSFLOW_MINIO_SECRET_KEY=替换为真实秘密密钥
GAUSSFLOW_MINIO_SECURE=False
GAUSSFLOW_MINIO_BUCKET=gaussflow

GAUSSFLOW_JWT_SECRET=替换为至少32字节的随机值
GAUSSFLOW_JWT_ISSUER=gaussflow
GAUSSFLOW_JWT_TTL_MINUTES=43200

GAUSSFLOW_CORS_ALLOWED_ORIGINS=http://192.168.8.120

TRAINING_SCRIPT_PATH=/nonexistent/training-disabled-on-nano.py
TRAINING_CONFIG_PATH=/nonexistent/training-disabled-on-nano.yml
```

生成 JWT随机值：

```bash
openssl rand -hex 32
```

创建日志目录并保护配置：

```bash
mkdir -p /home/jetson/gaussflow/platform/backend-fastapi/logs
chmod 700 /home/jetson/gaussflow/platform/backend-fastapi/logs
chmod 600 /home/jetson/gaussflow/platform/backend-fastapi/.env
```

为什么训练路径写成明确不存在：让误点训练按钮时清楚失败，而不是误以为 Nano 已具备 `gaussian` Conda环境。实时人形展示不依赖该训练路径。

### 14.6 旧主机 RustFS注意事项

如果 RustFS留在 `192.168.8.104:9000`：

- Nano 后端必须能访问该地址。
- 浏览器若直接访问预签名 URL，也必须能访问 `192.168.8.104:9000`。
- RustFS需允许网页来源 `http://192.168.8.120` 的 CORS。
- 防火墙只对管理网/Nano开放需要的端口，不暴露公网。
- 不要把 endpoint写成 `localhost:9000`；在 Nano 中 localhost指向 Nano自己。

简单连通性检查：

```bash
curl -I --max-time 5 http://192.168.8.104:9000/ || true
```

### 14.7 前台启动验证

必须从 backend目录启动，因为配置按相对 `.env` 读取：

```bash
source /home/jetson/gaussflow/venvs/backend/bin/activate
cd /home/jetson/gaussflow/platform/backend-fastapi

python -m uvicorn app.main:app \
  --host 127.0.0.1 \
  --port 8000 \
  --workers 1
```

另开一个 Nano终端：

```bash
curl -fsS http://127.0.0.1:8000/api/health
curl -I http://127.0.0.1:8000/docs
```

真实地址是：

```text
健康检查：http://127.0.0.1:8000/api/health
Swagger：  http://127.0.0.1:8000/docs
OpenAPI：  http://127.0.0.1:8000/api/openapi.json
WebSocket：ws://127.0.0.1:8000/api/realtime/ws
```

不是 `/api/docs`。

### 14.8 后端测试

停止前台 Uvicorn后，执行：

```bash
cd /home/jetson/gaussflow/platform/backend-fastapi
source /home/jetson/gaussflow/venvs/backend/bin/activate
python -m pytest -q tests/test_realtime_ws.py
```

如果后端无法启动：

| 日志 | 处理 |
|---|---|
| MySQL connection refused | 检查bind-address、防火墙、服务状态和IP |
| Access denied | 检查专用用户host是否正好是Nano固定IP、密码是否URL编码 |
| Unknown database | 先创建 `gaussflow` 数据库 |
| 日志目录 permission denied | 确认LOG_DIR存在且属于jetson用户 |
| RustFS warning | 健康检查可能仍成功；单独查endpoint/密钥/bucket |

初次联调保持 `LOG_LEVEL=INFO`。稳定运行后改为 `WARNING`，避免15 FPS逐帧日志写放大。

**G8 通过检查：**

- [ ] backend使用独立venv。
- [ ] `.env` 权限是600，不含默认密钥。
- [ ] Nano能连接远程MySQL。
- [ ] `/api/health` 成功。
- [ ] Uvicorn明确只有一个worker。
- [ ] WebSocket测试通过。

---

## 15. 部署 React 静态文件和 Nginx

### 15.1 把前端发布到 `/srv` 并建立可回滚软链接

**[Nano]**：

```bash
sudo install -d -m 0755 \
  /srv/gaussflow/releases/frontend-20260812-01

sudo cp -a \
  /home/jetson/gaussflow/releases/frontend-20260812-01/. \
  /srv/gaussflow/releases/frontend-20260812-01/

sudo chown -R root:root \
  /srv/gaussflow/releases/frontend-20260812-01

sudo find /srv/gaussflow/releases/frontend-20260812-01 \
  -type d -exec chmod 0755 {} +

sudo find /srv/gaussflow/releases/frontend-20260812-01 \
  -type f -exec chmod 0644 {} +

sudo ln -sfn \
  /srv/gaussflow/releases/frontend-20260812-01 \
  /srv/gaussflow/frontend-current

readlink -f /srv/gaussflow/frontend-current
```

这样下一次更新只需创建新版本目录、校验后切换软链接；回滚也是切回旧目录。

### 15.2 写 Nginx站点配置

```bash
sudo nano /etc/nginx/sites-available/gaussflow
```

完整粘贴：

```nginx
map $http_upgrade $connection_upgrade {
    default upgrade;
    ''      close;
}

server {
    listen 80 default_server;
    listen [::]:80 default_server;
    server_name _;

    root /srv/gaussflow/frontend-current;
    index index.html;

    client_max_body_size 600m;

    add_header Cross-Origin-Opener-Policy "same-origin" always;
    add_header Cross-Origin-Embedder-Policy "require-corp" always;

    location = /api/realtime/ws {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection $connection_upgrade;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400s;
        proxy_send_timeout 86400s;
        proxy_buffering off;
    }

    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_request_buffering off;
        proxy_read_timeout 600s;
    }

    location /docs {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location / {
        try_files $uri $uri/ /index.html;
    }
}
```

关键点：

- `proxy_pass` 后面没有额外 `/`，因此会保留 `/api/...` 路径。
- WebSocket必须有 HTTP/1.1、Upgrade、Connection 和长 timeout。
- `try_files` 让React路由刷新时不返回404。
- 600 MB覆盖后端默认最大512 MB文件；大文件分片也不会被Nginx默认1 MB限制拦截。
- COOP/COEP与Vite开发环境保持一致；若外部 RustFS资源被浏览器拦截，需要给外部资源补CORS/CORP，而不是随意删除安全头。

### 15.3 启用站点

先查看现有链接：

```bash
ls -l /etc/nginx/sites-enabled
```

创建链接：

```bash
sudo ln -s \
  /etc/nginx/sites-available/gaussflow \
  /etc/nginx/sites-enabled/gaussflow
```

若默认站点仍占用 `default_server`，只删除其启用链接，原配置仍保留在 `sites-available`：

```bash
sudo unlink /etc/nginx/sites-enabled/default
```

验证后再重载：

```bash
sudo nginx -t
sudo systemctl reload nginx
sudo systemctl status nginx --no-pager
```

`nginx -t` 不通过时不要强行重启。

### 15.4 Nginx检查

```bash
curl -I http://127.0.0.1/
curl -fsS http://127.0.0.1/api/health
curl -I \
  'http://127.0.0.1/假人/mixamo-avatar-delivery/public/avatars/character-a.glb'
```

再在局域网电脑访问：

```text
http://192.168.8.120/
```

页面若能打开但刷新某个路由返回404，检查 `try_files`。Mixamo 404时检查中文目录有没有在 rsync/build时漏掉。

---

## 16. G9–G11：systemd 自启动、WebSocket 和整机重启验收

> 本章的旧服务名为通用方案。当前实机服务名和入口是
> `gaussflow-realtime.service`、`humanpose-tensorrt.service`，以
> 第27.15节为准。

### 16.1 创建 FastAPI service

```bash
sudo nano /etc/systemd/system/gaussflow-api.service
```

粘贴：

```ini
[Unit]
Description=GaussFlow FastAPI and realtime WebSocket hub
Wants=network-online.target
After=network-online.target
StartLimitIntervalSec=0

[Service]
Type=simple
User=jetson
Group=jetson
WorkingDirectory=/home/jetson/gaussflow/platform/backend-fastapi
Environment=PYTHONUNBUFFERED=1
ExecStartPre=/usr/bin/test -x /home/jetson/gaussflow/venvs/backend/bin/python
ExecStartPre=/usr/bin/test -f /home/jetson/gaussflow/platform/backend-fastapi/.env
ExecStart=/home/jetson/gaussflow/venvs/backend/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
Restart=on-failure
RestartSec=3
TimeoutStopSec=20
UMask=0077

[Install]
WantedBy=multi-user.target
```

再次确认是 `--workers 1`。

### 16.2 创建 humanpose service

```bash
sudo nano /etc/systemd/system/humanpose.service
```

粘贴：

```ini
[Unit]
Description=LANXIN RGB-D multi-person humanpose publisher
Wants=network-online.target gaussflow-api.service
After=network-online.target gaussflow-api.service
StartLimitIntervalSec=0

[Service]
Type=simple
User=jetson
Group=jetson
WorkingDirectory=/home/jetson/gaussflow/humanpose
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONPATH=/home/jetson/gaussflow/humanpose/src
Environment=LD_LIBRARY_PATH=/opt/MRDVS/lib:/usr/local/cuda/lib64
ExecStartPre=/usr/bin/test -x /home/jetson/gaussflow/venvs/humanpose/bin/python
ExecStartPre=/usr/bin/test -f /home/jetson/gaussflow/humanpose/configs/live.nano.yaml
ExecStart=/home/jetson/gaussflow/venvs/humanpose/bin/python scripts/view_live_multi_person.py --live-config configs/live.nano.yaml --source sdk --device cuda:0 --detector auto --identity-tracker geometry --recovery-method depth_connected --max-persons 2 --publish-stickmen --headless
Restart=on-failure
RestartSec=5
KillSignal=SIGINT
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
```

`KillSignal=SIGINT` 和30秒停止窗口用于让 Python进入清理路径，关闭流并释放独占相机。

`StartLimitIntervalSec=0` 让数据库、网络或相机晚于系统启动时，服务仍可按 `RestartSec` 继续重试；否则 systemd默认启动限流可能在设备恢复前把服务永久标成failed。

`After=gaussflow-api.service` 只控制启动顺序，不保证 API 已完成数据库连接；humanpose publisher自身会退避重连。即使 Hub稍晚就绪，也不应阻塞相机主循环。

### 16.3 检查 unit 内容

```bash
sudo systemd-analyze verify \
  /etc/systemd/system/gaussflow-api.service \
  /etc/systemd/system/humanpose.service

sudo systemctl daemon-reload
```

如果 Nano 用户名不是 `jetson`，必须先修改两个 unit中的 `User`、`Group`、所有 `/home/jetson` 路径。用 `id -gn jetson` 检查主组；若主组名不是 `jetson`，两个 unit里的 `Group=` 也要换成真实输出。systemd会继承该用户已经加入的 `video`/`render`附加组。

### 16.4 先启动 API

```bash
sudo systemctl enable --now gaussflow-api.service
sudo systemctl status gaussflow-api.service --no-pager
curl -fsS http://127.0.0.1:8000/api/health
```

若失败：

```bash
journalctl -u gaussflow-api.service -b -n 200 --no-pager
```

### 16.5 再启动 humanpose

先确认没有手工测试进程还占相机：

```bash
pgrep -af 'view_live_multi_person|inspect_lx_camera' || true
```

如果输出的是你刚刚运行的手工命令，先 Ctrl+C让它正常退出。

启动：

```bash
sudo systemctl enable --now humanpose.service
sudo systemctl status humanpose.service --no-pager
```

跟踪日志：

```bash
journalctl -u humanpose.service -f
```

应看到：

- 相机打开成功。
- 推理设备是 `cuda:0`。
- WebSocket publisher连接到 `127.0.0.1:8000`。
- topic是 `avatar:stickman:FF6690772788`。
- 多人事件是 `avatar.stickmen.updated`。
- `sent` 和 `ack` 随时间增长，`last_error=None`或为空。

### 16.6 启动/确认 Nginx

```bash
sudo systemctl enable nginx
sudo systemctl restart nginx
sudo systemctl status nginx --no-pager
```

启动顺序现在是：

```text
远程MySQL/RustFS → gaussflow-api → humanpose → Nginx/浏览器
```

停机和排错时反过来，尤其先停 humanpose再打开相机测试程序：

```bash
sudo systemctl stop humanpose.service
```

### 16.7 浏览器实时验收

在局域网电脑打开：

```text
http://192.168.8.120/
```

在运动页面选择：

- 实时。
- 火柴人或 Mixamo小人。
- 默认 source `FF6690772788`。

前端当前默认协议应是：

```text
WebSocket URL：由网页同源推导 /api/realtime/ws
topic：avatar:stickman:FF6690772788
event：avatar.stickmen.updated
payload：persons[]
每人：track_id + 固定26项 Halpe关节
```

验收顺序：

1. 先看火柴人，确认多人位置、track ID和坐标正确。
2. 再切 Mixamo，确认动作实时、左右脚未反向、平足不异常翘起。
3. 两人并排站立，确认前端不再把根节点都重定位到同一点。
4. 一人短暂遮挡，确认另一人不消失或串 ID。
5. 浏览器刷新，确认能自动重连。

如果页面显示“已连接、3人在画面中”，但三个人重叠：优先确认部署的是当前新 `dist`，不是远端 Git里的旧前端；检查 GLB SHA和 `frontend-current` 指向。

### 16.8 WebSocket排查

查看连接：

```bash
ss -ntp | grep ':8000' || true
journalctl -u gaussflow-api.service -b -n 200 --no-pager
journalctl -u humanpose.service -b -n 200 --no-pager
```

| 现象 | 检查 |
|---|---|
| Nginx返回502 | FastAPI未运行或127.0.0.1:8000不可达 |
| WS握手失败 | Nginx Upgrade/Connection配置、路径是否正好 `/api/realtime/ws` |
| WS连接但无骨架 | Uvicorn是否多worker、publisher是否连上、topic/event/source是否一致 |
| publisher sent增长但ack不增长 | Hub连接、协议错误或后端异常；看两端日志 |
| HTTPS页面拒绝WS | 混合内容；必须让同源自动使用 `wss://` |
| 约500ms后小人消失 | 推理低于约2 FPS触发前端stale门控，先处理性能 |

当前 WebSocket端点没有鉴权，不能直接暴露到公网。FastAPI只监听回环，外部仅通过 Nginx；路由器不要做公网端口映射。

### 16.9 整机重启验收

先确认所有配置已保存，再重启：

```bash
sudo reboot
```

重新 SSH 后：

```bash
systemctl is-active gaussflow-api.service
systemctl is-active humanpose.service
systemctl is-active nginx

curl -fsS http://127.0.0.1:8000/api/health
journalctl -u humanpose.service -b -n 100 --no-pager
```

然后刷新局域网浏览器，确认实时人形自动恢复。

**G9–G11 通过检查：**

- [ ] humanpose发布端与FastAPI单worker Hub连接。
- [ ] 浏览器能收到多人事件。
- [ ] 火柴人多人位置正确。
- [ ] Mixamo实时动作正确。
- [ ] 浏览器刷新可重连。
- [ ] Nano重启后API、humanpose、Nginx均自动恢复。

---

## 17. G12：性能、温度、内存和日志稳定性

### 17.1 监控命令

```bash
tegrastats --interval 1000
free -h
df -h
sudo nvpmodel -q --verbose
sudo jetson_clocks --show
journalctl --disk-usage
```

发生异常退出时：

```bash
sudo dmesg -T | grep -Ei 'oom|killed process|thermal|throttl' | tail -n 100
```

### 17.2 电源模式不要猜编号

先运行：

```bash
sudo nvpmodel -q --verbose
```

只选择该设备实际列出的模式编号，再执行：

```bash
sudo nvpmodel -m 实际支持的编号
```

稳定性通过并有可靠散热后，才考虑：

```bash
sudo jetson_clocks
sudo jetson_clocks --show
```

NVIDIA文档指出，启用 `jetson_clocks` 后通常不能直接切换 `nvpmodel`，需要重启后再换模式。因此先选电源模式，再锁频。

参考：[Jetson Orin Nano Platform Power and Performance](https://docs.nvidia.com/jetson/archives/r36.5/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html)

### 17.3 性能不足时按这个顺序减负

1. 浏览器放在局域网电脑，不在 Nano 本地运行。
2. 禁用本地 Open3D/RGB GUI，保持 `--headless`。
3. `max-persons` 从2降到1，判断多人增量。
4. 保持 `depth_connected`，不要先换 `hybrid/pointcloud_cluster`。
5. 保持 `geometry`；不要把 ID问题和性能问题一起改。
6. 后端稳定后把 `.env` 的 `LOG_LEVEL` 从 `INFO` 改为 `WARNING`，重启 API。
7. 检查是否热降频、swap持续使用或内存泄漏。
8. 再评估降低检测频率、模型尺寸或导出 TensorRT；不要直接用缺足点的 RTMO替代。

### 17.4 日志写放大

当前 FastAPI WebSocket在 INFO级别会记录高频消息；humanpose也有逐帧统计。15 FPS长期运行会造成日志和磁盘写放大。

后端已有20 MB轮转和14份保留，但 journal仍会保存stdout。稳定后：

```bash
nano /home/jetson/gaussflow/platform/backend-fastapi/.env
```

改成：

```dotenv
LOG_LEVEL=WARNING
```

然后：

```bash
sudo systemctl restart gaussflow-api.service
```

可选地限制整个系统 journal体积：

```bash
sudo mkdir -p /etc/systemd/journald.conf.d
sudo nano /etc/systemd/journald.conf.d/gaussflow-size.conf
```

内容：

```ini
[Journal]
SystemMaxUse=1G
RuntimeMaxUse=256M
```

应用：

```bash
sudo systemctl restart systemd-journald
journalctl --disk-usage
```

这是全系统日志上限；若 Nano还承载其他关键服务，应按实际容量调整。

### 17.5 稳定性测试矩阵

| 测试 | 建议时长/动作 | 通过条件 |
|---|---|---|
| 空闲 | 30分钟 | 服务无反复重启，内存稳定 |
| 单人 | 30分钟连续动作 | 无CUDA OOM、无相机超时风暴 |
| 双人 | 30分钟交叉/遮挡 | ID总体稳定，前端不重叠 |
| 浏览器重连 | 关闭再打开、刷新10次 | 每次能重新订阅 |
| API重启 | `restart gaussflow-api` | publisher自动重连，不影响相机进程 |
| 拔插相机 | 停服务后拔插，再启动 | 能恢复；不遗留独占句柄 |
| 整机重启 | 断电前正常关机，再启动 | 全链自动恢复 |
| 长稳 | 最少4小时，最好24小时 | 温度、内存、日志和FPS无持续恶化 |

---

## 18. 故障速查总表

### 18.1 系统、CUDA和依赖

| 现象 | 最可能原因 | 首个检查命令 | 处理原则 |
|---|---|---|---|
| JetPack 6 SD卡不启动 | QSPI固件过旧 | 查看官方Getting Started流程 | 先更新固件，不反复重写同一镜像 |
| `uname -m`不是aarch64 | 登录错机器/镜像错误 | `hostname; uname -m` | 立即停止安装 |
| `nvcc`不存在 | 未装开发组件 | `dpkg -l | grep nvidia-jetpack` | 补完整JetPack开发栈 |
| `torch.cuda.is_available=False` | 普通PyPI CPU torch或JP不匹配 | `pip show torch; python -m torch.utils.collect_env` | 卸掉错误环境，按JetPack整组重建 |
| `libcusparseLt.so.0`缺失 | 24.06+ Torch前置库未装 | `ldconfig -p | grep cusparseLt` | 按NVIDIA官方Tegra包安装 |
| torchvision没有`nms` | torchvision与Torch ABI不配 | 运行G3 NMS脚本 | 用当前Torch在Nano重编vision |
| MMCV编译被`Killed` | OOM | `dmesg -T | grep -i oom` | `MAX_JOBS=1`，NVMe临时swap |
| `mmcv._ext`不存在 | mmcv-lite或ops未编入 | `find .../mmcv -name '_ext*.so'` | `MMCV_WITH_OPS=1`重编 |
| MMCV `undefined symbol` | Torch升级后ABI失配 | `pip show torch mmcv` | 恢复配套Torch或重编MMCV |
| `no kernel image` | 未编sm_87 | 检查构建日志 | `TORCH_CUDA_ARCH_LIST=8.7` |
| pip又下载torch | 依赖解析准备覆盖NVIDIA版 | 安装日志 | 取消；固定Jetson wheel并重新验收 |
| `GLIBCXX_x.y not found` | 混入其他系统/Conda二进制 | `ldd`、`file` | 所有原生扩展在Nano同系统重建 |

### 18.2 相机和网络

| 现象 | 最可能原因 | 检查 |
|---|---|---|
| 管理SSH插相机后断开 | 相机口抢默认路由 | `ip route`，相机连接设`never-default` |
| 相机ping不通 | IP/掩码/线缆/供电 | `ip -br a`、`ethtool`、`ping` |
| ping通但ID找不到 | 广播枚举或防火墙 | 同二层、UFW、临时按IP打开 |
| `LX_E_DEVICE_NOT_FOUND` | 网段、链路、相机未就绪 | G2/G5所有检查 |
| `LX_E_CTRL_PERMISS_ERROR` | 其他进程独占 | 主机和Nano分别`pgrep -af` |
| `libLxCameraApi.so`找不到 | 动态库路径 | `ldconfig -p`、`ldd`、systemd Environment |
| 手工运行正常，systemd失败 | `.bashrc`环境没进入systemd | unit中的WorkingDirectory/PYTHONPATH/LD_LIBRARY_PATH |
| RGB/Depth mismatch增长 | 丢包或同步问题 | SDK stats、socket buffer、网卡链路 |
| 关节随机膨胀/缺失 | 先排RGB-D错帧/坏深度 | mismatch、有效深度、对齐，不先调滤波 |

### 18.3 后端、Nginx和网页

| 现象 | 最可能原因 | 处理 |
|---|---|---|
| API启动即退出 | MySQL不可达或日志目录无权限 | `journalctl -u gaussflow-api` |
| `/api/health`成功但上传失败 | RustFS仅warning | 查9000 endpoint、密钥、bucket |
| 页面刷新404 | 缺SPA fallback | `try_files $uri $uri/ /index.html` |
| API返回413 | Nginx body限制过小 | `client_max_body_size 600m` |
| WS 404/502 | Nginx路径或后端没运行 | `nginx -t`、curl health、Upgrade配置 |
| WS连接但没骨架 | 多worker/topic不一致 | 确认workers=1和topic/event/source |
| 多人都在原点重叠 | 旧前端dist | 查`frontend-current`、构建时间、GLB/JS资源 |
| Mixamo模型404 | 中文资源目录漏传 | curl完整GLB URL、检查dist |
| 模型脚反/翘 | 部署了旧retargeter | 重新由当前工作树build，不只clone远端 |
| HTTPS下WS被拒绝 | 页面HTTPS但连接ws | 同源Nginx使前端自动使用wss |
| WASM/Spark被拦截 | COOP/COEP或外部CORS/CORP | 浏览器控制台、Nginx响应头、RustFS CORS |
| 页面旧版本 | 浏览器缓存/旧软链 | 查响应资源hash、硬刷新、`readlink -f` |

### 18.4 实时质量和性能

| 现象 | 优先判断 |
|---|---|
| 前端约0.5秒后人物消失 | Nano推理低于stale门限；先查FPS |
| 小人拖影/肢体拉长 | 先确认部署当前滤波/骨长稳定代码，再查推理丢帧 |
| 某关节突然膨胀 | 深度错帧、无效深度、旧前端头部半径逻辑 |
| 背影检测差 | 2D姿态模型能力/置信度，不是Nginx问题 |
| 两人串ID | 先记录geometry基线，再单独A/B shadow |
| 一运行浏览器FPS就降 | 浏览器在Nano本地争GPU/统一内存 |
| 运行一段时间越来越慢 | 热降频、swap、日志写放大、内存增长 |
| RTMDet首次联网下载 | cache层级或权重未复制 |

---

## 19. 回滚手册

### 19.1 最快恢复旧主机实时链路

1. 在 Nano停止 humanpose，让相机正常释放：

   ```bash
   sudo systemctl stop humanpose.service
   systemctl is-active humanpose.service
   ```

2. 等待几秒，确认 Nano无相机进程：

   ```bash
   pgrep -af 'view_live_multi_person|inspect_lx_camera' || true
   ```

3. 把相机网络接回旧主机。
4. 在旧主机启动原 publisher/Viewer。
5. 不要在两个主机上同时重试相机。

Nano 的 Nginx/FastAPI可以继续运行，但它不会收到新骨架；如需要完全退回旧平台，再停止它们：

```bash
sudo systemctl stop gaussflow-api.service
sudo systemctl stop nginx
```

### 19.2 前端回滚

查看现有发布：

```bash
ls -ld /srv/gaussflow/releases/frontend-*
readlink -f /srv/gaussflow/frontend-current
```

切回已验证旧版：

```bash
sudo ln -sfn \
  /srv/gaussflow/releases/frontend-旧版本号 \
  /srv/gaussflow/frontend-current

sudo nginx -t
sudo systemctl reload nginx
```

### 19.3 Python环境回滚

不要在已验证 venv里直接大升级。新方案用新目录：

```text
/home/jetson/gaussflow/venvs/humanpose          # 当前稳定
/home/jetson/gaussflow/venvs/humanpose-next     # 候选
```

候选环境通过 G3–G7 后，再修改 systemd `ExecStart` 的 Python路径。失败时改回旧路径并：

```bash
sudo systemctl daemon-reload
sudo systemctl restart humanpose.service
```

保留：

- NVIDIA Torch wheel。
- 自编 torchvision wheel。
- 自编 MMCV wheel及SHA。
- working `pip freeze`。

### 19.4 相机 SDK回滚

如果安装前存在：

```text
/opt/MRDVS.backup-before-gaussflow
```

先停 humanpose，再人工核对备份内容和架构，之后才恢复。不要在服务运行时替换 `.so`。

### 19.5 数据回滚

第一阶段 MySQL/RustFS留旧主机，因此 Nano故障不会改变数据位置。若后续迁库：

- 切换前做最终只读窗口和全量备份。
- SQL dump与对象桶快照使用同一个时间点标记。
- 回滚时数据库记录和对象一起回滚。

---

## 20. 后续更新流程

### 20.1 更新前端

**[主机]**：

```bash
cd /home/fr1511b/program/3DGSAlgPlatform/fronted/fronted-react
npm ci
VITE_API_BASE_URL=/api npm run build
```

验证新dist后，上传到新的版本号，例如 `frontend-20260820-01`，不要覆盖正在服务的目录。Nano端：

```bash
sudo ln -sfn \
  /srv/gaussflow/releases/frontend-20260820-01 \
  /srv/gaussflow/frontend-current
sudo nginx -t
sudo systemctl reload nginx
```

验证失败立即把软链接指回前一版本。

### 20.2 更新 humanpose 源码

1. 停止 Nano humanpose，释放相机。
2. 把当前 Nano代码复制成带版本的备份，或把新代码传到 staging目录。
3. 不传 `.venv`、`outputs`、缓存日志。
4. 不排除 `assets/models/cache`，除非确认权重未变化且目标已有。
5. 使用现有稳定 venv先跑测试和100帧验证。
6. 配置变更写入 `live.nano.yaml`，不要覆盖主机原配置。
7. 验收后才重启 systemd服务。

如果依赖版本变化，创建 `humanpose-next` venv，不在稳定环境上原地升级。

### 20.3 更新 FastAPI

- 先备份 `.env`，rsync时继续排除它。
- 检查数据库模型是否变动；`create_all` 不是迁移脚本。
- 在候选 venv运行测试。
- 始终保持 `--workers 1`，直到 WebSocket Hub改为外部broker。
- 重启后先curl health，再启动/检查publisher。

### 20.4 更新 JetPack/Torch/MMCV

把这三者视为一个不可拆分的兼容组：

```text
JetPack/L4T/CUDA
       ↕
NVIDIA aarch64 PyTorch + torchvision
       ↕
本机编译的 MMCV CUDA ops
```

任何一层升级，都创建新环境并重跑 G3–G7。不要在生产当天直接覆盖稳定组。

---

## 21. 可选第二阶段：把 MySQL 和 RustFS 也迁到 Nano

只有 G12 长稳通过、NVMe容量和内存充足时再做。全部本地化会增加：

- NVMe写入和备份责任。
- MySQL/RustFS与推理争内存/CPU。
- 单板故障时同时失去实时服务和数据。

### 21.1 MySQL迁移检查点

1. 在旧主机停止写入或进入维护窗口。
2. 重新执行一致性 `mysqldump`。
3. Nano安装可用的 MySQL 8 ARM64包，并先创建数据库。
4. 导入 dump，比较表数、关键行数和用户登录。
5. 修改 Nano `.env` datasource为 `127.0.0.1:3306`。
6. 重启 API，验证旧记录和新写入。
7. 旧数据库保留只读一段时间，不立即删除。

不要使用仓库中的小型旧SQL文件冒充现网全量数据。

### 21.2 RustFS迁移检查点

RustFS官方[安装文档](https://docs.rustfs.com/installation/)说明支持 ARM；下载页提供 `linux-aarch64` 构建。只使用官方发布的 aarch64文件，并执行：

```bash
file rustfs
./rustfs --version
sha256sum rustfs
```

不要复制旧主机的 x86_64二进制。数据迁移优先通过 S3兼容客户端逐桶镜像/校验，而不是在不了解版本内部格式时直接搬数据目录。

迁移必须核对：

- bucket `gaussflow`存在。
- 对象数量和总字节数一致。
- access key/secret已换成高强度值。
- 后端上传、下载、删除、预签名URL均工作。
- `.env` endpoint改为 `http://127.0.0.1:9000`。
- 浏览器需要的预签名URL不能错误指向浏览器自己的localhost；若URL发给远程浏览器，应使用浏览器可解析的Nano地址/域名。

在不知道当前 RustFS真实数据目录、版本和对象清单之前，不要执行迁移。

---

## 22. 安全最低线

- `.env` 权限600，不放进 Git，不随诊断包发送。
- JWT至少32字节随机值。
- 不使用 RustFS默认 `root/hongchuwudi` 或 `rustfsadmin/rustfsadmin`。
- MySQL使用专用非root用户，只授权 `gaussflow` schema和Nano固定IP。
- FastAPI只绑定 `127.0.0.1:8000`。
- 外部只开放 Nginx 80/443；3306、9000、9001不开放公网。
- 当前 WebSocket没有鉴权，只用于可信局域网；路由器不要端口映射。
- 不用 `chmod 777`，服务不以root运行。
- 若将来对公网开放，先做 HTTPS、WSS、用户鉴权、publisher权限和速率限制。
- 相机网卡不设默认网关，不与办公网混为同一广播域。

---

## 23. 没有 Codex 时如何收集完整诊断包

发生问题时，不要只拍最后一行报错。运行下面步骤，诊断包不包含 `.env` 和密码。

```bash
export DIAG_DIR=/home/jetson/gaussflow/diagnostics/support-20260812-01
mkdir -p "$DIAG_DIR"

uname -a > "$DIAG_DIR/uname.txt"
cat /etc/nv_tegra_release > "$DIAG_DIR/nv-tegra-release.txt"
dpkg-query -W nvidia-jetpack > "$DIAG_DIR/nvidia-jetpack.txt" 2>&1 || true
nvcc --version > "$DIAG_DIR/nvcc.txt" 2>&1 || true
free -h > "$DIAG_DIR/free.txt"
df -h > "$DIAG_DIR/df.txt"
ip -br address > "$DIAG_DIR/ip-address.txt"
ip route > "$DIAG_DIR/ip-route.txt"
sudo nvpmodel -q --verbose > "$DIAG_DIR/nvpmodel.txt" 2>&1
sudo jetson_clocks --show > "$DIAG_DIR/jetson-clocks.txt" 2>&1

/home/jetson/gaussflow/venvs/humanpose/bin/python \
  -m torch.utils.collect_env > "$DIAG_DIR/torch-env.txt" 2>&1

/home/jetson/gaussflow/venvs/humanpose/bin/python \
  -m pip freeze > "$DIAG_DIR/humanpose-freeze.txt"

/home/jetson/gaussflow/venvs/backend/bin/python \
  -m pip freeze > "$DIAG_DIR/backend-freeze.txt"

systemctl status gaussflow-api.service --no-pager \
  > "$DIAG_DIR/api-status.txt" 2>&1
systemctl status humanpose.service --no-pager \
  > "$DIAG_DIR/humanpose-status.txt" 2>&1
systemctl status nginx --no-pager \
  > "$DIAG_DIR/nginx-status.txt" 2>&1

journalctl -u gaussflow-api.service -b -n 500 --no-pager \
  > "$DIAG_DIR/api-journal.txt"
journalctl -u humanpose.service -b -n 500 --no-pager \
  > "$DIAG_DIR/humanpose-journal.txt"

sudo dmesg -T | tail -n 500 > "$DIAG_DIR/dmesg-tail.txt"
sudo nginx -T > "$DIAG_DIR/nginx-full.txt" 2>&1

sha256sum \
  /home/jetson/gaussflow/humanpose/assets/models/rtmpose/*.pth \
  /home/jetson/gaussflow/humanpose/assets/models/cache/hub/checkpoints/rtmdet*.pth \
  > "$DIAG_DIR/model-sha256.txt"

tar -C /home/jetson/gaussflow/diagnostics \
  -czf /home/jetson/gaussflow/diagnostics/support-20260812-01.tar.gz \
  support-20260812-01
```

复制回主机：

```bash
scp jetson@192.168.8.120:/home/jetson/gaussflow/diagnostics/support-20260812-01.tar.gz \
  /home/fr1511b/program/nano-migration-records/
```

发送诊断包前再检查其中没有 `.env`、数据库密码、JWT或RustFS密钥。

实际上 Codex 不需要安装在 Nano 上：只要当前电脑还能 SSH 到 Nano，就可以在当前 IDE/主机上继续让 Codex根据这些日志协助排查。

---

## 24. 一页式最终验收清单

### 系统

- [ ] `aarch64`、JetPack、L4T、CUDA版本已记录。
- [ ] 使用NVMe、稳定电源和主动散热。
- [ ] 没有执行普通 Ubuntu大版本升级。

### 网络/相机

- [ ] 管理网和相机网分开。
- [ ] 相机口无默认网关，管理路由正常。
- [ ] Nano能ping `192.168.1.125`。
- [ ] `/opt/MRDVS` 是aarch64库且SHA正确。
- [ ] 30帧相机测试约15 FPS，mismatch不增长。

### 推理

- [ ] NVIDIA Jetson PyTorch CUDA可用。
- [ ] torchvision CUDA NMS通过。
- [ ] MMCV CUDA ops在Nano本机构建并通过。
- [ ] RTMDet/RTMPose本地权重SHA正确，不需要临时下载。
- [ ] 单图、单人100帧、双人100帧依次通过。
- [ ] 生产参数为 `auto + geometry + depth_connected + max-persons 2 + headless`。

### 平台

- [ ] FastAPI独立venv，绑定127.0.0.1，workers=1。
- [ ] MySQL专用账户可连接，RustFS endpoint可达。
- [ ] `/api/health`成功，真实Swagger地址是`/docs`。
- [ ] Nginx有SPA fallback、WebSocket Upgrade、上传大小和COOP/COEP。
- [ ] 前端由当前工作树构建，中文Mixamo资产未遗漏。

### 实时/自启动

- [ ] publisher sent/ack增长。
- [ ] 浏览器收到 `avatar.stickmen.updated`。
- [ ] 多人位置不重叠，track ID独立。
- [ ] 火柴人和Mixamo都正常；脚方向/俯仰为当前修正版。
- [ ] 浏览器刷新、API重启、Nano重启都能恢复。

### 运维/安全

- [ ] `.env` 600权限、无默认密钥。
- [ ] WebSocket和8000端口未暴露公网。
- [ ] 日志体积受控，INFO逐帧日志已处理。
- [ ] 4小时/24小时稳定性、温度、内存、FPS已记录。
- [ ] 旧主机链路、前端旧release、旧venv均可回滚。

---

## 25. 官方参考资料

- [NVIDIA JetPack Archive](https://developer.nvidia.com/embedded/jetpack-archive)
- [JetPack 6.2.2说明](https://developer.nvidia.com/embedded/jetpack-sdk-622)
- [Orin Nano首次启动和QSPI固件说明](https://developer.nvidia.com/embedded/learn/get-started-jetson-orin-nano-devkit)
- [Jetson Linux 36.5 Developer Guide](https://docs.nvidia.com/jetson/archives/r36.5/DeveloperGuide/index.html)
- [NVIDIA Jetson PyTorch安装指南](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html)
- [NVIDIA PyTorch/JetPack兼容矩阵](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html)
- [MMCV源码编译指南](https://mmcv.readthedocs.io/en/2.x/get_started/build.html)
- [Jetson Orin Nano电源和性能模式](https://docs.nvidia.com/jetson/archives/r36.5/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html)
- [RustFS安装文档](https://docs.rustfs.com/installation/)

---

## 26. 第一次真正操作时的最短顺序

> 这是最初的通用顺序。当前这台 `nvidia-desktop` 重装或复现时，直接
> 使用下一章的真实路径、冻结制品和TensorRT/systemd命令。

如果面对长文不知道从哪里开始，只做下面这些，任何一步失败就停：

1. 在 Nano执行 G1环境记录，把输出保存。
2. 完成 G2双网段，只验证 ping。
3. 从当前主机 rsync humanpose、SDK、backend和新build的dist。
4. 手工安装 ARM64相机SDK并核对SHA。
5. 完成 G3 PyTorch CUDA门禁。
6. 完成 G4 MMCV CUDA ops门禁。
7. 相机30帧测试。
8. 单图CUDA姿态。
9. 单人100帧。
10. 双人100帧。
11. FastAPI前台运行和health。
12. humanpose前台发布，确认sent/ack。
13. Nginx和浏览器先看火柴人，再看Mixamo。
14. 最后才写/启用systemd，并做整机重启测试。

到这里，Nano的第一阶段迁移才算完成。

---

## 27. 2026-08-14 实机验证版：从主机迁移到 Nano

本章记录的是已经在当前 Jetson Orin Nano 上逐项执行并通过的流程，
不是预想方案。以后重装、复制第二台 Nano 或现场排障，优先按本章执行。

### 27.1 最终已验证状态

| 项目 | 当前实机值 |
|---|---|
| Nano主机名/用户 | `nvidia-desktop` / `nvidia` |
| 架构/系统 | `aarch64` / Ubuntu 22.04.5 |
| JetPack/L4T | JetPack 6.2.2 / R36.5.0 |
| Python/CUDA | Python 3.10.12 / CUDA Toolkit 12.6 |
| TensorRT | 10.3.0，`trtexec=/usr/src/tensorrt/bin/trtexec` |
| 内存/系统盘 | 7.4 GiB统一内存 / 238.5 GB NVMe |
| 电源模式 | 已验证 `MAXN_SUPER`，模式ID为2 |
| humanpose目录 | `/home/nvidia/program/humanpose` |
| humanpose环境 | `/home/nvidia/program/humanpose/.venv-nano` |
| 平台目录 | `/home/nvidia/program/3DGSAlgPlatform` |
| 后端环境 | `fronted/backend-fastapi/.venv-nano` |
| 生产推理 | RTMDet-M + RTMPose-M Halpe26，TensorRT FP16 |
| 3D恢复/ID | `depth_connected` / `geometry` |
| 生产人数 | 最多2人 |
| FastAPI入口 | `app.realtime_app:app`，不加载数据库/对象存储 |
| 服务名 | `gaussflow-realtime.service`、`humanpose-tensorrt.service`、`nginx` |
| 当前网页 | `http://192.168.8.119/scene` |
| 推荐局域网名称 | `http://nvidia-desktop.local/scene`（需完成mDNS检查） |

本次没有在 Nano 上部署 MySQL、RustFS 和 3DGS训练。完整
`app.main:app`会尝试连接数据库，不是当前实时边缘模式的启动入口。

### 27.2 当前最终拓扑

```text
澜芯 S10（192.168.1.125）
        │ RGB-D，约15 FPS
        ▼
USB千兆网卡 enxec1ac30273a3
Nano地址 192.168.1.188/24，无网关
        │
        ▼
view_live_multi_person_tensorrt.py
  RTMDet-M FP16：默认每2帧检测一次
  RTMPose-M Halpe26 FP16：每帧运行
  depth_connected：每个track每3帧完整恢复一次，其余帧快速引导恢复
  geometry：多人ID
        │ avatar.stickmen.updated
        ▼
ws://127.0.0.1:8000/api/realtime/ws
        │
FastAPI realtime-only，单worker
        │ Nginx同源反代
        ▼
浏览器 /scene（火柴人或Mixamo）
```

错峰参数只改变计算频率，不改变协议：RTMPose仍使用当前RGB帧，快速
深度恢复仍读取当前深度和当前2D关节。新人物、快速恢复失败或异常情况
会回退完整恢复。原始PyTorch入口和默认间隔1均保留，可随时A/B。

### 27.3 实际路径映射

| 内容 | x86主机 | Nano |
|---|---|---|
| humanpose | `/home/fr1511b/program/workspace/humanpose` | `/home/nvidia/program/humanpose` |
| 平台 | `/home/fr1511b/program/3DGSAlgPlatform` | `/home/nvidia/program/3DGSAlgPlatform` |
| 相机SDK包 | `/home/fr1511b/下载/MRDVS-2.4.60.260126-ubuntu-sdk/MRDVS` | `/home/nvidia/program/MRDVS-2.4.60.260126-ubuntu-sdk/MRDVS` |
| 相机运行库 | 不复制x86库 | `/opt/MRDVS/lib`中的aarch64库 |
| 前端静态文件 | `fronted/fronted-react/dist` | 同相对路径，由Nginx提供 |
| TensorRT ONNX | `humanpose/outputs/tensorrt_fp16/onnx` | 同相对路径 |
| TensorRT Engine | 不在x86生成 | `humanpose/outputs/tensorrt_fp16/engines` |

目录名 `fronted` 是现有仓库真实拼写，不要自行改成 `frontend`。

### 27.4 第一步：记录 Nano 基线

在 Nano 执行：

```bash
hostname
whoami
uname -m
cat /etc/os-release
cat /etc/nv_tegra_release
dpkg-query -W nvidia-jetpack nvidia-l4t-core
python3 --version
readlink -f /usr/local/cuda
nvcc --version
free -h
df -h /
lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINTS
```

当前实机关键输出为：

```text
nvidia-desktop
nvidia
aarch64
Ubuntu 22.04.5 LTS
R36 REVISION 5.0
nvidia-jetpack 6.2.2+b24
nvidia-l4t-core 36.5.0
Python 3.10.12
/usr/local/cuda-12.6
```

最初 `nvcc: command not found` 只是 `PATH` 尚未包含 CUDA；确认
`/usr/local/cuda`存在后把下面内容加入当前shell或用户配置：

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
```

### 27.5 第二步：分离管理网和相机网

当前实机接口分工：

| 接口 | 用途 | 地址 | 默认路由 |
|---|---|---|---|
| `enP8p1s0` | 管理、SSH、网页、互联网 | DHCP `192.168.8.119/24` | 是，网关`192.168.8.1` |
| `enxec1ac30273a3` | 澜芯相机专网 | 静态`192.168.1.188/24` | 否 |
| `wlP1p1s0` | 可选mDNS同网/备用Nano热点 | 当前未作为生产必需项 | 否 |

USB网卡已识别为 ASIX AX88179，驱动 `ax88179_178a`。创建相机连接：

```bash
sudo nmcli connection add \
  type ethernet \
  ifname enxec1ac30273a3 \
  con-name lanxin-camera

sudo nmcli connection modify lanxin-camera \
  ipv4.method manual \
  ipv4.addresses 192.168.1.188/24 \
  ipv4.gateway '' \
  ipv4.dns '' \
  ipv4.never-default yes \
  ipv6.method disabled \
  connection.autoconnect yes

sudo nmcli connection up lanxin-camera
```

如果 `lanxin-camera` 已存在，不要重复 `add`，只执行 `modify` 和 `up`。

验收：

```bash
ip -br address
ip route
nmcli device status
ip route get 192.168.1.125
ping -c 4 192.168.1.125
sudo ethtool enxec1ac30273a3 \
  | grep -E 'Speed|Duplex|Auto-negotiation|Link detected'
```

当前实机相机链路协商为 `100Mb/s Full`，但已经稳定传输816×612、
15 FPS RGB-D且无frame-ID mismatch。千兆仍是后续布线优化目标，不是
当前正确性阻塞项。最终路由必须保留：

```text
default via 192.168.8.1 dev enP8p1s0
192.168.1.0/24 dev enxec1ac30273a3 src 192.168.1.188
```

### 27.6 第三步：从主机复制真实工作目录

第一次迁移不能只 `git clone`：模型权重、缓存、TensorRT实验代码和前端
Mixamo资源不一定都在远端仓库。先在主机定义目标：

```bash
export NANO_TARGET=nvidia@192.168.8.119
```

复制 humanpose，保留模型但排除x86环境和运行输出：

```bash
rsync -aH --info=progress2 \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '.venv-*' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude 'outputs/' \
  --exclude 'logs/' \
  /home/fr1511b/program/workspace/humanpose/ \
  "$NANO_TARGET:/home/nvidia/program/humanpose/"
```

这里不能排除 `assets/models/cache`，否则 `--detector auto` 会在 Nano
首次运行时重新下载 RTMDet。

复制相机SDK完整包：

```bash
rsync -aH --info=progress2 \
  '/home/fr1511b/下载/MRDVS-2.4.60.260126-ubuntu-sdk/' \
  "$NANO_TARGET:/home/nvidia/program/MRDVS-2.4.60.260126-ubuntu-sdk/"
```

前端应先在x86主机构建；不要把 `node_modules` 复制到ARM64：

```bash
cd /home/fr1511b/program/3DGSAlgPlatform/fronted/fronted-react
npm ci
VITE_API_BASE_URL=/api npm run build

test -s dist/index.html
test -s 'dist/假人/mixamo-avatar-delivery/public/avatars/character-a.glb'
sha256sum 'dist/假人/mixamo-avatar-delivery/public/avatars/character-a.glb'
```

复制平台后端源码和前端dist：

```bash
rsync -aH --info=progress2 \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '.venv-*' \
  --exclude '.env' \
  --exclude '__pycache__/' \
  --exclude 'node_modules/' \
  /home/fr1511b/program/3DGSAlgPlatform/ \
  "$NANO_TARGET:/home/nvidia/program/3DGSAlgPlatform/"
```

Nano端检查：

```bash
cd /home/nvidia/program

test -f humanpose/scripts/view_live_multi_person.py
test -f 3DGSAlgPlatform/fronted/backend-fastapi/requirements.txt
test -f 3DGSAlgPlatform/fronted/fronted-react/dist/index.html

find 3DGSAlgPlatform/fronted/fronted-react/dist \
  -type f -name character-a.glb -exec sha256sum {} \;
```

当前GLB正确SHA256：

```text
45bb2e7d3471cd6033b248d80c37898dda2e26ea216425bcc1dcf99e118e72ce
```
### 27.7 第四步：手工安装相机SDK

本次没有运行厂商 `install.sh`，因为它会覆盖 `/opt/MRDVS`、修改shell
配置和系统参数。实际只安装下列三个制品：

```text
MRDVS/SDK/lib/linux_aarch64/libLxCameraApi.so
MRDVS/SDK/lib/linux_aarch64/libLxDataProcess.so
MRDVS/Sample/python/lx_camera_py-1.3.3-py3-none-any.whl
```

先确认架构和传输完整性：

```bash
SDK_ROOT=/home/nvidia/program/MRDVS-2.4.60.260126-ubuntu-sdk/MRDVS

file \
  "$SDK_ROOT/SDK/lib/linux_aarch64/libLxCameraApi.so" \
  "$SDK_ROOT/SDK/lib/linux_aarch64/libLxDataProcess.so" \
  "$SDK_ROOT/Sample/python/lx_camera_py-1.3.3-py3-none-any.whl"

sha256sum \
  "$SDK_ROOT/SDK/lib/linux_aarch64/libLxCameraApi.so" \
  "$SDK_ROOT/SDK/lib/linux_aarch64/libLxDataProcess.so" \
  "$SDK_ROOT/Sample/python/lx_camera_py-1.3.3-py3-none-any.whl"

python3 -m zipfile -t \
  "$SDK_ROOT/Sample/python/lx_camera_py-1.3.3-py3-none-any.whl"
```

已验证SHA256：

```text
0da4cece44018324689b23141e1b22dac61d30d0333118a426a669f97cf9abb2  libLxCameraApi.so
246e6f46d34ec81d9da72cbad7103177b28be2acefb4692a45744484a5f68b83  libLxDataProcess.so
a24032b9d31ac4c6dd543993f539aef6374ecc1588bba6996c16c23e869c2369  lx_camera_py wheel
```

安装运行库：

```bash
sudo test -e /opt/MRDVS && echo EXISTS || echo NOT_EXISTS
sudo install -d -m 0755 /opt/MRDVS/lib

sudo install -m 0755 \
  "$SDK_ROOT/SDK/lib/linux_aarch64/libLxCameraApi.so" \
  /opt/MRDVS/lib/libLxCameraApi.so

sudo install -m 0755 \
  "$SDK_ROOT/SDK/lib/linux_aarch64/libLxDataProcess.so" \
  /opt/MRDVS/lib/libLxDataProcess.so

printf '/opt/MRDVS/lib\n' \
  | sudo tee /etc/ld.so.conf.d/mrdvs.conf
sudo ldconfig
```

检查：

```bash
ldconfig -p | grep -E 'libLxCameraApi|libLxDataProcess'
ldd /opt/MRDVS/lib/libLxCameraApi.so
ldd /opt/MRDVS/lib/libLxDataProcess.so
```

`ldd`不能有 `not found`。设置相机socket缓冲：

```bash
printf 'net.core.rmem_max=10485760\nnet.core.wmem_max=10485760\n' \
  | sudo tee /etc/sysctl.d/90-lanxin-rgbd.conf
sudo sysctl --system
sysctl net.core.rmem_max net.core.wmem_max
```

`sysctl --system`中Jetson其他配置偶尔报告某些键不存在；只要最后两个
相机缓冲值都是 `10485760`，本步骤即通过。

### 27.8 第五步：创建 Nano 专用 humanpose 环境

为了能直接看到JetPack系统提供的 TensorRT Python包，本次 humanpose
venv使用 `--system-site-packages`；这也是为什么必须执行 `pip check`
并防止pip覆盖NVIDIA系统栈。

```bash
sudo apt-get update
sudo apt-get install -y \
  python3-venv python3-dev build-essential cmake ninja-build \
  libopenblas-dev curl git rsync pkg-config

cd /home/nvidia/program/humanpose
python3 -m venv --system-site-packages .venv-nano
source .venv-nano/bin/activate

python -m pip install --upgrade \
  pip setuptools==69.5.1 wheel

python -m pip install \
  numpy==1.26.4 \
  scipy==1.11.4 \
  opencv-python==4.10.0.84 \
  PyYAML==6.0.2 \
  websockets==16.1.1 \
  Pillow packaging psutil cffi

python -m pip install --no-deps -e .
python -m pip install --no-deps \
  /home/nvidia/program/MRDVS-2.4.60.260126-ubuntu-sdk/MRDVS/Sample/python/lx_camera_py-1.3.3-py3-none-any.whl
```

创建Nano配置副本，不覆盖主机配置：

```bash
cp -n configs/live.yaml configs/live.nano.yaml
sed -i \
  's#/home/fr1511b/下载/MRDVS-2.4.60.260126-ubuntu-sdk#/home/nvidia/program/MRDVS-2.4.60.260126-ubuntu-sdk#' \
  configs/live.nano.yaml

grep -nE 'python_wheel|library_path|open_param|source:' \
  configs/live.nano.yaml
```

必须确认：

```yaml
python_wheel: "/home/nvidia/program/MRDVS-2.4.60.260126-ubuntu-sdk/MRDVS/Sample/python/lx_camera_py-1.3.3-py3-none-any.whl"
library_path: "/opt/MRDVS/lib/libLxCameraApi.so"
open_mode: "id"
open_param: "FF6690772788"
align_depth_to_rgb: true
sync_frame: true
require_matching_frame_ids: true
```

基础导入测试：

```bash
PYTHONPATH=src python - <<'PY'
import platform
import numpy
import scipy
import cv2
import yaml
import websockets
import LxCameraSDK
import rgbd_avatar

print("machine:", platform.machine())
print("numpy:", numpy.__version__)
print("scipy:", scipy.__version__)
print("opencv:", cv2.__version__)
print("yaml:", yaml.__version__)
print("websockets:", websockets.__version__)
print("LxCameraSDK:", LxCameraSDK.__file__)
print("rgbd_avatar:", rgbd_avatar.__file__)
PY
```

### 27.9 第六步：安装本次已验证的 Jetson PyTorch栈

#### 27.9.1 重要兼容性说明

当前Nano是JetPack 6.2.2，但实机最终使用并通过测试的NVIDIA wheel为：

```text
torch 2.5.0a0+872d972e41.nv24.08
CUDA build 12.6
cuDNN 9.3.0
```

这个组合已经在本机完成GPU矩阵运算、torchvision CUDA NMS、MMCV
CUDA NMS和实际RTMDet/RTMPose推理。它是**当前设备的冻结制品**，不是
对任意JetPack 6.2设备的通用兼容承诺。重装时优先复用已保存wheel和
SHA，不要用普通PyPI `pip install torch`覆盖。

#### 27.9.2 安装 cuSPARSELt 0.8.1

下载的Tegra本地仓库包：

```text
cusparselt-local-tegra-repo-ubuntu2204-0.8.1_0.8.1-1_arm64.deb
SHA256=c3445239a57331eedcd57cc760d3220983e0fbe7458cc12811bb2e8fa7fb60ad
```

安装步骤：

```bash
cd /home/nvidia/program/vendor-packages/cusparselt-0.8.1

sudo dpkg -i \
  cusparselt-local-tegra-repo-ubuntu2204-0.8.1_0.8.1-1_arm64.deb

sudo cp \
  /var/cusparselt-local-tegra-repo-ubuntu2204-0.8.1/cusparselt-*-keyring.gpg \
  /usr/share/keyrings/

sudo apt-get update
sudo apt-get install -y cusparselt-cuda-12
sudo ldconfig

dpkg -l | grep -E 'cusparselt|libcusparselt'
ldconfig -p | grep -i cusparseLt
```

若直接 `ctypes.CDLL("libcusparseLt.so.0")`出现C++符号错误，可用下面
方式验证动态加载；PyTorch实机导入已经通过：

```bash
python - <<'PY'
import ctypes
from ctypes.util import find_library

ctypes.CDLL(find_library("stdc++"), mode=ctypes.RTLD_GLOBAL)
lt = ctypes.CDLL(find_library("cusparseLt"))
print("cuSPARSELt load: OK", lt._handle)
PY
```

#### 27.9.3 安装 NVIDIA PyTorch wheel

已保存wheel：

```text
torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
SHA256=6f75fd2d2ef840ede1a90dbcf40a5458214bee26cc803fa510cda2e8978d972a
```

安装：

```bash
source /home/nvidia/program/humanpose/.venv-nano/bin/activate
python -m pip install \
  /home/nvidia/program/vendor-packages/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
```

CUDA门禁：

```bash
python - <<'PY'
import platform
import torch

print("machine:", platform.machine())
print("torch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("cuDNN:", torch.backends.cudnn.version())
print("CUDA available:", torch.cuda.is_available())
assert torch.cuda.is_available()
print("GPU:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
x = torch.randn((1024, 1024), device="cuda")
y = x @ x
torch.cuda.synchronize()
print("GPU test: PASS", y.shape, y.device)
PY
```

当前输出必须包含 `Orin`、能力 `(8, 7)`、`CUDA available: True`。

#### 27.9.4 在 Nano 构建 torchvision 0.20.0

GitHub clone在现场网络曾超时，最终使用官方tag压缩包。制品：

```text
vision-v0.20.0.tar.gz
SHA256=b59d9896c5c957c6db0018754bbd17d079c5102b82b9be0b438553b40a7b6029

torchvision-0.20.0-cp310-cp310-linux_aarch64.whl
SHA256=cdda7b3641bebe7cf7719ea1b91166e11858897ae4b15a4c317de6da0362dc29
```

构建：

```bash
cd /home/nvidia/program/vendor-packages/torchvision
mkdir -p torchvision-0.20.0-tar-src built
tar -xzf vision-v0.20.0.tar.gz \
  -C torchvision-0.20.0-tar-src \
  --strip-components=1

cd torchvision-0.20.0-tar-src
export BUILD_VERSION=0.20.0
export CUDA_HOME=/usr/local/cuda
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST=8.7
export MAX_JOBS=2
set -o pipefail

python -m pip wheel . \
  --no-deps \
  --no-build-isolation \
  -w ../built \
  2>&1 | tee ../torchvision-build.log

cd /home/nvidia/program/humanpose
python -m pip install --no-deps \
  /home/nvidia/program/vendor-packages/torchvision/built/torchvision-0.20.0-cp310-cp310-linux_aarch64.whl
```

安装后必须离开源码目录再导入，否则Python可能误导入源码树。验证：

```bash
python - <<'PY'
import torch
import torchvision
from torchvision.ops import nms

print(torch.__version__)
print(torchvision.__version__)
print("native ops:", torchvision.extension._has_ops())
boxes = torch.tensor([[0,0,10,10],[1,1,11,11],[30,30,40,40]],
                     dtype=torch.float32, device="cuda")
scores = torch.tensor([0.9,0.8,0.7], device="cuda")
print(nms(boxes, scores, 0.5))
print("TorchVision CUDA NMS: PASS")
PY
```

直接 `ldd torchvision/_C.so`可能显示Torch库 `not found`，因为它没有
继承Python进程加载Torch后的搜索环境。正确的静态检查方式：

```bash
TORCH_LIB="$(python -c 'from pathlib import Path; import torch; print(Path(torch.__file__).parent / "lib")')"
TV_EXT="$(python -c 'import torchvision; print(torchvision._C.__file__)' 2>/dev/null || find .venv-nano -path '*/torchvision/_C.so' -print -quit)"
LD_LIBRARY_PATH="$TORCH_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  ldd "$TV_EXT" | grep 'not found' || echo 'No missing libraries'
```

### 27.10 第七步：构建 MMCV并安装OpenMMLab

#### 27.10.1 安装构建依赖

```bash
cd /home/nvidia/program/humanpose
source .venv-nano/bin/activate

python -m pip install \
  mmengine==0.10.7 \
  Cython==3.2.9 \
  six==1.16.0 \
  ninja psutil packaging addict yapf==0.40.2

python -m pip check
python -m pip freeze > diagnostics/pre-mmcv-freeze.txt
```

#### 27.10.2 构建 MMCV 2.1.0 CUDA ops

已验证源码包和wheel：

```text
mmcv-2.1.0.tar.gz
SHA256=d387bcab66b467479b6660310e23746cfc79c6e57acf04094680adb499a5cd3f

mmcv-2.1.0-cp310-cp310-linux_aarch64.whl
SHA256=da64c4db3a8989036cdefdf98766489bce0d426a78b116b7c9898f5afa39cab0
```

构建命令：

```bash
cd /home/nvidia/program/vendor-packages/mmcv
mkdir -p mmcv-2.1.0-src built
tar -xzf mmcv-2.1.0.tar.gz \
  -C mmcv-2.1.0-src \
  --strip-components=1

cd mmcv-2.1.0-src
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=8.7
export MMCV_WITH_OPS=1
export FORCE_CUDA=1
export MAX_JOBS=1
set -o pipefail

python -m pip wheel . \
  --no-deps \
  --no-build-isolation \
  -v \
  -w ../built \
  2>&1 | tee ../mmcv-build.log

cd /home/nvidia/program/humanpose
python -m pip install --no-deps \
  /home/nvidia/program/vendor-packages/mmcv/built/mmcv-2.1.0-cp310-cp310-linux_aarch64.whl
```

同样必须离开 MMCV源码目录再验证。

#### 27.10.3 修复当前NVIDIA PyTorch无distributed组件的兼容问题

当前wheel的 `torch.distributed.is_available()` 为False，而且没有
`torch.distributed.ReduceOp`和 `_distributed_c10d`。MMEngine 0.10.7会
无条件解析这些类型/导入FSDP，因此需要三个最小补丁。这个补丁只用于
本机单进程推理，不能把它解释为分布式训练已经可用。

执行下面一次性补丁脚本；它会先生成 `.before-jetson` 备份：

```bash
cd /home/nvidia/program/humanpose
source .venv-nano/bin/activate

python - <<'PY'
from pathlib import Path
import mmengine

root = Path(mmengine.__file__).parent

dist_path = root / "dist" / "dist.py"
wrappers_path = root / "model" / "wrappers" / "__init__.py"
model_path = root / "model" / "__init__.py"

for path in (dist_path, wrappers_path, model_path):
    backup = path.with_name(path.name + ".before-jetson")
    if not backup.exists():
        backup.write_text(path.read_text())

text = dist_path.read_text()
if "from __future__ import annotations" not in text:
    lines = text.splitlines(keepends=True)
    lines.insert(1, "from __future__ import annotations\n")
    dist_path.write_text("".join(lines))

for path in (wrappers_path, model_path):
    text = path.read_text()
    if "import torch.distributed as torch_dist" not in text:
        marker = "from mmengine.utils.dl_utils import TORCH_VERSION\n"
        text = text.replace(
            marker,
            "import torch.distributed as torch_dist\n\n" + marker,
            1,
        )
    text = text.replace(
        "if digit_version(TORCH_VERSION) >= digit_version('2.0.0'):",
        "if (digit_version(TORCH_VERSION) >= digit_version('2.0.0') "
        "and torch_dist.is_available()):",
    )
    path.write_text(text)

print("patched:", dist_path)
print("patched:", wrappers_path)
print("patched:", model_path)
PY
```

保存差异和语法检查：

```bash
MMENGINE_ROOT="$(python -c 'from pathlib import Path; import mmengine; print(Path(mmengine.__file__).parent)')"

python -m py_compile \
  "$MMENGINE_ROOT/dist/dist.py" \
  "$MMENGINE_ROOT/model/wrappers/__init__.py" \
  "$MMENGINE_ROOT/model/__init__.py"

diff -u \
  "$MMENGINE_ROOT/dist/dist.py.before-jetson" \
  "$MMENGINE_ROOT/dist/dist.py" \
  > diagnostics/mmengine-dist-jetson.patch || true
```

重建venv时必须重新应用补丁；重装 `mmengine==0.10.7` 也会覆盖它。

#### 27.10.4 安装mmdet、mmpose及其运行依赖

先安装纯Python wheel，不让pip替换Torch/MMCV：

```bash
cd /home/nvidia/program/vendor-packages/openmmlab
python -m pip install --no-deps \
  ./mmdet-3.2.0-py3-none-any.whl \
  ./mmpose-1.3.2-py2.py3-none-any.whl
```

补齐已经在aarch64实测的依赖：

```bash
cd /home/nvidia/program/humanpose

python -m pip install --no-deps \
  shapely==2.1.2 \
  terminaltables==3.1.10 \
  tqdm==4.67.1 \
  json-tricks==3.17.3 \
  munkres==1.1.4

python -m pip install \
  --no-deps \
  --no-build-isolation \
  chumpy==0.70

python -m pip install \
  --no-deps \
  --no-build-isolation \
  pycocotools==2.0.11 \
  xtcocotools==1.14.3
```

最终门禁：

```bash
python -m pip check

python - <<'PY'
from importlib.metadata import version
import torch
import mmengine
import mmcv
import mmcv._ext
import mmdet
import mmpose
from mmcv.ops import nms

for name in ("torch", "torchvision", "mmengine", "mmcv", "mmdet", "mmpose"):
    print(name, version(name))

boxes = torch.tensor([[0,0,10,10],[1,1,11,11],[30,30,40,40]],
                     dtype=torch.float32, device="cuda")
scores = torch.tensor([0.9,0.8,0.7], device="cuda")
dets, indices = nms(boxes, scores, 0.5)
print(dets, indices, dets.device)
print("MMCV CUDA NMS: PASS")
PY

python -m pip freeze > diagnostics/openmmlab-working-freeze.txt
```

### 27.11 第八步：相机和PyTorch姿态基线

先确保旧主机、Viewer和其他Nano进程没有占用相机：

```bash
pgrep -af 'inspect_lx_camera|view_live_multi_person|LxCamera' || true
```

只采相机30帧：

```bash
cd /home/nvidia/program/humanpose
source .venv-nano/bin/activate

PYTHONPATH=src python scripts/inspect_lx_camera.py \
  --live-config configs/live.nano.yaml \
  --camera-config configs/camera.yaml \
  --frames 30
```

本次实机结果：

```text
RGB/Depth：816×612
host interval：66.43 ms，15.05 FPS
sensor interval：66.66 ms，15.00 FPS
有效深度中位数：91.2%
RGB减Depth时间戳中位数：-19.44 ms
delivered_frame_count：30
frame_id_mismatch_count：0
last_depth_frame_id == last_rgb_frame_id
```

第一次曾出现 `LX_E_NOT_RECEIVE_STREAM`。当时SDK枚举到了管理网的另一台
S10Ultra和目标S10；确认目标ID、相机独占、接口路由后再次运行即成功。
不要因为一次流启动失败就修改标定或姿态参数。

PyTorch单图基线：

```bash
mkdir -p /tmp/humanpose-mpl

PYTHONPATH=src \
MPLCONFIGDIR=/tmp/humanpose-mpl \
python scripts/test_rtmpose_single.py \
  --image diagnostics/person-test.png \
  --device cuda:0 \
  --detector auto \
  --output-dir outputs/nano-pose2d
```

本次结果为1人、26/26有效关节、平均分约0.780。RTMDet日志显示HTTP
backend并不代表一定重新下载；项目会把Torch cache指向
`assets/models/cache`。判断离线命中的可靠方法是断网仍能成功推理，
并核对权重SHA：

```text
RTMPose 4d3e73ddd31222b7b0db36caeda396af1d7630c3b5a60451bdfa99a79e8dbb90
RTMDet  35b0c7406499e0d141dd6a0235db07c10d2bee8f891f8f4e353c16a009de30e8
```

### 27.12 第九步：导出并部署TensorRT FP16

TensorRT流程与原PyTorch脚本隔离，详细实现说明还可阅读：

```text
scripts/tensorrt_fp16/README.md
```

#### 27.12.1 Nano探针

```bash
cd /home/nvidia/program/humanpose
bash scripts/tensorrt_fp16/probe_nano.sh \
  | tee diagnostics/nano-tensorrt-probe.txt
```

已确认：TensorRT 10.3.0、CUDA 12.6、`trtexec`可用、GPU能力8.7。

#### 27.12.2 在x86主机导出ONNX

Engine不能在RTX主机生成后复制到Nano；主机只负责ONNX导出和数值
一致性检查：

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

export MMDEPLOY_DIR=/home/fr1511b/program/vendor-packages/mmdeploy-1.3.1
export PYTHON_BIN="$PWD/.venv-trt-export/bin/python"
bash scripts/tensorrt_fp16/export_onnx_host.sh

sha256sum -c outputs/tensorrt_fp16/SHA256SUMS
```

当前ONNX SHA256：

```text
ef09300eb9cff2dc7730947e9503b2ac3b1b4dbcafe0fa1a98df9200184deea8  onnx/detector/rtmdet_m_person_640.onnx
32f62e8a7ef4bee1e6394882c75980bb41399137c828a99b31276cf01edb5627  onnx/pose/rtmpose_m_halpe26_256x192.onnx
```

#### 27.12.3 复制ONNX，在Nano构建Engine

主机：

```bash
rsync -av --progress \
  /home/fr1511b/program/workspace/humanpose/outputs/tensorrt_fp16/ \
  nvidia@192.168.8.119:/home/nvidia/program/humanpose/outputs/tensorrt_fp16/
```

Nano：

```bash
cd /home/nvidia/program/humanpose/outputs/tensorrt_fp16
sha256sum -c SHA256SUMS

cd /home/nvidia/program/humanpose
bash scripts/tensorrt_fp16/build_engines_nano.sh
```

该脚本构建：

```text
outputs/tensorrt_fp16/engines/rtmdet_m_person_640_fp16.engine
outputs/tensorrt_fp16/engines/rtmpose_m_halpe26_256x192_fp16.engine
```

RTMDet Engine本次约56 MiB，构建约494秒。Engine绑定当前Orin、
TensorRT/CUDA和构建配置；JetPack/TensorRT升级后应重新构建。

#### 27.12.4 单图TensorRT验收

```bash
cd /home/nvidia/program/humanpose
source .venv-nano/bin/activate

PYTHONPATH=src python \
  scripts/tensorrt_fp16/test_single_image_nano.py \
  --image diagnostics/person-test.png
```

本次结果：1人、26/26有效关节、平均分约0.793，并生成JSON和overlay。

#### 27.12.5 TensorRT实时发布命令

```bash
cd /home/nvidia/program/humanpose
source .venv-nano/bin/activate

env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  HUMANPOSE_TRT_DETECTOR_INTERVAL=2 \
  HUMANPOSE_TRT_DEPTH_CONNECTED_INTERVAL=3 \
  PYTHONPATH=src \
  MPLCONFIGDIR=/tmp/humanpose-mpl \
  python scripts/view_live_multi_person_tensorrt.py \
    --live-config configs/live.nano.yaml \
    --camera-config configs/camera.yaml \
    --pose-config configs/pose.yaml \
    --tracking-config configs/tracking.yaml \
    --source sdk \
    --device cuda:0 \
    --detector auto \
    --identity-tracker geometry \
    --recovery-method depth_connected \
    --max-persons 4 \
    --publish-stickmen \
    --headless
```

需要测原始逐帧路径时，把两个interval设为1。需要阶段耗时日志时加：

```bash
HUMANPOSE_TRT_PROFILE=1
```

优化前实测TensorRT阶段大致为：

```text
RTMDet preprocess：约9 ms
RTMDet engine：约27.5 ms
RTMPose单人：预处理约3.8 ms，engine约3.8 ms
RTMPose双人：预处理约7.3 ms，engine约4.8 ms
```

真正剩余瓶颈主要是RTMDet及其预处理、多人depth_connected，而不是
RTMPose Engine。错峰实现已通过45项相关测试，但仍要通过现场快速移动、
遮挡和交叉测试；若出现跟框滞后，先把detector interval退回1。

### 27.13 第十步：部署realtime-only FastAPI

当前Nano只需要WebSocket Hub，不需要数据库、RustFS、登录和训练路由。
专用入口为：

```text
app.realtime_app:app
```

它提供：

```text
GET /api/health
WS  /api/realtime/ws
```

创建后端独立环境：

```bash
cd /home/nvidia/program/3DGSAlgPlatform/fronted/backend-fastapi
python3 -m venv .venv-nano
source .venv-nano/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip check
```

实机安装后的主要版本为FastAPI 0.141.1、Uvicorn 0.52.2、SQLAlchemy
2.0.52、aiomysql 0.3.2、websockets 16.1.1、aioboto3 15.5.0。
虽然requirements包含数据库依赖，realtime入口不会初始化数据库。

前台启动：

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
python -m uvicorn app.realtime_app:app \
  --host 127.0.0.1 \
  --port 8000 \
  --workers 1
```

必须只有1个worker，因为Hub位于进程内。另一个终端验证：

```bash
curl --noproxy '*' -sS http://127.0.0.1:8000/api/health
```

预期：

```json
{"status":"ok","mode":"realtime-only"}
```

不要用 `app.main:app` 做当前边缘实时启动；Nano没有MySQL时它会在启动
阶段失败。realtime-only也不提供Swagger `/docs`，这是有意的最小边界。

WebSocket自检：

```bash
python - <<'PY'
import asyncio
import json
import websockets

async def main():
    uri = (
        "ws://127.0.0.1:8000/api/realtime/ws"
        "?client_type=browser"
        "&client_id=nano-smoke"
        "&topics=avatar:stickman:FF6690772788"
    )
    async with websockets.connect(uri) as ws:
        welcome = json.loads(await ws.recv())
        print(welcome["type"], welcome["topics"])
        await ws.send(json.dumps({
            "type": "ping",
            "message_id": "nano-ping-1",
        }))
        pong = json.loads(await ws.recv())
        print(pong["type"], pong.get("reply_to"))

asyncio.run(main())
PY
```

### 27.14 第十一步：Nginx提供前端并代理WebSocket

前端应继续在x86主机构建，Nano只运行Nginx。当前静态目录：

```text
/home/nvidia/program/3DGSAlgPlatform/fronted/fronted-react/dist
```

安装Nginx：

```bash
sudo apt-get install -y nginx
```

`map`必须位于Nginx的 `http` 上下文；Ubuntu的 `conf.d`正好在该层：

```bash
sudo tee /etc/nginx/conf.d/gaussflow-websocket-map.conf >/dev/null <<'EOF'
map $http_upgrade $connection_upgrade {
    default upgrade;
    ''      close;
}
EOF
```

站点配置：

```bash
sudo tee /etc/nginx/sites-available/gaussflow >/dev/null <<'EOF'
server {
    listen 80 default_server;
    listen [::]:80 default_server;
    server_name _;

    root /home/nvidia/program/3DGSAlgPlatform/fronted/fronted-react/dist;
    index index.html;

    add_header Cross-Origin-Opener-Policy "same-origin" always;
    add_header Cross-Origin-Embedder-Policy "require-corp" always;

    location = /api/realtime/ws {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection $connection_upgrade;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400s;
        proxy_send_timeout 86400s;
        proxy_buffering off;
    }

    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 600s;
        proxy_send_timeout 600s;
        proxy_request_buffering off;
    }

    location / {
        try_files $uri $uri/ /index.html;
    }
}
EOF
```

启用前先确认没有其他业务占用默认站点：

```bash
ls -l /etc/nginx/sites-enabled
sudo ln -sfnT \
  /etc/nginx/sites-available/gaussflow \
  /etc/nginx/sites-enabled/gaussflow

if test -L /etc/nginx/sites-enabled/default; then
  sudo unlink /etc/nginx/sites-enabled/default
fi

sudo nginx -t
sudo systemctl enable --now nginx
sudo systemctl reload nginx
```

检查：

```bash
curl --noproxy '*' -sS http://127.0.0.1/api/health

curl --noproxy '*' -sSI http://127.0.0.1/scene \
  | grep -Ei 'HTTP/|content-type|cross-origin'

curl --noproxy '*' -sSI \
  'http://127.0.0.1/%E5%81%87%E4%BA%BA/mixamo-avatar-delivery/public/avatars/character-a.glb' \
  | grep -Ei 'HTTP/|content-type|content-length'
```

已验证GLB响应为200、`Content-Length: 11257640`。`application/octet-stream`
对当前Three.js加载可用。访问 `/scene` 后进入“运动 → 实时”，选择火柴人
或Mixamo并点击“开始采集”。网页刷新后实时开关会回到默认状态，需要
重新点击开始采集；这不等于WebSocket自动重连失败。

### 27.15 第十二步：配置开机自启动

#### 27.15.1 FastAPI realtime服务

```bash
sudo tee /etc/systemd/system/gaussflow-realtime.service >/dev/null <<'EOF'
[Unit]
Description=GaussFlow realtime-only FastAPI gateway
Wants=network-online.target
After=network-online.target
StartLimitIntervalSec=0

[Service]
Type=simple
User=nvidia
Group=nvidia
WorkingDirectory=/home/nvidia/program/3DGSAlgPlatform/fronted/backend-fastapi
Environment=PYTHONUNBUFFERED=1
Environment=http_proxy=
Environment=https_proxy=
Environment=HTTP_PROXY=
Environment=HTTPS_PROXY=
ExecStartPre=/usr/bin/test -x /home/nvidia/program/3DGSAlgPlatform/fronted/backend-fastapi/.venv-nano/bin/python
ExecStart=/home/nvidia/program/3DGSAlgPlatform/fronted/backend-fastapi/.venv-nano/bin/python -m uvicorn app.realtime_app:app --host 127.0.0.1 --port 8000 --workers 1
Restart=on-failure
RestartSec=3
TimeoutStopSec=20

[Install]
WantedBy=multi-user.target
EOF
```

#### 27.15.2 TensorRT humanpose服务

```bash
sudo tee /etc/systemd/system/humanpose-tensorrt.service >/dev/null <<'EOF'
[Unit]
Description=LANXIN RGB-D TensorRT multi-person publisher
Wants=network-online.target
Requires=gaussflow-realtime.service
After=network-online.target gaussflow-realtime.service
StartLimitIntervalSec=0

[Service]
Type=simple
User=nvidia
Group=nvidia
WorkingDirectory=/home/nvidia/program/humanpose
RuntimeDirectory=humanpose-tensorrt
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONPATH=/home/nvidia/program/humanpose/src
Environment=MPLCONFIGDIR=/run/humanpose-tensorrt/matplotlib
Environment=LD_LIBRARY_PATH=/opt/MRDVS/lib:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu
Environment=CUDA_MODULE_LOADING=LAZY
Environment=HUMANPOSE_TRT_DETECTOR_INTERVAL=2
Environment=HUMANPOSE_TRT_DEPTH_CONNECTED_INTERVAL=3
Environment=http_proxy=
Environment=https_proxy=
Environment=HTTP_PROXY=
Environment=HTTPS_PROXY=
ExecStartPre=/usr/bin/test -x /home/nvidia/program/humanpose/.venv-nano/bin/python
ExecStartPre=/usr/bin/test -f /home/nvidia/program/humanpose/configs/live.nano.yaml
ExecStartPre=/usr/bin/test -f /home/nvidia/program/humanpose/outputs/tensorrt_fp16/engines/rtmdet_m_person_640_fp16.engine
ExecStartPre=/usr/bin/test -f /home/nvidia/program/humanpose/outputs/tensorrt_fp16/engines/rtmpose_m_halpe26_256x192_fp16.engine
ExecStart=/home/nvidia/program/humanpose/.venv-nano/bin/python scripts/view_live_multi_person_tensorrt.py --live-config configs/live.nano.yaml --camera-config configs/camera.yaml --pose-config configs/pose.yaml --tracking-config configs/tracking.yaml --source sdk --device cuda:0 --detector auto --identity-tracker geometry --recovery-method depth_connected --max-persons 4 --publish-stickmen --headless
Restart=on-failure
RestartSec=5
KillSignal=SIGINT
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
EOF
```

启用：

```bash
sudo systemd-analyze verify \
  /etc/systemd/system/gaussflow-realtime.service \
  /etc/systemd/system/humanpose-tensorrt.service

sudo systemctl daemon-reload
sudo systemctl enable --now gaussflow-realtime.service
sudo systemctl enable --now humanpose-tensorrt.service
sudo systemctl enable --now nginx
```

`systemd-analyze verify`打印其他Jetson预装unit的warning不代表这两个unit
失败；重点是不能出现指向本unit的语法/路径错误。

检查：

```bash
systemctl is-enabled \
  gaussflow-realtime.service \
  humanpose-tensorrt.service \
  nginx

systemctl is-active \
  gaussflow-realtime.service \
  humanpose-tensorrt.service \
  nginx

curl --noproxy '*' -sS http://127.0.0.1/api/health

journalctl -u gaussflow-realtime.service -b -n 80 --no-pager
journalctl -u humanpose-tensorrt.service -b -n 120 --no-pager
```

publisher正常结束或统计日志应看到submitted/sent/ack同步增长，
`last_error=None`。`connected=False`若只出现在进程退出后的最终统计中，
表示关闭流程已经断开连接，不代表运行时发布失败。

### 27.16 第十三步：网络访问方式

#### 27.16.1 当前DHCP地址

当前 `192.168.8.119`来自路由器DHCP，不保证换路由器后不变。它只适合
当前局域网：

```text
http://192.168.8.119/scene
ssh nvidia@192.168.8.119
```

不要把管理网卡永久写死为 `192.168.8.119` 后带到任意网络；如果现场
路由器不是 `192.168.8.0/24`，静态地址会失联或冲突。

#### 27.16.2 推荐局域网方式：mDNS

mDNS让DHCP地址变化后仍按主机名访问。Nano和控制主机必须连接同一个
路由器、交换机或Nano热点；mDNS不会跨互联网和不同路由器。

Nano：

```bash
hostname
systemctl is-active avahi-daemon

sudo apt-get install -y avahi-daemon
sudo systemctl enable --now avahi-daemon
```

控制主机：

```bash
getent hosts nvidia-desktop.local
ping -c 3 nvidia-desktop.local
curl --noproxy '*' -I http://nvidia-desktop.local/scene
```

成功后固定使用：

```text
网页：http://nvidia-desktop.local/scene
SSH： ssh nvidia@nvidia-desktop.local
```

访客Wi-Fi、企业VLAN或启用客户端隔离的AP可能屏蔽mDNS和设备互访。

#### 27.16.3 无路由器备用：Nano热点

无线网卡已确认支持 `AP`，但本次实机记录只完成能力探测，是否正式启用
要按现场需求决定。推荐独立网段 `10.77.0.1/24`，不要使用相机网段。
控制主机必须先连接 `HumanPose-Nano`，然后访问：

```text
http://10.77.0.1/scene
ssh nvidia@10.77.0.1
```

热点有距离范围，适合近距离现场操作。控制主机没有连接热点或其他可达
网络时，不能凭空访问 `10.77.0.1`。

#### 27.16.4 跨地区远程：Tailscale或等价私有组网

异地远程不要公网映射当前无鉴权WebSocket。Nano和控制主机均联网后，
可使用私有组网的固定虚拟IP/名称进行SSH、VS Code Remote-SSH和网页
访问。它和mDNS用途不同：mDNS只解决同一局域网内的动态地址。

### 27.17 第十四步：整机重启验收

在重启前先确保没有手工启动的Uvicorn或humanpose进程占端口/相机：

```bash
pgrep -af 'uvicorn|view_live_multi_person|inspect_lx_camera'
ss -lntp | grep -E ':(80|8000)\b'
```

然后：

```bash
sudo reboot
```

重新连接后执行：

```bash
systemctl is-active nginx
systemctl is-active gaussflow-realtime.service
systemctl is-active humanpose-tensorrt.service

curl --noproxy '*' -sS http://127.0.0.1/api/health
ip -br address
ip route
ping -c 3 192.168.1.125

journalctl -u gaussflow-realtime.service -b -n 80 --no-pager
journalctl -u humanpose-tensorrt.service -b -n 150 --no-pager
```

最后从控制主机打开 `/scene`，重新进入实时模式并点击开始采集，完成：

1. 单人正面和背面；
2. 两人并排、交叉和短暂遮挡；
3. 快速横向移动，观察隔帧检测是否滞后；
4. 火柴人与Mixamo切换；
5. 观察足部方向、足部俯仰、头部半径和躯干长度；
6. 查看publisher sent/ack是否持续同步增长。

只有重启后上述项目仍通过，才算真正达到“Nano通电后只需打开网页”。

### 27.18 当前已完成、待完成和不要做的事

已完成：

- [x] JetPack/CUDA/TensorRT环境探针。
- [x] 双网段和相机SDK aarch64部署。
- [x] 相机30帧、15 FPS、frame-ID一致性验收。
- [x] NVIDIA PyTorch、torchvision CUDA ops、MMCV CUDA ops。
- [x] RTMDet/RTMPose单图PyTorch基线。
- [x] RTMDet/RTMPose ONNX导出和Nano TensorRT FP16 Engine。
- [x] TensorRT单图26/26验收。
- [x] 多人TensorRT实时入口和错峰优化。
- [x] realtime-only FastAPI、Nginx、React/Mixamo同源链路。
- [x] systemd服务创建并启用。

仍建议补做：

- [ ] 完整断电/重启后的最终现场验收。
- [ ] mDNS从实际控制主机解析、SSH和网页三项验收。
- [ ] 单人/双人至少4小时，最好24小时稳定性测试。
- [ ] 记录最终平均FPS、P95延迟、温度、内存和journal增长。
- [ ] 若需要无路由器使用，再正式创建并测试Nano热点。

当前不要做：

- 不把x86 `.venv`、Conda环境或MMCV `.so`复制到Nano。
- 不用普通PyPI Torch覆盖当前NVIDIA wheel。
- 不把Nano生成的TensorRT Engine当成跨设备通用文件。
- 不同时运行手工humanpose和systemd humanpose，避免相机独占冲突。
- 不启动多Uvicorn worker。
- 不把8000端口和无鉴权WebSocket直接映射公网。
- 不因网络迁移重新交换坐标轴、左右关节、toe/heel索引。

### 27.19 当前实机最短运维命令

查看总状态：

```bash
systemctl is-active nginx gaussflow-realtime.service humanpose-tensorrt.service
curl --noproxy '*' -sS http://127.0.0.1/api/health
```

实时日志：

```bash
journalctl -u humanpose-tensorrt.service -f
```

更新humanpose前先释放相机：

```bash
sudo systemctl stop humanpose-tensorrt.service
```

更新后启动：

```bash
sudo systemctl start humanpose-tensorrt.service
```

服务失败：

```bash
systemctl status humanpose-tensorrt.service --no-pager -l
journalctl -u humanpose-tensorrt.service -b -n 200 --no-pager
```

Nginx 502：

```bash
systemctl status gaussflow-realtime.service --no-pager -l
curl --noproxy '*' -sS http://127.0.0.1:8000/api/health
sudo nginx -t
```

回退到PyTorch入口时，先停TensorRT服务，再手工使用
`scripts/view_live_multi_person.py`；两条链路不能同时占相机。

## 28. 2026-08-14 增量功能：启动自动校准与实时出生原点编辑

本节是在第27章已经部署成功的基础上做增量更新，不需要重装CUDA、
PyTorch、MMCV、TensorRT、相机SDK、FastAPI或Nginx。

### 28.1 “自动校准”具体校准什么

当前实现把容易混淆的三层参数分开处理：

1. 相机内参 `fx/fy/cx/cy`：每次打开LANXIN SDK后，从**已经对齐到RGB的
   实时流**读取，不再把 `camera.yaml` 的静态值当SDK输入。启动校准会检查
   多帧内参和分辨率是否保持一致，并把结果写进校准JSON。
2. RGB/Depth之间的工厂外参：仍由LANXIN SDK的
   `DEPTH_TO_RGB + ENABLE_SYNC_FRAME + frame ID门控`负责。应用层不重复估计
   这组厂家标定参数。
3. 相机坐标到应用坐标外参：启动时从多帧深度图下半部采样地面，排除检测
   到的人框，用RANSAC估计地面法向和相机离地高度。由此自动更新roll、
   pitch和Z平移。

只观察地面无法唯一确定绝对yaw和场景XY原点。因此实现会保留
`application_extrinsics`里已有的水平朝向和XY平移，只校平和修正相机高度。
这既避免坐标轴突然旋转90度，也不会声称完成了地面数据无法提供的标定。

自动校准只在进程启动时执行一次，接受后立即冻结；不会运行中持续修改外参，
所以人物不会因地面估计微小波动而漂移。若地面被遮挡、支持率不足或残差过大，
默认记录错误并回退到原YAML外参，服务仍可启动。

主机配置新增：

```yaml
live:
  auto_calibration:
    enabled: true
    sample_frame_count: 12
    max_attempt_frame_count: 30
    exclude_detected_people: true
    min_inlier_ratio: 0.35
    max_residual_p95_m: 0.04
    fallback_to_config: true
    output_path: "outputs/calibration/live_camera_calibration.json"
```

校准时应满足：相机固定不动、画面下半部能看到足够地面、不要让多人完全挡住
地面。移动相机支架后只需重启humanpose服务，即会重新校准。

### 28.2 主机代码和测试结果

本次涉及的humanpose文件：

```text
configs/live.yaml
src/rgbd_avatar/live/auto_calibration.py
src/rgbd_avatar/live/extrinsics.py
src/rgbd_avatar/live/__init__.py
src/rgbd_avatar/pipeline/live_multi_person.py
tests/test_live_auto_calibration.py
```

已执行：

```bash
conda run -n rgbd-avatar pytest -q \
  tests/test_live_auto_calibration.py \
  tests/test_ground_plane.py \
  tests/test_live_contracts.py \
  tests/test_lx_camera_source.py \
  tests/test_local_multi_person.py \
  tests/test_live_multi_profile.py
```

当前定向结果为 `49 passed`。除本机无显示环境会触发原生GUI段错误的
`tests/test_viser_mixamo_viewer.py` 外，全套结果为 `245 passed, 1 skipped`；
该GUI测试不在Nano headless生产链路中。

### 28.3 实时小人出生原点编辑

原实现已经能移动预设小人的出生原点，但三处逻辑只允许 `preset`：

- 点击拾取只在预设模式启用；
- 右下角XYZ/TransformControls面板只在预设模式显示；
- 实时track短暂 `visible=false` 时会删除实体，手调原点随之丢失。

现在预设和实时模式共用同一套原点编辑：

1. 进入 `/scene`，打开“运动 → 实时”。此时**不需要启动采集，也不需要画面中已经有人**。
2. 点击“显示布置标轴”，在3DGS场景中把黄色出生点标轴拖到客厅等目标位置。
   标轴展开后可以在“移动/旋转”之间切换：移动模式调整X/Y/Z，旋转模式只开放
   绿色场景Y轴，调整人物的水平正面和行走方向，不允许把人物倾斜或倒置。布置完成后
   点击“完成布置”。侧栏会同时显示当前3DGS场景X/Y/Z和方向角（度）。
3. 点击“开始采集”。首个可靠人物的脚底根会精确落在预布置标轴上；多人同时出现时，
   其他人保留相对于首人的水平间距，不会全部重叠在标轴上。标轴旋转会同时应用到
   人物身体朝向、后续根运动以及多人的相对位置，而不是只旋转一个可视化图标。
   旋转拖动期间由Three.js直接预览，松手时才一次性提交方向；内部使用连续yaw解开
   `+180°/-180°` 回绕，避免转到180度附近时标轴卡顿或跳变。
4. 画面出现人物后，仍可直接点击目标火柴人或Mixamo小人。右下角的“实时出生原点”
   面板用于二次调整该人物的独立X/Y/Z；预布置全局标轴会自动退出，避免两套TransformControls竞争。
5. 实时模式与预设预览使用相同的“出生点锚定”语义：前端记录该人物第一个
   可靠帧的左右脚踝中点（脚踝不可用时回退到髋中点）作为参考根。渲染位置为
   `出生点 + (当前根 - 参考根)`。因此把出生点设到客厅后，参考帧的脚底
   就落在客厅，之后人物只在该锚点周围重现自身相对移动，不再叠加后端绝对根坐标。
6. 后端roster仍保留该track但一帧不可见时，原点不会被清零。

预布置和手动输入的X/Y/Z都是**3DGS场景坐标**；场景已做米制标定时，多人之间的初始
根坐标差会按 `1 / realPerSceneUnit` 换算为场景单位。坐标映射仍是
后端 `+X -> Three +X`、后端 `+Y -> Three -Z`，场景高度使用出生点Y。

实时人物是会话级实体。人物真正离开、后端删除track、浏览器整页刷新或重新开始
会话后，手动原点会回到默认值；这避免旧track ID复用时把新人物放到旧位置。

本次涉及的前端文件：

```text
fronted/fronted-react/src/pages/SceneViewer/SceneViewer.tsx
fronted/fronted-react/src/pages/SceneViewer/components/MotionModePanel.tsx
fronted/fronted-react/src/pages/SceneViewer/components/StickmanActorPanel.tsx
fronted/fronted-react/src/pages/SceneViewer/controllers/useStickmanActorEntities.ts
fronted/fronted-react/src/pages/SceneViewer/controllers/useRealtimeSpawnOriginEdit.ts
fronted/fronted-react/src/pages/SceneViewer/hooks/useObjectPostureGizmo.ts
fronted/fronted-react/src/pages/SceneViewer/hooks/useStickmanControls.ts
fronted/fronted-react/src/pages/SceneViewer/hooks/useStickmanRenderer.ts
fronted/fronted-react/src/pages/SceneViewer/hooks/useStickmanAvatarRenderer.ts
fronted/fronted-react/src/pages/SceneViewer/utils/stickmanMotion.ts
```

主机已执行 `npm run build`，TypeScript和Vite生产构建均成功。更新后必须复制整个
`dist/`，不要只复制一个JS文件，因为Vite资源文件名包含内容hash。

### 28.4 火柴人身体宽度、移动距离和高度的真实比例

前端现在对火柴人采用两段独立的尺度换算，不能再把身体和根运动一起只缩放Y轴：

1. 从26点中减去脚踝中点，得到以米为单位的身体局部坐标；
2. 根据稳定身体高度计算默认 `1.7 m / 检测身体高度`，对身体X/Y/Z三轴统一等比
   缩放，因此肩宽、髋宽、四肢长度和高度保持同一比例；
3. 根节点位移单独保留为米，不乘人物身高修正系数；
4. 最后身体与根运动都乘 `sceneUnitsPerMeter = 1 / realPerSceneUnit`。

可概括为：

```text
场景身体点 = 身体局部米坐标 × 身高等比系数 × sceneUnitsPerMeter
场景根位移 = 真实根位移米坐标 × sceneUnitsPerMeter
最终关节点 = 场景身体点 + 场景根位移
```

骨骼、关节和头部的半径也使用同一个等比系数，不会在大尺度或小尺度3DGS场景中
突然变粗或变细。这里的 `1.7 m` 是当前默认人物身高；本版不在实时面板暴露身高输入。

要启用严格米制比例，必须先完成下面的平面约束标定。未完成米制标定时，前端只能按
`1 场景单位 = 1 米` 回退预览，不能保证与3DGS房间的物理尺度一致。

#### 28.4.1 在确定平面内完成尺度标定

不要直接在3DGS空间中连续点击A/B。高斯表面的命中深度会随视角、透明区域和点密度
变化，两个看似在同一墙面或地面的点可能实际落在不同深度，导致场景距离偏长。

新版标尺采用“先选平面，再在平面内测距”：

1. 场景应先完成自动校平，使地面对应Three.js的XZ平面；
2. 打开“标尺 → 比例标定”；
3. 在“测量约束平面”选择方向：
   - `地面 XZ`：固定Y高度，适合地板；
   - `正立面 XY`：固定Z深度，适合前后朝向的墙面；
   - `侧立面 YZ`：固定X位置，适合左右朝向的墙面；
4. 点击“在场景中选择平面”，只需在目标地面或墙面上点击一次。前端使用这次表面
   命中确定平面位置，并显示青色平面网格；
5. 再依次点击A、B。此时不再使用A/B点击处的高斯深度，而是计算相机射线与已确定
   数学平面的交点，所以A/B严格共面；
6. 输入A-B之间已知的真实距离，单位填写小写 `m`，点击“应用”，然后保存场景。

A/B坐标编辑器中垂直于平面的那一个坐标会被禁用；场景里的点位标轴也只显示平面内
两个方向。切换平面方向、重新选择平面位置或清除平面都会清空旧测点，防止把不同平面
的数据混在一次标定中。平面网格是本次编辑会话的辅助物；应用后的
`realPerSceneUnit`仍随场景元数据保存。

建议现场验收：

- 用场景中一段已知1米的边复核标尺读数；
- 让人物横向张开双臂，确认身体没有只变高而宽度不变；
- 让人物沿地面实际移动1米，确认场景根节点也移动1米；
- 切换火柴人/Mixamo显示，二者脚底落点和整体身高应一致，不应出现一个明显巨大。

对应实现文件：

```text
fronted/fronted-react/src/pages/SceneViewer/components/RulerPanel.tsx
fronted/fronted-react/src/pages/SceneViewer/hooks/useRuler.ts
fronted/fronted-react/src/pages/SceneViewer/hooks/useStickmanRenderer.ts
```

### 28.5 把增量更新同步到Nano

以下命令在**主机**执行。Nano使用mDNS时可把地址写成
`nvidia@nvidia-desktop.local`；若当前mDNS不可用，替换成Nano实际管理网IP。

先停Nano humanpose，释放独占相机：

```bash
ssh nvidia@nvidia-desktop.local \
  'sudo systemctl stop humanpose-tensorrt.service'
```

同步humanpose增量代码和配置：

```bash
cd /home/fr1511b/program/workspace/humanpose

rsync -av --relative \
  configs/live.yaml \
  configs/ground.yaml \
  src/rgbd_avatar/live/auto_calibration.py \
  src/rgbd_avatar/live/extrinsics.py \
  src/rgbd_avatar/live/__init__.py \
  src/rgbd_avatar/pipeline/live_multi_person.py \
  tests/test_live_auto_calibration.py \
  nvidia@nvidia-desktop.local:/home/nvidia/program/humanpose/
```

若Nano生产仍使用独立的 `configs/live.nano.yaml`，不要用主机的SDK绝对路径覆盖它。
可以让systemd命令显式增加 `--auto-calibrate`，这样使用上述默认阈值：

```bash
sudo systemctl edit --full humanpose-tensorrt.service
```

在现有 `ExecStart=` 的参数末尾增加：

```text
--auto-calibrate
```

`--ground-config`不写时默认就是
`/home/nvidia/program/humanpose/configs/ground.yaml`。如果需要指定校准输出，可再加：

```text
--calibration-output /home/nvidia/program/humanpose/outputs/calibration/live_camera_calibration.json
```

然后同步已经在主机构建完成的前端：

```bash
rsync -av --delete \
  /home/fr1511b/program/3DGSAlgPlatform/fronted/fronted-react/dist/ \
  nvidia@nvidia-desktop.local:/home/nvidia/program/3DGSAlgPlatform/fronted/fronted-react/dist/
```

### 28.6 Nano更新后检查和启动

以下命令在**Nano**执行：

```bash
cd /home/nvidia/program/humanpose
source .venv-nano/bin/activate

python -m py_compile \
  src/rgbd_avatar/live/auto_calibration.py \
  src/rgbd_avatar/live/extrinsics.py \
  src/rgbd_avatar/pipeline/live_multi_person.py

PYTHONPATH=src pytest -q \
  tests/test_live_auto_calibration.py \
  tests/test_ground_plane.py \
  tests/test_live_contracts.py
```

重新加载并启动：

```bash
sudo systemctl daemon-reload
sudo systemctl restart gaussflow-realtime.service
sudo systemctl restart humanpose-tensorrt.service
sudo nginx -t
sudo systemctl reload nginx
```

观察校准日志：

```bash
journalctl -u humanpose-tensorrt.service -b --no-pager \
  | grep -E 'auto-calibration|intrinsics|height=|fallback'
```

成功应看到类似：

```text
Camera auto-calibration accepted: fx=... fy=... cx=... cy=...
height=... m tilt=... deg inliers=.../... p95=... output=...
```

检查产物：

```bash
python -m json.tool \
  /home/nvidia/program/humanpose/outputs/calibration/live_camera_calibration.json \
  | sed -n '1,120p'
```

若日志显示fallback，先不要改阈值：移开画面下半部遮挡物、确认深度有效，再重启
服务复测。需要临时禁用可把服务参数改为 `--no-auto-calibrate`，它会继续使用
`live.nano.yaml`中的固定外参。

最后从控制主机验收：

```bash
curl --noproxy '*' -sS http://nvidia-desktop.local/api/health
ssh nvidia@nvidia-desktop.local \
  "systemctl is-active gaussflow-realtime.service humanpose-tensorrt.service nginx"
```

浏览器打开：

```text
http://nvidia-desktop.local/scene
```

依次确认：多人位置不重叠、地面方向正确、脚底高度合理、点击实时人物能打开出生
原点面板、拖动过程中人物不会被下一帧弹回、短时遮挡后原点仍保持。
