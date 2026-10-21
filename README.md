<div align="center">

# SOP Monitor · Jetson Xavier NX 视觉推理

**工具 / 手 OBB + 人体姿态的 TensorRT 实时推理实验台**

![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=cplusplus&logoColor=white)
![TensorRT](https://img.shields.io/badge/TensorRT-8.5.2-76B900?logo=nvidia&logoColor=white)
![JetPack](https://img.shields.io/badge/JetPack-5.1.1-76B900)
![Jetson](https://img.shields.io/badge/Jetson-Xavier%20NX-76B900)
![License](https://img.shields.io/badge/License-Apache%202.0-blue)

</div>

> **2026-10-19 硬件核实**：`192.168.137.230` 实际为 **Jetson Xavier NX Developer Kit**，TensorRT 报告 `Selected Device: Xavier`、Compute Capability 7.2、6 个 SM、可见显存 6848 MiB。仓库早期文档沿用的 Orin NX 名称不准确，历史实测数字均属于这台 Xavier NX。

C++17 + TensorRT FP16 的现场视觉推理工程：读取工位视频，执行工具与手 OBB、人体检测、人体与双手关键点，输出多画面实时标注与逐帧指标。

**本仓库不包含** TCN 时序判定、MQTT 通讯、SOP 步骤状态机、图片上传与业务报表——这些在工位 TL3576 与中央 TL3576 上实现。

## 效果预览

**双路实时识别**（左 03 / 右 07，联合 OBB + RTMW，`sop_nx_joint_probe`）

[![双路实时识别](cpp_infer/docs/media/twolane_10s_v2.gif)](cpp_infer/docs/media/twolane_10s_v2.mp4)

**四路识别 · 批量调度**（03 / 03 / 07 / 07，640×384 + batch 4，`sop_nx_batch_probe`）

[![四路批量识别](cpp_infer/docs/media/fourlane_10s_v2.gif)](cpp_infer/docs/media/fourlane_10s_v2.mp4)

> 两段素材均为 2026-10-21 在 NX 桌面上实录的 10 秒片段，点图可看原速 MP4（1100×620 / 25 fps）。
> 输入是 `03.mp4`、`07.mp4` **文件模拟**，不是真实 RTSP 摄像头；每格左上角的小字 `FPS … | … ms` 是程序叠加的绘制帧率与单帧耗时，不是源视频内容。
> 录制时板端正在报 `System throttled due to Over-current`（20W 6CORE），所以窗口里的任务 Hz 低于测试报告中的数值。

## 模块地图

![cpp_infer 模块地图](cpp_infer/docs/media/module-map.png?v=2)

六个模块分别对应：采集解码、检测与姿态、调度流水、加速量化、观测验收、构建文档。每个框里的文件名都是真实存在的源文件，可直接跳读。更细的调用关系见 [cpp_infer/CODE_MAP.md](cpp_infer/CODE_MAP.md)。

## 目录

- [它在系统里的位置](#它在系统里的位置)
- [模型与两条链路](#模型与两条链路)
- [代码结构](#代码结构)
- [快速开始](#快速开始)
- [运行示例](#运行示例)
- [性能速查](#性能速查)
- [文档索引](#文档索引)
- [已知边界](#已知边界)
- [提交与推送](#提交与推送)
- [模型文件校验值](#模型文件校验值)
- [许可](#许可)

## 它在系统里的位置

| 层 | 职责 | 与本仓库的关系 |
|---|---|---|
| 中央 TL3576（`:8000`） | 网页、FastAPI、SQLite、快照图片、生产统计 | 不直接通讯 |
| 工位 TL3576（Qt 上位机） | 抓图、本地队列、HTTP 上报、SOP 判定 | 未来经 NX 本地 MQTT Broker 取识别结果 |
| **NX（本仓库）** | 模型推理与结果显示 | 目前只吃视频文件，尚未接 MQTT |

三条链路的划分（中央 ↔ NX 管理走 MQTT、NX ↔ 工位生产走 NX 本地 Broker、工位 → 中央数据走 HTTP）见工作区文档 `当前系统架构.md`。**本仓库是这套系统里的算法验证台，不是已经联调完成的 NX 服务。**

## 模型与两条链路

| 链路 | 检测 | 人体 | 骨架 | 入口 |
|---|---|---|---|---|
| 三模型（旧） | YOLO-OBB 640×640，3 类 | YOLOX-tiny 416×416 | RTMW 192×256 | `sop_nx_infer`、`sop_nx_dual_probe` |
| 两模型（新） | 联合 OBB 640×640 / 640×384，4 类 | 由联合 OBB 的 person 类给出 | 同一份 RTMW 引擎 | `sop_nx_joint_probe`、`sop_nx_batch_probe` |

两类差别只有一处：**联合 OBB 多一个 person 类，省掉 YOLOX 的一次前向**；RTMW 权重、裁剪规则、SimCC 解码完全复用，所以骨架质量受人体框变化影响，必须单独验收。

联合 OBB 引擎依赖自写 SiLU 插件 `build/libsop_silu_plugin.so`（替换 93 组 `x*sigmoid(x)`）——**plan 与 so 必须一起部署**，只拷 `.plan` 会因反序列化找不到插件而失败。

## 代码结构

```text
sop_Monitor/
├── README.md                  ← 本文件
└── cpp_infer/
    ├── CODE_MAP.md            ← 模块关系、包含关系、调度流程图
    ├── CMakeLists.txt         ← 默认 3 个目标
    ├── accel.cmake / stage2.cmake   ← 4 个实验目标（SOP_BUILD_ACCEL=ON）
    ├── src/
    │   ├── main.cpp           ← 唯一主入口，4 种调度模式
    │   ├── dual_probe.cpp     ← 1/2/4 路共享引擎轮询
    │   ├── batch_probe.cpp    ← 矩形输入 + 跨工位 batch
    │   ├── joint_detector.hpp  ← 4 类联合 OBB
    │   ├── hardware_capture.hpp ← GStreamer 硬解
    │   ├── silu_plugin.cu     ← TensorRT SiLU 插件
    │   └── pipeline.hpp / bounded_queue.hpp / stage2_* / batch_*
    ├── docs/media/            ← 上图的 GIF / MP4 / PNG 素材
    ├── tests/                 ← CTest 5 项
    └── *.md                   ← 16 份按轮次归档的实验报告
```

编译期有个特殊设计：探针入口只有 3 行，靠 `#define` + 直接 `include "dual_probe.cpp"` 复用同一份实现。**先读 [CODE_MAP.md](cpp_infer/CODE_MAP.md) 第 3 节再读源码**，否则会误以为文件残缺。

## 快速开始

**依赖**：JetPack 5.1.1 / CUDA 11.4 / TensorRT 8.5.2 / OpenCV 4.5.4。引擎必须在目标 NX 上生成，不可跨机复制。

```bash
cd cpp_infer
bash build_engines.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
```

需要加速与批量实验目标时（额外要求 `libavformat` / `libavutil` / `libswscale`）：

```bash
cmake -S . -B build -DSOP_BUILD_ACCEL=ON
cmake --build build -j2
(cd build && ctest --output-on-failure)
```

**本地资源**：克隆后需自行放入以下文件，仓库不含模型下载器。

```text
sop_Monitor/
  test_vedio/07.mp4
  cpp_infer/models/
    yolo_20261007_latest/best.raw_trt2.onnx
    yolox_tiny_8xb8-300e_humanart-6f3252f9.raw_trt.onnx
    rtmw-dw-l-m_simcc-cocktail14_270e-256x192_20231122.onnx
```

前两个是适配 TensorRT 8.5 的兼容图（截断 TopK），**不能用原始 end-to-end ONNX 直接替代**。

## 运行示例

单路（三模型，逐帧）：

```bash
cd cpp_infer
./build/sop_nx_infer --source ../test_vedio/07.mp4 --model-dir models \
  --obb-stride 2 --person-stride 2 --pose-stride 3
```

单路实时并行 + 预览优化（`--preview-first` 只减少显示开销，模型仍读原图）：

```bash
./build/sop_nx_infer --source ../test_vedio/07.mp4 --model-dir models \
  --mode parallel --realtime --preview-first
```

双路（三模型共享引擎，对应上文“两工位”容量实验）：

```bash
./build/sop_nx_dual_probe --source ../test_vedio/03.mp4 \
  --second-source ../test_vedio/07.mp4 --model-dir models \
  --start-sec 180 --duration 30 --metrics output/dual.csv
```

双路（两模型，预览 GIF 里的那条命令，窗口按 2 宫格裁成 1100×330）：

```bash
./build/sop_nx_joint_probe --source ../test_vedio/03.mp4 \
  --second-source ../test_vedio/07.mp4 --model-dir models \
  --display-width 1100 --display-height 330 --duration 300
```

四路批量（预览 GIF 里的那条命令，需已生成 `models/batch_20261020` 下的引擎）：

```bash
./build/sop_nx_batch_probe --source ../test_vedio/03.mp4 \
  --second-source ../test_vedio/03.mp4 --third-source ../test_vedio/07.mp4 \
  --fourth-source ../test_vedio/07.mp4 --det-height 384 \
  --det-batch 4 --pose-batch 4 --batch-wait-ms 5 --duration 60
```

> `--duration` 在多路探针里必须大于 5 秒，前 5 秒作为启动预热不计入稳态任务率。
> NX 需先登录 GNOME 桌面；SSH 启动时程序会尝试连接当前用户的 `:0`，也可显式指定
> `DISPLAY=:0 XAUTHORITY=/run/user/1000/gdm/Xauthority`。窗口按 `q` 或 `Esc` 退出。

## 性能速查

全部为同一台 Xavier NX、`03/03/07/07` 四路文件模拟（除注明单路外），**不是真实 RTSP**。

| 轮次 | 变更 | 结果 |
|---|---|---|
| 10-15 | 三模型单路逐帧 | 8.31–10.75 FPS（180 s 位置取 5 s）；02 连续 10 s 为 12.67 FPS |
| 10-16 | 解码/推理/显示流水线（单路 07） | 27.1–27.4 FPS（无窗口）、24.65（带窗口） |
| 10-16 | `--preview-first` | 单路 23.7–24.0 → 26.26–26.30 FPS，结果指纹一致 |
| 10-16 | 两工位共享串行（三模型） | 10.36 次新识别/秒/路 |
| 10-16 | 四路共享串行（三模型） | 3.76 次/秒/路 |
| 10-19 | 联合两模型替换 OBB+YOLOX | 4.12 次/秒/路（+9.6%），骨架年龄 P95 360→320 ms，约 50% 帧仍被 200 ms 规则隐藏 |
| 10-19 | GPU RTMW 预处理 + 硬解 + 复用缓冲 | 四路 11.3 次/秒/路（单任务 22.1 ms 无窗口 / 24.9 ms 带画面） |
| 10-20 | 640×384 矩形输入 + batch 4 | **18.92 次/秒/路**（无画面）/ 16.71（带画面），四路合计 66.8 Hz |

三条必须一起读的限定：

1. **窗口 FPS 是预览消费率，不是识别更新率**；四宫格约 25 FPS 只说明显示跟得上。
2. 分阶段测量显示瓶颈在引擎执行区间（OBB 约 11.9 ms、RTMW 约 7.5 ms），CPU 后处理合计仅 0.63 ms；CUDA Graph 只提升 4–5%。
3. 矩形 + batch 组**没有通过精度验收**：12 张固定帧中 1 张手框数量变化，共同可见点差 P95 由 8.7 px 升至 32 px。速度结论不等于可用配置。

四路各 25 次识别/秒（合计 100）**尚未达到**：按当前串行结构，单任务预算需压到 10 ms。

## 文档索引

| 文档 | 内容 |
|---|---|
| [CODE_MAP.md](cpp_infer/CODE_MAP.md) | 代码地图、包含关系、调度流程图、关键常量、踩坑清单 |
| [ARCHITECTURE.md](cpp_infer/ARCHITECTURE.md) | 单路/四路实现、线程职责、帧与结果时间语义、后续多路设计 |
| [cpp_infer/README.md](cpp_infer/README.md) | 运行参数详解、四种模式对照、指标 CSV 字段 |
| [NX_TEST_REPORT.md](cpp_infer/NX_TEST_REPORT.md) | 首轮迁移测试（部分限制描述早于后续隔帧与窗口修改） |
| [PIPELINE_TEST_REPORT_20261016.md](cpp_infer/PIPELINE_TEST_REPORT_20261016.md) | 并行流水线首轮实测 |
| [OPTIMIZATION_REPORT_20261016.md](cpp_infer/OPTIMIZATION_REPORT_20261016.md) | 预览优化与多开进程容量实验 |
| [FOUR_STREAM_REPORT_20261016.md](cpp_infer/FOUR_STREAM_REPORT_20261016.md) | 四路三模型从 0 秒开始 |
| [JOINT_MODEL_GUIDE.md](cpp_infer/JOINT_MODEL_GUIDE.md) | 联合 OBB 导出、SiLU 插件、引擎构建与回退 |
| [JOINT_TEST_REPORT_20261019.md](cpp_infer/JOINT_TEST_REPORT_20261019.md) | 两模型四路性能与数值检查（含未通过项） |
| [STAGE2_GUIDE.md](cpp_infer/STAGE2_GUIDE.md) / [STAGE2_TEST_REPORT_20261019.md](cpp_infer/STAGE2_TEST_REPORT_20261019.md) | GPU 端 RTMW 预处理与 IO 复用 |
| [ACCEL_GUIDE.md](cpp_infer/ACCEL_GUIDE.md) / [ACCEL_TEST_REPORT_20261019.md](cpp_infer/ACCEL_TEST_REPORT_20261019.md) | 硬件解码、CUDA 预处理、1/2/4 推理线程 |
| [PROFILE_GUIDE.md](cpp_infer/PROFILE_GUIDE.md) / [PROFILE_REPORT_20261020.md](cpp_infer/PROFILE_REPORT_20261020.md) | 分阶段计时、CUDA Graph 对照、Nsight 采集 |
| [BATCH_GUIDE.md](cpp_infer/BATCH_GUIDE.md) / [BATCH_REPORT_20261020.md](cpp_infer/BATCH_REPORT_20261020.md) | 矩形输入与跨工位批处理完整矩阵 |

## 已知边界

- 输入是**视频文件模拟**；真实 RTSP 的网络抖动、断流重连、跨设备时钟同步均未验证。
- 多人场景只输出**得分最高的一个**人体骨架，不是多人姿态产品。
- 无跟踪、无平滑、无运动补偿；快速动作会出现结果错位，靠 200 ms 过期隐藏兜底。
- 未做 INT8 校准，未把 CUDA Graph 接入应用，未实现单进程多路作为产品功能。
- 所有性能数字都是单轮、同一段视频、桌面负载未隔离的结果，没有多轮置信区间。

## 提交与推送

目标仓库：`git@github.com:kalapaopao/sop_Monitor.git`（分支 `main`）。提交身份 `kalapaopao <214327724+kalapaopao@users.noreply.github.com>`。

每次提交前先核对：

```bash
git status --short
git diff --cached --stat
git diff --check
```

确认视频、权重、ONNX、TensorRT plan、`build/`、`output/`、日志与凭据均未被暂存。`.gitignore` 只忽略文件，**不会删除 NX 上已存在的这些资源**；如果某个文件曾被跟踪，仅改忽略规则不会自动取消跟踪。README 用到的演示素材放在 `cpp_infer/docs/media/`，是 `.gitignore` 中唯一的 `*.mp4` 例外。

### NX 连不上 GitHub 时

`192.168.137.230` 的 DNS 会被代理软件的 fake-ip 接管（`github.com` 解析到 `198.18.x.x`），22/443 直连都会超时。实测可用 Windows 侧的 HTTP 代理建立 SSH 通道：

```bash
git -c core.sshCommand="ssh -o ProxyCommand='nc -X connect -x 192.168.137.1:12000 %h %p'" push origin main
```

`192.168.137.1` 是 Windows 的 ICS 网关，`12000` 是代理的 HTTP 端口；换网络或换代理端口后需要同步修改。需要固化时：

```bash
git config core.sshCommand "ssh -o ProxyCommand='nc -X connect -x 192.168.137.1:12000 %h %p'"
```

代理不可用时的兜底：板端 `git bundle create /tmp/sop.bundle main`，用 `scp` 取回，在 Windows 侧 `git fetch /tmp/sop.bundle main:tmp && git push origin tmp:main`。

## 模型文件校验值

以下为当前 NX 上三个 ONNX 源文件的 SHA256，可用 `sha256sum` 复核。它们**不代表** raw TensorRT 兼容图或 `.plan` 的哈希。

```text
8ffd9f6ccd16b0352a37166ef3c86529a04a5ce7c6ebd463aea2fb625eedf46a  cpp_infer/models/yolo_20261007_latest/best.raw_trt2.onnx
12d3dff921d16dd22408b49569cd0f46ef829142e67c5bb694d578c85eabf570  cpp_infer/models/yolox_tiny_8xb8-300e_humanart-6f3252f9.raw_trt.onnx
68db326c5c6f3273456f803eae0faf5440478369f2c3ee67258d2f36512e8d76  cpp_infer/models/rtmw-dw-l-m_simcc-cocktail14_270e-256x192_20231122.onnx
```

## 许可

Apache License 2.0，见 [LICENSE](LICENSE)。模型权重来自第三方，授权与分发需单独处理，不在本仓库许可范围内。
