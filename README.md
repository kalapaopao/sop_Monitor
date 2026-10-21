# SOP Monitor — Jetson Xavier NX 视觉推理

> **2026-10-19 硬件核实**：`192.168.137.230` 实际为 **Jetson Xavier NX Developer Kit**，TensorRT 报告 `Selected Device: Xavier`、Compute Capability 7.2、6 个 SM、可见显存 6848 MiB。此前文档沿用的 Orin NX 名称不准确，历史实测数字均属于这台 Xavier NX，不能作为 Orin NX 的性能依据。

C++17 + TensorRT FP16 的现场视觉推理实验台：读取视频文件（后续接相机），执行工具与手 OBB、人体检测、人体与双手关键点，输出窗口画面与 CSV 指标。

**本仓库不包含**：TCN 时序判定、MQTT 通讯、SOP 步骤状态机、图片上传与业务报表。这些在工位 TL3576 与中央 TL3576 上实现。

## 1. 它在系统里的位置

| 层 | 职责 | 与本仓库的关系 |
|---|---|---|
| 中央 TL3576（`:8000`） | 网页、FastAPI、SQLite、快照图片、生产统计 | 不直接通讯 |
| 工位 TL3576（Qt 上位机） | 抓图、本地队列、HTTP 上报、SOP 判定 | 未来经 NX 本地 MQTT Broker 取识别结果 |
| **NX（本仓库）** | 模型推理与结果显示 | 目前只吃视频文件，尚未接 MQTT |

三条链路的划分（中央↔NX 管理走 MQTT、NX↔工位 生产走 NX 本地 Broker、工位→中央 数据走 HTTP）见工作区文档 `当前系统架构.md`。**本仓库是这套系统里的算法验证台，不是已经联调完成的 NX 服务。**

## 2. 代码地图

模块关系、编译期包含关系、单路/四路/批量调度流程图、关键常量与踩坑清单集中在 **[cpp_infer/CODE_MAP.md](cpp_infer/CODE_MAP.md)**。先读它再读源码，可以少走一轮弯路——尤其是「探针入口只有 3 行」这种现象。

核心文件：

| 文件 | 作用 |
|---|---|
| `cpp_infer/src/main.cpp` | 唯一主入口，`legacy` / `serial` / `pipeline` / `parallel` 四种调度 |
| `cpp_infer/src/dual_probe.cpp` | 1/2/4 路共享引擎、单推理线程轮询的容量实验 |
| `cpp_infer/src/batch_probe.cpp` | 矩形输入 + 跨工位 batch 的实验入口（当前吞吐最高） |
| `cpp_infer/src/silu_plugin.cu` | TensorRT SiLU 插件，联合 OBB 引擎依赖它 |
| `cpp_infer/CMakeLists.txt` | 默认目标；`SOP_BUILD_ACCEL=ON` 时再引入 `accel.cmake` → `stage2.cmake` |

## 3. 本地资源准备

克隆仓库后需要另行放入以下文件；仓库不含模型下载器，也不含生成 raw ONNX 兼容图的工具。

```text
sop_Monitor/
  test_vedio/07.mp4
  cpp_infer/models/
    yolo_20261007_latest/best.raw_trt2.onnx
    yolox_tiny_8xb8-300e_humanart-6f3252f9.raw_trt.onnx
    rtmw-dw-l-m_simcc-cocktail14_270e-256x192_20231122.onnx
```

前两个是适配 TensorRT 8.5 的兼容图，**不能用原始 end-to-end ONNX 直接替代**。联合模型与批量实验另需 `models/joint_20261018/`、`models/batch_20261020/` 下的引擎，均由脚本在本机生成。

已验证的目标环境：JetPack 5.1.1、CUDA 11.4、TensorRT 8.5.2、OpenCV 4.5.4。**TensorRT 引擎必须在目标 NX 上生成**，不可跨机复制。

## 4. 构建

```bash
cd cpp_infer
bash build_engines.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
```

需要 `sop_nx_accel_probe` / `sop_nx_stage2_probe` / `sop_nx_profile_probe` / `sop_nx_batch_probe` 时：

```bash
cmake -S . -B build -DSOP_BUILD_ACCEL=ON
cmake --build build -j2
(cd build && ctest --output-on-failure)
```

该开关额外要求 `libavformat` / `libavutil` / `libswscale`（pkg-config）；默认构建不需要。

## 5. 运行

单路：

```bash
cd cpp_infer
./build/sop_nx_infer --source ../test_vedio/07.mp4 --model-dir models \
  --obb-stride 2 --person-stride 2 --pose-stride 3
```

无窗口测试：

```bash
./build/sop_nx_infer --source ../test_vedio/07.mp4 --model-dir models \
  --headless --duration 30 --start-sec 180
```

实时并行 + 预览优化：

```bash
./build/sop_nx_infer --source ../test_vedio/07.mp4 --model-dir models \
  --mode parallel --realtime --preview-first
```

显示默认约 1100×620。NX 需先登录 GNOME 桌面；SSH 启动时程序会尝试连接当前用户的 `:0`，也可显式指定 `DISPLAY=:0 XAUTHORITY=/run/user/1000/gdm/Xauthority`。窗口按 `q` 或 `Esc` 退出。

三个 stride 参数都为 1 表示逐帧推理；大于 1 时中间帧复用旧结果。**提高显示吞吐量不等于提高模型实际更新次数。**

两工位共享串行（三模型，左 03 右 07）：

```bash
./build/sop_nx_dual_probe --source ../test_vedio/03.mp4 \
  --second-source ../test_vedio/07.mp4 --model-dir models \
  --start-sec 180 --duration 30 --metrics output/dual.csv
```

两模型（联合 OBB + RTMW）四路：

```bash
./build/sop_nx_joint_probe --source ../test_vedio/03.mp4 \
  --second-source ../test_vedio/03.mp4 --third-source ../test_vedio/07.mp4 \
  --fourth-source ../test_vedio/07.mp4 --model-dir models \
  --yolo-engine models/joint_20261018/joint_fp16.plan \
  --start-sec 0 --duration 300 --metrics output/joint_four_live.csv
```

矩形输入 + batch（当前最快，需已生成 `models/batch_20261020` 下的引擎）：

```bash
./build/sop_nx_batch_probe --source ../test_vedio/03.mp4 \
  --second-source ../test_vedio/03.mp4 --third-source ../test_vedio/07.mp4 \
  --fourth-source ../test_vedio/07.mp4 --det-height 384 \
  --det-batch 4 --pose-batch 4 --batch-wait-ms 5 --duration 60
```

新增实验入口共享同一套模型与阈值，**不替换默认程序**。两工位/四路入口的 `--duration` 必须大于 5 秒，前 5 秒作为启动预热不计入稳态任务率。

## 6. 性能速查

全部为同一台 Xavier NX、`03/03/07/07` 四路文件模拟（除注明单路外），**不是真实 RTSP**。

| 轮次 | 变更 | 结果 |
|---|---|---|
| 10-15 | 三模型单路逐帧 | 8.31–10.75 FPS（180s 位置取 5s）；02 连续 10s 为 12.67 FPS |
| 10-16 | 解码/推理/显示流水线（单路 07） | 27.1–27.4 FPS（无窗口）、24.65（带窗口） |
| 10-16 | `--preview-first` | 单路 23.7–24.0 → 26.26–26.30 FPS，结果指纹一致 |
| 10-16 | 两工位共享串行（三模型） | 10.36 次新识别/秒/路 |
| 10-16 | 四路共享串行（三模型） | 3.76 次/秒/路 |
| 10-19 | 联合两模型替换 OBB+YOLOX | 4.12 次/秒/路（+9.6%），骨架年龄 P95 360→320 ms，约 50% 帧仍被 200ms 规则隐藏 |
| 10-19 | GPU RTMW 预处理 + 硬解 + 复用缓冲 | 四路 11.3 次/秒/路（单任务 22.1 ms 无窗口 / 24.9 ms 带画面） |
| 10-20 | 640×384 矩形输入 + batch 4 | **18.92 次/秒/路**（无画面）/ 16.71（带画面），四路合计 66.8 Hz |

三条必须一起读的限定：

1. **窗口 FPS 是预览消费率，不是识别更新率**；四宫格约 25 FPS 只说明显示跟得上。
2. 分阶段测量显示瓶颈在引擎执行区间（OBB 约 11.9 ms、RTMW 约 7.5 ms），CPU 后处理合计仅 0.63 ms；CUDA Graph 只提升 4–5%。
3. 矩形 + batch 组**没有通过精度验收**：12 张固定帧中 1 张手框数量变化，共同可见点差 P95 由 8.7 px 升至 32 px。速度结论不等于可用配置。

四路各 25 次识别/秒（合计 100）**尚未达到**：按当前串行结构，单任务预算需压到 10 ms。

## 7. 文档索引

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

## 8. 已知边界

- 输入是**视频文件模拟**；真实 RTSP 的网络抖动、断流重连、跨设备时钟同步均未验证。
- 多人场景只输出**得分最高的一个**人体骨架，不是多人姿态产品。
- 无跟踪、无平滑、无运动补偿；快速动作会出现结果错位，靠 200ms 过期隐藏兜底。
- 未做 INT8 校准，未把 CUDA Graph 接入应用，未实现单进程多路作为产品功能。
- 所有性能数字都是单轮、同一段视频、桌面负载未隔离的结果，没有多轮置信区间。

## 9. 提交与推送

目标仓库：`git@github.com:kalapaopao/sop_Monitor.git`（分支 `main`）。提交身份 `kalapaopao <214327724+kalapaopao@users.noreply.github.com>`。

每次提交前先核对：

```bash
git status --short
git diff --cached --stat
git diff --check
```

确认视频、权重、ONNX、TensorRT plan、`build/`、`output/`、日志与凭据均未被暂存。`.gitignore` 只忽略文件，**不会删除 NX 上已存在的这些资源**；如果某个文件曾被跟踪，仅改忽略规则不会自动取消跟踪。

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

## 10. 当前模型文件校验值

```text
8ffd9f6ccd16b0352a37166ef3c86529a04a5ce7c6ebd463aea2fb625eedf46a  cpp_infer/models/yolo_20261007_latest/best.raw_trt2.onnx
12d3dff921d16dd22408b49569cd0f46ef829142e67c5bb694d578c85eabf570  cpp_infer/models/yolox_tiny_8xb8-300e_humanart-6f3252f9.raw_trt.onnx
68db326c5c6f3273456f803eae0faf5440478369f2c3ee67258d2f36512e8d76  cpp_infer/models/rtmw-dw-l-m_simcc-cocktail14_270e-256x192_20231122.onnx
```

这些是 ONNX 源文件的哈希，**不代表 raw TensorRT 兼容图或 `.plan` 的哈希**。

## 11. 许可

Apache License 2.0，见 [LICENSE](LICENSE)。模型权重来自第三方，授权与分发需单独处理，不在本仓库许可范围内。
