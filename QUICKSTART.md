# 快速开始（Quick Start）

> 本文档回答一个问题：**在 Jetson Xavier NX 上，怎样把这个项目跑起来。**
>
> 更新日期：2026-10-21　适用版本：目录重构后的仓库根目录

---

## 一、这个项目是什么

`sop_Monitor` 是工业 SOP 视觉监测系统的**推理侧**，跑在 Jetson Xavier NX 上。

它做的事：**多路视频输入 → 检测工具和手 → 提取人体骨架 → 输出结构化结果**。

```
视频（1–4 路）
   ↓ 硬件解码 H.265
   ↓ 联合 OBB 模型（4 类：screwdriver1 / screwdriver2 / hand / person）
   ↓ RTMW 姿态模型（133 个人体关键点）
输出：检测框 + 骨架（后续会接 TCN 判 SOP 步骤）
```

**目前还没有做的**：MQTT 通讯、SOP 状态机、TCN 接入。这些在 `CODE_MAP.md` 第 11 节列为未实现。

---

## 二、三条命令跑起来

### 前提

- 设备已开机、已连网
- 已登录过 NX 桌面（GDM 会话存在）
- `build/` 里已有编译好的程序，`models/` 里已有 `.plan` 引擎

---

### 日常使用（推荐）：三种给参数的方式

```bash
cd /home/nvidia/Music/project/sop_Monitor

# 方式 1：预设（最常用）
bash scripts/run.sh           # 四路
bash scripts/run.sh 2lane     # 两路
bash scripts/run.sh 1lane     # 单路

# 方式 2：直接给视频（位置参数，其余全默认）
./build/sop_nx_batch_probe ../../test_vedio/03.mp4 ../../test_vedio/07.mp4 \
  --duration 40 --headless --model-dir ../models

# 方式 3：预设 + 覆盖某一项
./build/sop_nx_batch_probe --preset 4lane --duration 10
```

**优先级（高到低）**：位置参数 > 命令行选项 > preset 的 YAML > 内置默认值。

**换视频**：改 `config/4lane.yaml` 里的 `sources`，或直接用位置参数。

**常用覆盖**：
```bash
bash scripts/run.sh 4lane --duration 10          # 只跑 10 秒
bash scripts/run.sh 1lane ../../test_vedio/04.mp4  # 换视频
```

### 配置文件（YAML）

| 文件 | 路数 | 视频源 |
|---|---|---|
| `config/4lane.yaml` | 4 | 03 / 03 / 07 / 07 |
| `config/2lane.yaml` | 2 | 03 / 07 |
| `config/1lane.yaml` | 1 | 03 |

格式示例：

```yaml
sources:
  - ../../test_vedio/03.mp4
  - ../../test_vedio/07.mp4
duration: 40
headless: true
det_height: 384
det_batch: 4
pose_batch: 4
model_dir: ../models
```

**路径相对 `build/` 目录**（程序约定从 `build/` 启动）。

---

### 三条命令跑起来（手动敲，不走 preset）

#### 快速验证（不带画面，纯性能）

```bash
cd /home/nvidia/Music/project/sop_Monitor/build

./sop_nx_batch_probe \
  ../../test_vedio/03.mp4 ../../test_vedio/07.mp4 \
  --det-height 384 --det-batch 4 --pose-batch 4 \
  --model-dir ../models --duration 20 --headless
```

**预期输出**：

```
status=complete lane=0 ... lanes=2 ... steady_job_hz=24.99 ...
actual_batch=2 detector_calls=NNN
actual_batch=4 pose_calls=NNN
```

### 看画面

去掉 `--headless`，并设置 DISPLAY：

```bash
cd /home/nvidia/Music/project/sop_Monitor/build
export DISPLAY=:0 XAUTHORITY=/run/user/1000/gdm/Xauthority

./sop_nx_batch_probe \
  ../../test_vedio/03.mp4 ../../test_vedio/07.mp4 \
  --det-height 384 --det-batch 4 --pose-batch 4 \
  --model-dir ../models --duration 30
```

按 `q` 或 `Esc` 退出。

### 四路生产配置

```bash
cd /home/nvidia/Music/project/sop_Monitor/build

./sop_nx_batch_probe \
  ../../test_vedio/03.mp4 ../../test_vedio/03.mp4 \
  ../../test_vedio/07.mp4 ../../test_vedio/07.mp4 \
  --det-height 384 --det-batch 4 --pose-batch 4 --batch-wait-ms 5 \
  --model-dir ../models --duration 40 --headless
```

**预期输出**：每路 `steady_job_hz≈19`，四路合计约 76。

---

## 三、为什么老跑不起来：四个必踩的坑

### 坑 1：`--duration` 必须大于 5

```cpp
// src/config/batch_args.hpp:231
if (a.duration <= 5 || !std::isfinite(a.duration))
    throw std::runtime_error("--duration must be greater than 5 seconds (first 5s warmup)");
```

**内置默认值是 0**，所以不传 `--duration`、preset 里也没写，就会直接报错退出。
前 5 秒是预热，不计入统计。

### 坑 2：`--model-dir` 默认值指向错误位置

从 `build/` 启动时，必须写 `--model-dir ../models`。

不写会去找 `build/models/`，那里没有引擎，报错：

```
Cannot open TensorRT engine: models/batch_20261020/obb_640x384_b1_4.plan
```

### 坑 3：视频路径是 `../../test_vedio/`

`test_vedio/` 在**项目根目录**，不在 `sop_Monitor/` 下。

从 `build/` 出发，正确路径是 `../../test_vedio/03.mp4`。

写 `../test_vedio/` 会报 `Cannot inspect file color metadata` 或 `Cannot open video`。

### 坑 4：不带 `--headless` 必须设 DISPLAY

```bash
export DISPLAY=:0 XAUTHORITY=/run/user/1000/gdm/Xauthority
```

不设会报 `cannot open display`。而且**NX 上必须先登录过桌面会话**，否则 `:0` 不存在。

---

## 四、哪个程序该用哪个

`build/` 下有 7 个可执行文件，但**日常只用 `sop_nx_batch_probe`**。

| 程序 | 用途 | 状态 |
|---|---|---|
| **`sop_nx_batch_probe`** | **640×384 + batch 4，当前最快** | ✅ **推荐** |
| `sop_nx_joint_probe` | 4 类联合 OBB + RTMW，单帧串行 | 保留（参考实现） |
| `sop_nx_infer` | 单路，4 种调度模式（legacy/serial/pipeline/parallel） | 保留（教学用） |
| `sop_nx_dual_probe` | 1/2/4 路共享串行 | 旧版 |
| `sop_nx_accel_probe` | 硬解 + CUDA 预处理 | 已并入 batch |
| `sop_nx_stage2_probe` | GPU 预处理 + IO 复用 | 已并入 batch |
| `sop_nx_profile_probe` | 分阶段计时 | 诊断用 |

**判断依据**：`sop_nx_batch_probe` 四路 19 Hz，其他入口在 4–11 Hz 量级。

---

## 五、`sop_nx_batch_probe` 参数速查

> 现在的参数系统有三层：**位置参数 > 命令行选项 > preset 的 YAML > 内置默认值**。

### 视频源（位置参数，最常用）

直接写在命令后面，1–4 个：

```bash
./sop_nx_batch_probe 03.mp4                     # 1 路
./sop_nx_batch_probe 03.mp4 07.mp4              # 2 路
./sop_nx_batch_probe 03.mp4 03.mp4 07.mp4 07.mp4  # 4 路
./sop_nx_batch_probe rtsp://192.168.1.100:554/s1   # 相机
```

**超过 4 个会直接报错**：`At most 4 sources are supported`。

也可以用显式别名（老写法，仍然可用）：

| 别名 | 说明 |
|---|---|
| `--source V` | 第 1 路 |
| `--second-source V` | 第 2 路 |
| `--third-source V` | 第 3 路（必须同时给第 4 路） |
| `--fourth-source V` | 第 4 路 |
| `--single-source` | 显式声明单路 |

### 预设

| 参数 | 说明 |
|---|---|
| `--preset NAME` | 读取 `config/NAME.yaml`，依次尝试 `../config/`、`config/`、当前目录 |

### 常用选项

| 参数 | 内置默认 | 说明 |
|---|---|---|
| `--duration SEC` | 0（必须显式给） | **必须 > 5**，前 5 秒是预热 |
| `--model-dir DIR` | `models` | **从 build/ 启动时必须写 `../models`** |
| `--headless` | 关 | 不显示画面 |
| `--det-height 384\|640` | **384** | 384 是实测最优 |
| `--det-batch 1\|2\|4` | **4** | 4 是实测最优 |
| `--pose-batch 1\|2\|4` | **4** | b2 会慢 8.7% |
| `--batch-wait-ms MS` | 5 | 实测 0/5/10 无差异，不用调 |
| `--display-width/height` | 1100 / 620 | 窗口尺寸 |
| `--max-result-age-ms MS` | 200 | 超过这个年龄的骨架不画 |
| `--start-sec SEC` | 0 | 从视频第几秒开始 |
| `--metrics CSV` | — | 逐帧结果年龄明细 |
| `--batch-metrics CSV` | — | 逐组耗时明细 |
| `--sample-dir` / `--validation-dir` | — | 批量校验模式（必须成对） |

### 引擎覆盖（一般不用）

| 参数 | 说明 |
|---|---|
| `--yolo-engine PLAN` | 覆盖 OBB 引擎，默认按 `--det-height` 自动选 |
| `--rtmw-engine PLAN` | 覆盖姿态引擎，默认按 `--pose-batch` 自动选 |

### 不支持的参数

这些在 `batch_probe` 里**不再存在**，传了会直接报 `Unknown argument`：

```
--record  --realtime  --max-frames  --warmup-frames  --preview-first
--obb-stride  --person-stride  --pose-stride  --yolox-engine
--workers  --mode  --audit-dir
```

这是相对旧版本的**行为变化**：以前是“解析通过、初始化完 CUDA 后才说不支持”，现在启动就报错。


---

## 六、重新编译

### 只改 C++ 源码（最常见）

```bash
cd /home/nvidia/Music/project/sop_Monitor/build
make -j2 sop_nx_batch_probe
```

### 改了 CMake 配置（新增文件、改目标）

```bash
cd /home/nvidia/Music/project/sop_Monitor/build
cmake .. && make -j2
```

### 完整构建

```bash
cd /home/nvidia/Music/project/sop_Monitor
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSOP_BUILD_ACCEL=ON
cmake --build build -j2
cd build && ctest --output-on-failure
```

**预期**：`5/5 tests passed`。

### 重新生成引擎（耗时约 10 分钟）

```bash
cd /home/nvidia/Music/project/sop_Monitor
bash scripts/build/build_engines.sh                        # 旧三模型
bash scripts/build/build_joint_engine.sh                   # 联合 OBB
python3 scripts/build/build_batch_engine.py --help         # batch 引擎
```

---

## 七、目录结构

```
sop_Monitor/
├── CMakeLists.txt          CMake 入口
├── README.md               详细说明
├── QUICKSTART.md           本文档
├── config/                 ★ 运行配置（YAML 预设）
│   ├── 4lane.yaml
│   ├── 2lane.yaml
│   └── 1lane.yaml
├── src/                    C++/CUDA 源码，按模块划分
│   ├── core/               main.cpp（引擎封装+前后处理）、队列、计时
│   ├── capture/            视频采集与硬解
│   ├── preprocess/         GPU 预处理 kernel（letterbox/warp/色转）
│   ├── models/             模型后处理（OBB 解码/NMS、SimCC、批量）
│   ├── pipeline/           调度、批处理
│   ├── probes/             6 个可执行入口
│   ├── config/             参数系统（BatchArgs + YAML）
│   └── silu_plugin.cu      自定义 SiLU 插件
├── tests/                  7 个测试
├── tools/                  5 个引擎构建辅助脚本
├── cmake/                  accel.cmake、stage2.cmake
├── models/                 模型 JSON + .plan 引擎（不入库）
├── docs/
│   ├── architecture/       ARCHITECTURE.md、CODE_MAP.md
│   ├── guides/             5 个专题指南
│   ├── reports/            9 个测试报告
│   └── media/              README 用的 GIF/MP4/PNG
├── scripts/
│   ├── run.sh              ★ 一键启动（读 config/*.yaml）
│   ├── build/              引擎构建 + 模型导出
│   ├── benchmark/          基准测试 + 录屏
│   ├── analysis/           数据分析和报告生成
│   └── docs/               架构图生成
├── build/                  编译产物（不入库）
└── output/                 运行结果（不入库）
```

**重要**：所有脚本都要从仓库根目录或其子目录运行，脚本会自己 `cd` 到仓库根目录。

---

## 八、视频源

`test_vedio/` 在**项目根目录**：

| 文件 | 分辨率 | fps | 说明 |
|---|---|---|---|
| `02.mp4` ~ `09.mp4` | 1920×1080 | 25 | H.265 编码，测试用 |

替换成真实相机时，把 `--source` 换成 RTSP 地址即可，例如：

```bash
--source rtsp://192.168.1.100:554/stream1
```

**注意**：RTSP 的网络抖动和断流重连**尚未验证**，目前只在文件上测过。

---

## 九、常见报错对照表

| 报错 | 原因 | 解决 |
|---|---|---|
| `--duration must be greater than 5 seconds` | 没传 `--duration` 或 ≤5 | 加 `--duration 40` |
| `Cannot find preset 'xxx'. Tried: ...` | preset 名字错或 YAML 不在 | 检查 `config/` 下的 `.yaml` 文件名 |
| `Cannot read preset xxx: bad file` | YAML 语法错 | 检查缩进、`sources` 是否是列表 |
| `Cannot open TensorRT engine: models/...` | `--model-dir` 没写或写错 | 写 `--model-dir ../models` |
| `Cannot inspect file color metadata` | 视频路径错 | 用 `../../test_vedio/xx.mp4` |
| `Cannot open video` | 同上 | 同上 |
| `cannot open display` | 没设 DISPLAY | `export DISPLAY=:0 XAUTHORITY=/run/user/1000/gdm/Xauthority` |
| `SiLU plugin registration failed` | 插件没编译 | `make sop_silu_plugin` |
| `--det-width must be 640 and --det-height 384 or 640` | 分辨率参数非法 | 只用 640×384 或 640×640 |
| `At most 4 sources are supported` | 位置参数超过 4 个 | 最多 4 路 |
| `Unknown argument: --xxx` | 传了 batch_probe 不支持的参数 | 见第五节“不支持的参数” |
| `Batch probe requires one worker, CUDA preprocessing and reused I/O` | preset 里把 `preprocessing` 等改成了非默认值 | 改回 `cuda` / `on` |

---

## 十、性能基线（实测）

同一台 Xavier NX，四路 03/03/07/07，`--det-height 384 --det-batch 4 --pose-batch 4`：

| 场景 | 每路 Hz | 合计 | 备注 |
|---|---:|---:|---|
| 4 路 headless | **19.06** | 76.3 | 生产口径 |
| 4 路 带画面 | 16.73 | 66.9 | 显示要占 CPU |
| 2 路 headless | 24.995 | 50.0 | 撞 25 fps 源上限 |
| 1 路 headless | 24.999 | 25.0 | 同上 |

**2 路和 1 路已经跑满源帧率，只有 4 路还有优化空间。**

单组 4 帧耗时构成：OBB 31.3 ms + RTMW 21.2 ms + 上传 4.0 ms ≈ 52.5 ms。

---

## 十一、相关文档

| 想了解 | 看哪里 |
|---|---|
| 代码架构和模块关系 | `docs/architecture/ARCHITECTURE.md` |
| 模块划分、编译期宏、调度流程 | `docs/architecture/CODE_MAP.md` |
| 四类联合 OBB 模型说明 | `docs/guides/JOINT_MODEL_GUIDE.md` |
| batch 批处理实验 | `docs/guides/BATCH_GUIDE.md` |
| 硬件加速解码 | `docs/guides/ACCEL_GUIDE.md` |
| GPU 预处理 | `docs/guides/STAGE2_GUIDE.md` |
| 分阶段性能分析 | `docs/guides/PROFILE_GUIDE.md` |
| 全部测试报告 | `docs/reports/` |
