# 四路硬件解码、CUDA 预处理和并发验证（2026-10-19）

## 目标与实施范围

在 NX 上比较相同的联合 OBB + RTMW、03/03/07/07 四路视频，保留旧入口，
验证硬解、CUDA 预处理、1/2/4 个推理线程的实际效果。设备确认是 **Xavier NX**，
不是 Orin NX。没有安装依赖、修改功耗/频率、改网络或远程桌面、读取 Git 历史、
提交或推送代码。旧 demo 仅作为架构参考，没有修改或运行其业务服务。

最终保留两份 engine，共享权重；每个工作线程有独立 context、stream 和缓冲区。
固定工位归属确保同一工位最多一个在途任务；各工位仍保存最新帧、帧号和独立结果。
OBB → RTMW 的依赖不变；统一主线程显示，200 ms 过期隐藏规则不变。
根据性能实测，默认采用 **一个推理线程**，2/4 线程仅保留为可选实验。

## 修改、新增和构建产物

本轮修改：

- `src/main.cpp`：engine 共享所有权、独立执行上下文、可选 GPU OBB 输入。
- `src/dual_probe.cpp`：加速目标专用参数、硬解捕获、固定分组的工作线程。
- `src/joint_detector.hpp`：可选 CUDA letterbox；CPU 路径保留。
- `CMakeLists.txt`：新增默认关闭的 `SOP_BUILD_ACCEL` 构建选项。

本轮新增：

- `src/accel_probe.cpp`：独立加速入口。
- `src/accel_preprocess.hpp`、`src/accel_preprocess.cu`：带行跨度的 CUDA 预处理。
- `src/hardware_capture.hpp`：H.265 硬解、I420 下载、FFmpeg 颜色转换和原捕获对照。
- `tests/accel_preprocess_test.cpp`：尺寸、行跨度、FP32/FP16 数值检查。
- `tests/accel_model_test.cpp`：前处理、解码结果及并发上下文一致性检查。
- `tests/accel_decode_test.cpp`：逐像素解码对照与可选 seek 诊断。
- `accel.cmake`：加速目标和现有媒体库链接。
- `benchmark_accel.py`：顺序启动 C++ 对照组、采集指标与系统资源；不是 Python 推理。
- `ACCEL_GUIDE.md`：详细架构流程图、启动与回退方法。
- 本报告。

没有删除、移动或重命名现有项目文件。使用原 `build/`，生成新可执行文件、
CUDA 静态库、CTest 结果和 `output/accel_20261019/` 下的 CSV、JSON、日志及核对图。
原模型 plan、视频、旧入口和之前未提交的功能均保留。

## 输入与一致性验证

源图保持 **1920×1080**；四宫格总窗口 1100×620。联合 OBB 输入仍是
1×3×640×640，RTMW 输入仍是 1×3×256×192。预览缩小不改变人体裁剪所用源图。
RTMW 仍使用原 CPU 仿射变换和归一化，本轮没有加入跟踪、平滑或运动补偿。

- CUDA 数值测试包含 1920×1080、321×199、193×511、1×1、非连续行跨度、FP32/FP16。
  FP32 在 1080p 样例上误差为 0；奇数尺寸最大约一个 8-bit 像素级；包含 FP16 舍入的
  全部样例最大归一化误差约 0.004163，满足预设上界。
- 旧核对图四张上，CPU/CUDA 检测数量、置信度和框角点一致。
- 03、07 各连续比较 61 帧，最终硬解+libswscale 与原 OpenCV/FFmpeg BGR 图像逐像素一致。
- 两视频各第 0、30、60 帧，六组 CPU/硬解图在同一 TensorRT 模型上结果一致；
  四上下文同时处理不同输入、各重复三次，结果未出现串路或几何偏差。
- 以上只是数值/工程一致性，不是有标注数据的准确率评估，也不解决此前联合模型
  TensorRT 与 ONNX 的个别手框置信度差异。

## 失败、调整及警告

1. 初次测试目标编译因 `unsigned char**` 传入 `cudaMalloc(void**)` 失败，修正显式转换后编译通过。
2. 当前 CMake 3.16 不支持所用的 `ctest --test-dir` 方式：出现“No tests were found”。
   改在 build 目录执行 CTest，实际执行三项测试并通过，未将空测试视作通过。
3. 初始 BGRx 转换在 07 第零帧产生 MAE≈13.08、最大通道差 92，放弃该路径。
4. I420 + videoconvert + BT.601 后，MAE≈1.17、最大差 6，但人体框角点差约 5.59 px，
   超过事先设置的 3 px 核对阈值，未降低阈值放行。改为 I420 + libswscale 后误差归零。
5. `gst-launch` 的单 buffer 诊断没有按预期退出，已中断该诊断进程；不作为性能结果。
6. 文件复制时间与 NX 时间有差异，为防旧测试二进制被复用，显式更新时间并重新构建。
   最终日志明确包含四上下文并发检查。
7. 原 OpenCV/GStreamer 组合的 180 秒 seek 测试失败，伴有 demux 错误。
   加速目标对硬解的非零 `--start-sec` 提前报错，指定 `--decoder opencv` 可使用原 seek 路径。
8. 保留并如实记录 TensorRT 旧 API 弃用、嵌入入口未使用函数、GStreamer 时长/位置查询、
   GTK canberra 模块和 libswscale 无加速 BGR 转换实现等警告。逐帧读取和窗口验证单独检查。
9. 本轮部分编辑定位检查未匹配，未产生文件改动；后续根据实际内容完成修改。

## 性能测试方法

每组独立运行相同四路视频的前 30 秒，前 5 秒作为预热，不运行其他构建或推理测试。
先测无窗口五组，再测四宫格四组。源视频按源帧率读取；处理不过来时覆盖旧待处理帧。
显示帧率、新识别任务频率、源帧结果年龄分别记录。年龄按文件帧号/帧率估算，
不是实测摄像头端到端延迟。每个新任务包含联合检测，以及存在人体时的 RTMW。

只进行本轮顺序对照与后续短时稳定性检查，未做多轮随机顺序统计，不代表硬件极限。
系统资源记录包含桌面、远程显示和其他既有后台服务，不是当前进程独占显存。

| 组别 | 每路新任务 Hz（四路平均） | 各路结果年龄 P95 范围 ms | 隐藏比例范围 | 四宫格呈现 FPS | 系统 RAM 峰值 MB |
|---|---:|---:|---:|---:|---:|
| headless_baseline | 5.719 | 240–240 | 32.5–33.4% | 0.00 | 3694 |
| headless_hw_cpu1 | 6.340 | 200–200 | 0.3–23.5% | 0.00 | 3762 |
| headless_hw_cuda1 | 9.470 | 160–160 | 0.0–0.5% | 0.00 | 3752 |
| headless_hw_cuda2 | 9.190 | 200–200 | 0.3–7.5% | 0.00 | 3810 |
| headless_hw_cuda4 | 5.940 | 360–360 | 80.8–99.4% | 0.00 | 3887 |
| gui_baseline | 3.430 | 360–400 | 61.7–63.6% | 22.90 | 3758 |
| gui_hw_cuda1 | 7.087 | 160–200 | 0.0–5.9% | 24.72 | 3760 |
| gui_hw_cuda2 | 5.929 | 240–280 | 21.0–45.1% | 24.66 | 3825 |
| gui_hw_cuda4 | 4.168 | 480–521 | 97.4–100.0% | 24.72 | 3946 |

本轮带画面单线程加速版相对旧版约提升 106.6%；无窗口约提升 65.6%。带画面总新任务约 28.35 次/秒，仍低于四路各25次所需的100次/秒。所有九组 future_results=0、退出码0，均未使用交换分区。


## 构建、运行与测试状态

最终构建、CTest、参数回归及稳定性检查以报告末尾的最终执行记录为准。
真实摄像头、RTSP 断流恢复、故障注入、Windows 运行和长期运行均未验证。

主要查阅/检查：`git status --short`、`git diff --check`、源文件和构建脚本、设备树、
`pkg-config`、`gst-inspect-1.0`、`gst-discoverer-1.0`、`ffprobe`、`ps` 和实验日志。
未读取相机密码配置和 Git 历史。

主要执行命令：

```bash
cmake -S . -B build -DSOP_BUILD_ACCEL=ON
cmake --build build -j2
(cd build && ctest --output-on-failure)
./build/accel_preprocess_test
./build/accel_model_test models/joint_20261018/joint_fp16.plan IMAGE...
./build/accel_model_test models/joint_20261018/joint_fp16.plan --decode-pairs CPU_IMAGE HW_IMAGE ...
./build/accel_decode_test VIDEO EXISTING_OUTPUT_DIR
./build/accel_decode_test VIDEO EXISTING_OUTPUT_DIR --seek
python3 benchmark_accel.py --phase headless --duration 30
python3 benchmark_accel.py --phase gui --duration 30
python3 benchmark_accel.py --phase soak --duration 120 --workers 1
```

还执行了必要的单目标构建、Python/OpenCV 解码诊断、SCP/tar 文件传输和生成日志。
失败命令及处置见上节；成功与失败的原始日志保留在 output 中。

## 兼容性与决策

原可执行文件的命令、模型路径、阈值、输出数据格式和目录不变；新增入口才接受
`--decoder`、`--preprocess`、`--workers`。原引擎加载改为共享所有权，单实例仍使用
原执行路径。加速构建选项默认关闭，旧目标不因新媒体开发库缺失而强制失败。
主要平台验证是 Linux/ARM64/Xavier NX；没有做 Windows 或其他 Jetson 架构运行验证。

本轮对“增加线程必然加速”的假设进行了实测：四路下 2/4 推理线程未胜过 1 线程，
因此采用硬解+CUDA预处理+单推理线程作为默认配置，并保留并发开关供后续不同负载测试。
这不是否定取流/显示分线程：它们仍然独立运行。并发结果差异的具体 GPU/CPU 根因尚未
通过 profiler 定位，不能仅从吞吐推断 GPU 一定满载。

## 风险、验收与回退

- 本轮输入是四路文件模拟，不等于完成四台实际摄像头部署；新路径目前只验证 H.265 MP4。
- RTSP 支路仍假定 BT.709 limited-range，未完成通用相机颜色元数据处理和重连恢复。
- 硬解非零时间跳转不支持，使用 OpenCV 捕获作为保留方案。
- 没有零拷贝；颜色转换和 RTMW 预处理仍会消耗 CPU；四路各 25 次新识别/秒的目标未达到。
- 队列覆盖和 200 ms 过期规则有意牺牲部分帧的识别完整性，不能把流畅预览当成逐帧分析。
- BT.709/BT.2020/HDR、奇数解码布局和不同摄像头尚未做同等像素一致性验证。
- C++17/模型输入输出契约保持原样；并发仅共享权重，不共享可变推理状态。

按照 `ACCEL_GUIDE.md` 的启动命令，查看四宫格、新识别 Hz、结果年龄及日志；Q/Esc 退出。
图像仍是原分辨率取流，显示缩放；不应将四路 Hz 相加后当作单路 Hz。

最简单回退是运行原 `./build/sop_nx_joint_probe`。若需撤销本轮源代码集成，
从持久备份 `output/accel_20261019/before/`（另有 `/tmp/sop-accel-before-20261019/` 副本）恢复 main.cpp、dual_probe.cpp、joint_detector.hpp
到 src/，恢复 CMakeLists.txt 到 ，再重新构建。备份包含之前未提交的两模型工作；
不要对整个仓库执行 reset/clean。新文件可保留为未引用文件，视频、模型和旧报告不需要恢复。
使用 `git diff` 查看本轮集成时应注意工作区本来就有前一轮未提交改动，不能全部归于本轮。


## 最终执行记录

- 最终全量构建退出码0；原入口与加速入口均编译成功。
- 在 build 目录实际执行的 CTest 三项全部通过。
- 最终 C++ 解码对照03/07各61帧均逐像素一致，退出码0。
- workers=3、硬解start-sec=180均按设计返回1；OpenCV模式180秒跳转冒烟返回0。
- 数值与模型/四上下文一致性检查通过；git diff --check通过；没有自动提交或推送。
- 120秒四路GUI检查退出码0，平均每路7.716次新任务/秒；各路P95约160ms；future_results=0。
- 该检查系统RAM峰值3770MB，Swap峰值0MB，GPU最高47.0℃。这不是长期无泄漏证明。
- 该检查四宫格呈现约24.83FPS。每路隐藏比例：0.017, 0.017, 0.000, 0.000。
- 默认加速入口已重新启动四路300秒展示，日志live.log/live.csv，方便远程查看；Q/Esc退出。

- 手动核对：实际四宫格截图为1100×620，四路画面、检测框/骨架及各路任务Hz/结果年龄可见。截图为live-preview.png；顶部FPS表示预览节奏，不是模型更新频率。截图辅助探测最初未指定DISPLAY出现连接提示，最终明确指定:0和原Xauthority后捕获成功。
