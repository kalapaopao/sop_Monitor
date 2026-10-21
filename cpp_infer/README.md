# SOP YOLO + WholeBody C++ TensorRT 测试

> 2026-10-19已核实目标192.168.137.230为Jetson Xavier NX（TensorRT计算能力7.2），不是此前文档沿用的Orin NX。新四分类联合OBB替换旧OBB及YOLOX，接RTMW的独立两模型入口见 [JOINT_MODEL_GUIDE.md](JOINT_MODEL_GUIDE.md)。原三模型入口保留。

这是首轮 NX 迁移目录，只包含：

- YOLO OBB：装配工具和手框；
- YOLOX HumanArt：人体框；
- RTMW WholeBody：人体关键点和双手 21 点；
- 不包含 TCN。

模块关系、编译期包含关系与各入口调度流程图见 [CODE_MAP.md](CODE_MAP.md)；仓库级入口与性能速查见 [../README.md](../README.md)。

## 构建

目标设备是 JetPack 5.1.1 / CUDA 11.4 / TensorRT 8.5.2。先在 NX 上执行：

```bash
./build_engines.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
```

YOLO OBB 原始 end-to-end 导出图在 TensorRT 8.5.2 上会在 TopK 后处理节点失败。`models/yolo_20261007_latest/best.raw_trt2.onnx` 是从同一权重截断到 decoded raw 输出 `[1,8400,8]` 的兼容图；C++ 负责 top-k、类别选择和旋转框 NMS。YOLOX 原始图的动态 TopK/NMS 也被截断为 `models/yolox_tiny_8xb8-300e_humanart-6f3252f9.raw_trt.onnx`，C++ 负责 person 阈值和 NMS。原始 ONNX 文件均保留，没有覆盖。

## 运行

NX 本地桌面显示：

```bash
./build/sop_nx_infer --source ../test_vedio/02.mp4 --model-dir models
```

通过 SSH 启动时，程序会在 DISPLAY 未设置的情况下尝试连接当前用户的 GNOME `:0` 桌面，使用 `/run/user/<uid>/gdm/Xauthority`。需要先在 NX 上登录桌面，窗口显示在 NX 的共享桌面中。已有 DISPLAY 设置会被保留。

也可以显式指定（以下示例对应 nvidia 用户 UID 1000）：

```bash
DISPLAY=:0 XAUTHORITY=/run/user/1000/gdm/Xauthority \
  ./build/sop_nx_infer --source ../test_vedio/02.mp4 --model-dir models
```

SSH 无窗口性能测试：

```bash
./build/sop_nx_infer --source ../test_vedio/02.mp4 --model-dir models --headless --duration 30
```

程序每 30 帧打印一次分析帧数、总推理 FPS 和单帧端到端延迟，结束时打印汇总。窗口按 `q`、`Esc` 退出。

首轮 NX 实测（Jetson Xavier NX，FP16 TensorRT，单路三模型串行）：02–05 视频在 180 秒位置各运行 5 秒，分析速度约 8.31–10.75 FPS；02 视频连续 10 秒为 12.67 FPS、平均端到端延迟 67.9 ms。视频原始帧率约 25 FPS，因此当前版本是“单路可运行但处理速度低于原视频帧率”，尚未达到 25 FPS 实时全帧处理。

`--headless` 只用于性能测试；不带该参数会直接显示窗口。`--record` 可保存带框和骨架的 MP4，按源帧率写入实际显示/处理帧。完整逐帧处理文件不会因为处理慢而自动缩短；使用墙钟时限提前停止、实时模式丢弃预览帧时，记录帧数和输出时长会相应减少。

显示窗口默认限制在约 1100x620，程序仍然使用原始视频帧进行推理；可通过
`--display-width PX --display-height PX` 调整窗口显示大小。`--obb-stride N`、
`--person-stride N`、`--pose-stride N` 可以分别降低三类模型的更新频率并复用上一帧结果，
默认值都是 1，表示每帧推理。

## 可选线程流水线实验

旧命令默认使用 `--mode legacy`。可选模式：

- `--mode serial`：带测量的单线程对照。
- `--mode pipeline`：有界解码、推理和显示队列，三个模型仍串行。
- `--mode parallel`：在 pipeline 基础上，OBB 和 YOLOX→RTMW 分支使用各自的工作线程、模型上下文与 CUDA stream。

不带 `--realtime` 时，按帧编号合并同帧结果，不丢弃输入帧。`--max-frames 330 --warmup-frames 30` 固定读取 330 帧，统计后 300 帧吞吐量；三种模式应使用相同 source/start/stride 比较。

```bash
bash benchmark_07.sh
```

按原视频时间轴播放并异步识别：

```bash
./build/sop_nx_infer --source ../test_vedio/07.mp4 --model-dir models \
  --mode parallel --realtime --obb-stride 1 --person-stride 1 --pose-stride 1
```

实时模式的视频解码和显示不等待每次推理。推理忙时替换尚未开始的旧任务，因此播放 FPS 不等于识别更新 Hz；日志分别报告 inferred/obb_calls/person_calls/pose_calls。结果带源帧编号，只允许画到相同或更晚的帧；默认隐藏源时间年龄超过 200ms 的框或骨架，可用 `--max-result-age-ms` 调整。年龄为 -1 表示尚无有效结果。此模式不包含跟踪、运动补偿或平滑，仍可能存在识别滞后。

实时模式会对每个选中的推理帧重新检测人体框；`person-stride` 只控制离线匹配模式。姿态调用次数按实际人体裁剪计数，多人帧可能多于处理任务数。

`--metrics output/test.csv` 记录帧编号、模型调用次数、源时间结果年龄、解码耗时、推理步骤耗时、显示间隔和结果指纹。无窗口模式的显示间隔实际是消费帧的间隔。`capture_to_display` 是从应用开始读取该帧到准备绘制的时间，不包含远程桌面传输或屏幕呈现延迟；模型加载发生在计时前。指纹用于比较相同帧的解码结果一致性，不能证明识别准确率。

验证队列退出与取消：`(cd build && ctest --output-on-failure)`。

## 架构图与预览优化

完整图解见 [ARCHITECTURE.md](ARCHITECTURE.md)，包含原图尺寸、五线程职责、三个模型输入、帧与结果队列、时间约束以及尚未实现的多路调度设计。

可选的预览优化：

```bash
./build/sop_nx_infer --source ../test_vedio/07.mp4 --model-dir models \
  --mode parallel --realtime --preview-first
```

该选项只改变显示：原图先用双线性插值缩小，再缩放绘制副本的框/关键点并绘制文字。模型仍读取原图，显示插值和文字线条粗细会与原路径不同。省略选项即使用原预览路径。仅接受非legacy、有窗口且不录制的组合；其它组合报错。日志新增 `render_p95_ms`，包含绘制、imshow与waitKey，仍不包含远端实际屏幕呈现。

`bash benchmark_capacity.sh` 比较两种预览路径，再分别启动1、2、3个独立进程模拟多路。每路用同一段07视频、无窗口实时调度；这不是单进程共享模型多路实现，也不能代替真实相机、网络、多画面显示和长时间运行测试。结果见 [OPTIMIZATION_REPORT_20261016.md](OPTIMIZATION_REPORT_20261016.md)。

## 四路共享串行实验：两路03、两路07，从开头读取

保留原实验入口名 `sop_nx_dual_probe`。只提供source和second-source仍运行两路；同时提供third-source和fourth-source则运行四路，四路各自独立解码、定时读取、排队及保存识别结果，共享三个模型引擎。推理线程公平轮询，每个选中的输入运行OBB→YOLOX→RTMW链路（无人框时跳过RTMW）。

```bash
./build/sop_nx_dual_probe --source ../test_vedio/03.mp4 \
  --second-source ../test_vedio/03.mp4 --third-source ../test_vedio/07.mp4 \
  --fourth-source ../test_vedio/07.mp4 --model-dir models \
  --start-sec 0 --duration 300 --metrics output/four_live.csv
```

四宫格上排03/03，下排07/07。默认总窗口1100×620，每格约550×310；格内预留标签区域后，1080p图像约缩为508×286显示，保留宽高比。主线程按各路最高源FPS（本次约25）合批刷新，只重绘有新帧的格子。标签显示工位号、文件名、包含启动段的平均任务Hz、骨架结果年龄及HIDDEN状态。缩小只发生在绘制副本；算法仍从各路原始1920×1080帧分别准备640×640、416×416、人体裁剪192×256输入。窗口FPS是预览消费率；任务Hz才反映新识别频率。

四路仍使用默认200ms结果过期保护，过旧结果会隐藏，未做运动补偿或平滑。`--max-result-age-ms`仅控制显示旧结果的容忍时间，提高它不会提高推理速度。自动运行无窗口/四画面对比：`bash benchmark_four_station.sh`。详见 [FOUR_STREAM_REPORT_20261016.md](FOUR_STREAM_REPORT_20261016.md)。

