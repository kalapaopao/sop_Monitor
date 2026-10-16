# SOP YOLO + WholeBody C++ TensorRT 测试

这是首轮 NX 迁移目录，只包含：

- YOLO OBB：装配工具和手框；
- YOLOX HumanArt：人体框；
- RTMW WholeBody：人体关键点和双手 21 点；
- 不包含 TCN。

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

首轮 NX 实测（Jetson Orin NX，FP16 TensorRT，单路三模型串行）：02–05 视频在 180 秒位置各运行 5 秒，分析速度约 8.31–10.75 FPS；02 视频连续 10 秒为 12.67 FPS、平均端到端延迟 67.9 ms。视频原始帧率约 25 FPS，因此当前版本是“单路可运行但处理速度低于原视频帧率”，尚未达到 25 FPS 实时全帧处理。

`--headless` 只用于性能测试；不带该参数会直接显示窗口。`--record` 可保存带框和骨架的 MP4，但当前记录的是实际处理帧，若处理速度低于源视频帧率，输出时长不会自动补齐源时间轴。

显示窗口默认限制在约 1100x620，程序仍然使用原始视频帧进行推理；可通过
`--display-width PX --display-height PX` 调整窗口显示大小。`--obb-stride N`、
`--person-stride N`、`--pose-stride N` 可以分别降低三类模型的更新频率并复用上一帧结果，
默认值都是 1，表示每帧推理。

