# Jetson Orin NX 首轮迁移测试记录

测试环境：JetPack 5.1.1、CUDA 11.4、TensorRT 8.5.2、OpenCV 4.5.4、C++17。首轮范围为 YOLO OBB + YOLOX HumanArt 人体框 + RTMW WholeBody 人体/双手骨架，不包含 TCN 和 Python 推理。

## 引擎

| 引擎 | TensorRT 构建结果 | 备注 |
|---|---|---|
| `build/yolo.plan` | 成功，约 7.0 MB | FP16；使用 decoded raw 图，C++ 做 top-k 和旋转框 NMS |
| `build/yolox.plan` | 成功，约 12 MB | FP16；使用 decoded raw 图，C++ 做人体阈值和 NMS |
| `build/rtmw.plan` | 成功，约 65 MB | FP16；支持运行时 133 个关键点输出 |

TensorRT 8.5.2 无法直接解析两个原始模型的动态 TopK/NMS 尾部，因此保留原始 ONNX，并使用同一模型的 raw 输出图迁移后处理。原始文件没有覆盖。

## 性能

命令：

```bash
./build/sop_nx_infer --source ../test_vedio/02.mp4 --model-dir models --headless --duration 10 --start-sec 180
```

02 视频 180 秒位置连续 10 秒：127 帧，平均 12.669 FPS，平均端到端延迟 67.906 ms，YOLO 有输出 127/127 帧，人体骨架有输出 79/127 帧。该视频源帧率约 25 FPS，当前处理能力低于逐帧实时播放速度。

四个视频在 180 秒位置各运行 5 秒：

| 视频 | 分析 FPS | 平均延迟 | YOLO 输出帧 | 人体骨架输出帧 |
|---|---:|---:|---:|---:|
| 02 | 8.695 | 102.795 ms | 44/44 | 37/44 |
| 03 | 10.748 | 82.531 ms | 54/54 | 54/54 |
| 04 | 10.288 | 85.308 ms | 52/52 | 52/52 |
| 05 | 8.311 | 108.263 ms | 42/42 | 42/42 |

短时结果受 TensorRT 首次加载、缓存和运行片段内容影响；这些数字用于首轮容量评估，不代表检测准确率或完整视频平均值。

## 显示验证

使用 `DISPLAY=:0 XAUTHORITY=/run/user/1000/gdm/Xauthority` 启动窗口成功，OpenCV 窗口链路可用；通过 SSH 转发窗口会明显降低显示测试速度。生成的带框和骨架样例为 `output/verify_02.png`。

## 已知限制

- 当前程序串行执行三个模型；没有线程流水线、帧跳过或 TensorRT CUDA Graph 优化。
- `--record` 保存已处理帧，处理速度低于源帧率时输出时长不会自动补齐源时间轴。
- FP16 引擎构建日志有 TensorRT 的 subnormal weight 警告；引擎已成功构建并运行，尚未做 FP32/FP16 精度对比。
- 代码使用 TensorRT 8.5 的旧 binding API，编译有 deprecated 警告，但在当前 NX 环境已成功链接和运行。
