# SOP Monitor - Jetson Orin NX

C++17 / TensorRT FP16 单路视频识别验证：YOLO OBB 工具与手框、YOLOX 人体检测、RTMW 人体及双手关键点。不包含 TCN。

## 仓库内容

- `cpp_infer/src/main.cpp`：实际编译入口。
- `cpp_infer/CMakeLists.txt`：CMake 构建配置。
- `cpp_infer/build_engines.sh`：在 NX 本机生成三个 TensorRT 引擎。
- `cpp_infer/models/active_yolo.json`：原 YOLO 模型来源与校验信息，不代表 raw TensorRT 兼容图的哈希。
- `cpp_infer/README.md`：运行参数说明。
- `cpp_infer/NX_TEST_REPORT.md`：首轮历史测试记录，部分限制描述早于后续隔帧和窗口修改。

视频、模型权重、TensorRT 引擎、编译目录、运行日志和备份均不上传 Git；`.gitignore` 仅忽略文件，不会删除 NX 上的这些资源。

## 本地资源准备

克隆仓库后需要另行放入以下文件；仓库不含模型下载器或自动生成 raw ONNX 兼容图的工具。

```text
sop_Monitor/
  test_vedio/07.mp4
  cpp_infer/models/
    yolo_20261007_latest/best.raw_trt2.onnx
    yolox_tiny_8xb8-300e_humanart-6f3252f9.raw_trt.onnx
    rtmw-dw-l-m_simcc-cocktail14_270e-256x192_20231122.onnx
```

前两个为适配 TensorRT 8.5 的兼容 ONNX，不能用原始 end-to-end ONNX 直接替代。可从现有 NX 部署单独复制这些模型。模型的授权和分发方式需要单独处理。

已验证的目标环境：JetPack 5.1.1、CUDA 11.4、TensorRT 8.5.2、OpenCV 4.5.4。TensorRT 引擎在目标 NX 上生成。

```bash
cd cpp_infer
bash build_engines.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
```

## 运行 07 视频

```bash
cd cpp_infer
./build/sop_nx_infer --source ../test_vedio/07.mp4 --model-dir models \
  --obb-stride 2 --person-stride 2 --pose-stride 3
```

显示默认约 1100x620。NX 需已有登录的 GNOME 桌面；在 SSH 中未设置 DISPLAY 时程序尝试连接当前用户的 `:0` 桌面。窗口按 `q` 或 `Esc` 退出。

三个 stride 参数均设为 1 表示逐帧推理；大于 1 时中间帧复用旧识别结果。提高显示吞吐量不等于提高模型实际更新次数。

无窗口测试：

```bash
./build/sop_nx_infer --source ../test_vedio/07.mp4 --model-dir models \
  --headless --duration 30 --start-sec 180
```

## 上传 GitHub 前

目标仓库：git@github.com:kalapaopao/sop_Monitor.git。每次提交前先核对 git status 和 git diff --cached --stat，确认视频、权重、引擎、日志及凭据均未被暂存。

如未来曾经暂存或跟踪了被忽略的文件，仅修改 `.gitignore` 不会自动取消跟踪，需要先检查暂存区，再按明确路径处理。

## Required model SHA256 on the current NX

```text
8ffd9f6ccd16b0352a37166ef3c86529a04a5ce7c6ebd463aea2fb625eedf46a  cpp_infer/models/yolo_20261007_latest/best.raw_trt2.onnx
12d3dff921d16dd22408b49569cd0f46ef829142e67c5bb694d578c85eabf570  cpp_infer/models/yolox_tiny_8xb8-300e_humanart-6f3252f9.raw_trt.onnx
68db326c5c6f3273456f803eae0faf5440478369f2c3ee67258d2f36512e8d76  cpp_infer/models/rtmw-dw-l-m_simcc-cocktail14_270e-256x192_20231122.onnx
```
