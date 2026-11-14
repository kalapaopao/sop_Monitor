# API 教师视频分析

本目录是项目当前唯一的视频分析入口。它使用阿里云百炼兼容 OpenAI 的视觉模型 `qwen3-vl-plus` 分析视频窗口，并生成网页回放报告。

运行前在 PowerShell 设置 `DASHSCOPE_API_KEY` 和 `DASHSCOPE_BASE_URL`，然后执行：

```powershell
$py = ".venv\Scripts\python.exe"
& $py api_teacher\run_video.py `
  "clips\v2_5m_7m.mp4" `
  --duration 120 `
  --stride 2 `
  --frames 4 `
  --output api_runs\v2_5m_7m_clip
```

参数说明：

- `--start`：源视频起始秒数
- `--duration`：分析时长
- `--stride`：相邻窗口的间隔秒数
- `--frames`：每个窗口发送的画面数量
- `--output`：结果目录
- `--model`：模型名称，默认 `qwen3-vl-plus`

输出包含原始 JSON、抽取帧和网页报告。API Key 只通过环境变量提供，不写入文件。

## 三组证据对照

`run_ablation.py` 一次只运行一组输入，避免本地模型或请求并发占用过多资源。A 组只发送画面，B 组加入人员存在证据，C 组加入人员、手部检测和 RTMW 手部关键点证据。

示例：

```powershell
& $py api_teacher\run_ablation.py `
  "E:\study\SOP_train_model_resource\resource\video\vedio\unused\14.mp4" `
  --observations "E:\study\SOP_train_model_resource\project\experiments\20261028_roi_pose_tcn_14_latest\features\14\observations.jsonl.gz" `
  --duration 30 `
  --stride 3 `
  --frames 3 `
  --mode A `
  --output api_runs\ablation_14_30s
```

分别把 `--mode` 改成 `A`、`B`、`C`，并使用不同的输出目录，例如 `ablation_14_A`、`ablation_14_B`、`ablation_14_C`。每次完成后打开对应目录中的 `ablation_report.html`。
