# SOP 装配识别（离线分析 + 网页回放）

把工位装配视频做成"可回放的动作分析页"：视频上叠加左右手骨架，右侧显示动作语义卡片，
底部用时间轴标出左右手识别片段。目标是复刻 `vedio/reference2.mp4` 那种效果。

当前状态：**可演示**。骨架已经和原视频逐帧对齐（见下文"坐标坑"），网页播放的是
已经烧好骨架的视频；Qwen-VL 的动作语义结果仍是固定 3 秒窗口，尚未接入 SOP 状态机。

📚 **详细文档在 [`docs/`](docs/README.md)**：
- 开发者看 [`docs/给别的开发者的交接文档以及注意事项/`](docs/给别的开发者的交接文档以及注意事项)
  （架构、目录、运行手册、数据结构、踩坑）
- 项目负责人看 [`docs/给我看的文档/`](docs/给我看的文档)
  （现状与决策、参考效果拆解与技术路线、提示词与 SOP 表、清理与审核清单）

## 数据链路

```text
原始视频 (vedio/v2_5m_7m_h264.mp4, 1920x1080, 24.7fps, 120s)
   │
   ├─ YOLO(person) ──> 人物框
   │        └─ RTMW whole-body(133点) ──> 左右手各 21 点（原视频像素坐标）
   │                 └─> web/public/data/local_pose_2min.json   (5 Hz, 600 帧)
   │                          ├─> render_pose_video.py ──> web/public/video/v2_5m_7m_pose.mp4（网页播放这一份）
   │                          └─> 网页底部「左右手识别片段」进度条（按同一阈值切段）
   │
   └─ Qwen-VL(qwen3-vl-plus) 三组证据对照 A/B/C ──> web/public/data/ablation_14_{A,B,C}.json
                                                   └─> 网页右侧卡片 + 工序时间轴
```

## 目录结构

```text
api_teacher/
  run_local_pose.py     人数框 + RTMW 手部关键点，输出 local_pose_2min.json
  render_pose_video.py  把手骨架烧进视频，输出可直接播放的 mp4
  run_ablation.py       Qwen-VL 三组证据对照（A:纯画面 / B:+人物证据 / C:+手部证据）
  run_video.py          早期版本：抽帧调用 Qwen-VL，生成 reference_style.html
  build_dashboard.py    汇总生成 ablation_dashboard.html
  serve_dashboard.py    用本地 HTTP 服务托管 api_runs（端口 8765）
src/fastener_counter.py 紧固件计数（OpenCV 模板/轮廓）
scripts/count_fasteners.py 计数脚本入口
model/
  yolo/best.onnx        人体检测权重（不入库）
  rtmw_wholebody_256x192.onnx  RTMW 全身/手部关键点权重（不入库）
web/                    前端（Vue 3 + Vite）
  src/App.vue           单页应用：视频 / 卡片 / 时间轴 / 左右手识别片段条
  src/style.css         样式（含时间轴与骨架条）
  public/video/         网页播放的视频（不入库，需按下面命令生成）
  public/data/          ablation JSON、姿态 JSON、抽帧图
vedio/                  原始与参考视频（不入库）
clips/                  测试片段（不入库）
api_runs/               历史分析结果（不入库）
tests/                  pytest 用例
```

> `web/dist/`、`web/node_modules/`、`.venv/`、`.ffmpeg*/`、`.backup/`、`api_runs/`、
> `web/web/`（疑似误生成的嵌套副本）都在 `.gitignore` 里。

## 快速开始

前置：Python 环境 `.venv`、Node.js、`ffmpeg`（仓库里可用
`.ffmpeg/ffmpeg-9.0.2-essentials_build/bin/ffmpeg.exe`）。

```powershell
cd E:\study\SOP_train_model_resource\sop_offline_teacher
$py = ".venv\Scripts\python.exe"

# 1) 前端（默认 http://127.0.0.1:5173/，host/port 写在 vite.config.js）
cd web; npm install; npm run dev

# 2) 重新提取手部关键点（约 90 秒 / 600 帧）
& $py api_teacher\run_local_pose.py `
  --video vedio\v2_5m_7m_h264.mp4 `
  --person-model model\yolo\best.onnx `
  --pose-model model\rtmw_wholebody_256x192.onnx `
  --output web\public\data\local_pose_2min.json `
  --start 0 --duration 120 --rate 5

# 3) 重新渲染带骨架的视频（关键点会按相邻两帧插值到 25fps）
& $py api_teacher\render_pose_video.py `
  --video vedio\v2_5m_7m_h264.mp4 `
  --pose web\public\data\local_pose_2min.json `
  --output web\public\video\v2_5m_7m_pose.mp4 `
  --ffmpeg .ffmpeg\ffmpeg-9.0.2-essentials_build\bin\ffmpeg.exe

# 4) Qwen-VL 三组对照（需要 API Key，费用随 stride 上升）
$env:DASHSCOPE_API_KEY="你的 Key"
$env:DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
& $py api_teacher\run_ablation.py vedio\v2_5m_7m_h264.mp4 `
  --observations <观测数据路径> --output api_runs\ablation_14_A --mode A --duration 30 --stride 3 --frames 3
```

## 网页功能

- **视频**：播放 `v2_5m_7m_pose.mp4`，骨架（L 青 / R 橙）已烧进画面，帧帧对齐。
- **右侧卡片**：标准 SOP 描述 / 当前动作 / 左手 / 右手，随视频时间切换到对应 3 秒窗口。
- **底部时间轴**：
  - `工序` 一行是 0–30 秒抽样窗口的等宽分段（占位，后续改成不等长 SOP 片段）；
  - `左右手识别片段` 条与视频同一时间轴（0–119.99 秒）：青色=左手识别到、橙色=右手识别到、
    深灰=未识别，白色播放头随播放移动，点击任意位置可跳转。
- **调试参数**：`http://127.0.0.1:5173/?t=45` 直接定位到第 45 秒，方便截图核对。

## 坐标坑（重要）

`run_local_pose.py` 把人物框按比例缩放后**居中 letterbox** 到 192×256 再送进 RTMW，
因此回算关键点时必须先扣掉 letterbox 的 `left/top` 留边再除以 `scale`。
旧版本漏了这一步，关键点最多偏 57 像素，表现为"骨架贴在画面旁边"。
现在 `crop_person()` 与 `decode()` 用同一组几何参数，成对修改时务必一起改。

## 已知限制

- Qwen-VL 结果只覆盖 0–30 秒（10 个 3 秒窗口），视频是 120 秒，两者时间基准不同。
- 左右手识别片段区分度低：只要检测到人，两只手通常同时被判为有效（覆盖率约 76% / 84%），
  目前表达的是"有没有手"，还表达不了"左手在扶、右手在拧"。
- 模型在遮挡时会给出贴在画面边缘的假手，`keep_hand()` 用"点集范围过小/贴边"两个条件过滤，
  阈值（`min_extent=30`、`border=8`）是经验值，换分辨率需要重调。
- 关键点无长时序平滑，快速动作时仍有轻微抖动（靠渲染插值缓解）。
- 尚未实现：工具/工件的目标检测、SOP 状态机、不等长动作片段合并。

## 约定

- 视频、模型权重、`api_runs/` 等大文件不入库；需要时按上面命令本地生成或单独放置。
- 千万不要把 `DASHSCOPE_API_KEY` 写进项目文件，也不要在日志里打印。
- 网页只读 `web/public` 下的静态资源；换数据不需要改代码。
- 改动 `run_local_pose.py` 的坐标逻辑后，必须重新生成姿态 JSON **和**骨架视频，
  否则网页上的骨架与底部片段条会不一致。
