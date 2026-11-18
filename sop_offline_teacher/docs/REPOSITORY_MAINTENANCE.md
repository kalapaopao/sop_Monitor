# 仓库整理与发布记录

日期：2026-11-18。当前方案：`pred_hmm_full`；远端开发分支：`feat/roi-action`。

## 整理结果

- README 改为当前 MLP + HMM 基线说明，加入效果动图、完整录屏及技术实现入口。
- 新增 `TECHNICAL_IMPLEMENTATION.md`：源码调用链、真实评测口径、复现命令、已知缺陷和部署验收条件。
- 新增 `examples/pred_hmm_full/` 冻结预测、65段后处理和段级评测快照，使后处理不依赖未发布的 `api_runs/`。
- 保留现有脚本、GT、训练所需姿态输入、所有结果 run 和历史实验记录；默认仍为原65段基线。已有严格确认功能仅为独立对照。
- 更新 `.gitignore`，防止嵌套原始视频、模型权重、完整融合中间缓存、生成裁片和原始录屏误提交。
- 核对页保留 HTML 与片段索引，新增 `scripts/restore_review_clips.py`，可从本地源视频按原帧号重建172个实验裁片，不调用 API。该数量属于历史候选展示，不代表172个正确SOP步骤。
- 默认 run 的备注改为明确的本地 MLP + HMM 来源；步骤起止时间、类别、模板内容不变。

## 清出的旧文件

已检查 HTML、源码和文档未再引用旧索引或 max10 路径，且 `clips_original.json` 与当前 `clips.json` 字节一致：

| 旧路径 | 处理 | 理由 |
|---|---|---|
| `web/public/sop-discovery/max10/` | 从有效目录移出，留本地备份 | 已弃用机械10秒裁片，64个派生媒体文件，共31,297,595字节 |
| `web/public/sop-discovery/clips_max10.json` | 从有效目录移出，留本地备份 | 上述旧裁片索引 |
| `web/public/sop-discovery/clips_original.json` | 从有效目录移出，留本地备份 | 与当前索引完全一致 |
| `web/web/` | 从有效目录移出，留本地备份 | 已核实没有任何文件的旧嵌套目录 |
| 远端旧 `api_teacher/run_video.py` | 移除旧入口 | 当前已在 `api_teacher/legacy/run_video.py` 保留历史实现，避免误导读者 |

本地恢复目录：`.backup/repo-review-20261118/retired/`。如需恢复，逐项移回表中原路径；不需要恢复识别数据。一次批量递归清理命令被自动审批拒绝，未执行，最终改用明确绝对路径逐项移入备份；Python缓存保留且不入库。没有删除原始视频、人工标注、权重或有价值的实验记录。

## 媒体发布

`docs/media/sop-demo.mp4` 来自用户提供的 `效果视频.mp4`，保留约66秒完整画面，缩到1280像素宽、25fps、H.264、无声和faststart。约3.5MB。`sop-demo.gif` 为录屏12–20秒，960像素宽、8fps、96色，约2.7MB；`sop-demo.jpg` 为12秒封面。原始约41MB录屏未改动，留在本地。录屏中选中的是 `pred_hmm_confirmed_test`（66段），属于同链路的严格确认对照；项目默认仍为原 `pred_hmm_full`（65段）。

主仓库 README 用可点击的动图链接 MP4，并链接本项目说明。原始工位视频与172个批量裁片不随仓库发布，避免重复保存约200MB派生视频。新克隆必须先按 README 准备对应工位视频，不能直接将录屏当成分析源视频。

## 验证记录

- `npm run build`：通过，Vue / Vite生产构建成功。
- `.venv/Scripts/python.exe -m pytest -q`：未能运行，环境没有pytest；没有为本次整理安装依赖。
- `.venv/Scripts/python.exe -m unittest discover -s tests -v`：现有3项测试全部通过。
- 冻结预测重新后处理：65段，起止时间和类别与原结果一致；段级复测27/37，IoU阈值0.3。
- 重建脚本：全量索引dry-run及单片实际重建、越界路径拒绝和缺源拒绝检查，均通过。
- 文档演示：ffprobe检查编码、时长，OpenCV抽帧检查画面可解码。
- 发布前检查选中文件清单、凭据特征和文件大小；不读取任何`.env`。
- 发布依赖检查补入历史 `hand_fused_clip` 所需的姿态文件；172项全量裁片索引dry-run通过。源视频名称已核对，基线 `vedio/side3.mp4` 与 `v2_full.mp4` 的SHA256一致。
- `git diff --cached --check` 的4处文件末尾空行提示已修正；一次验证脚本因未显式指定UTF-8失败，修正编码后复检。
- 本次未重训MLP，未重跑全片RTMW，未验证NX完整SOP实时链路，也未在新机位重新量化准确率。

## 发布与回退

本地项目位于独立仓库`master`，与主仓库历史不相连。为保留远端NX工程，使用开发分支的隔离检出，将本项目同步到既有`sop_offline_teacher/`路径，只补充根README展示区；不覆盖主仓库其他模块，不强推，不修改main。

本次范围可由开发分支提交的diff单独审阅。恢复文档展示时仅恢复README、技术文档和新增演示素材；恢复发布前项目时在开发分支对本次提交做正常revert，不使用reset或force push。后处理对照不影响默认结果。
