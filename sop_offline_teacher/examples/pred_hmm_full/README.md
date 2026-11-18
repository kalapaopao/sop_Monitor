# 冻结基线

`pred_mlp_hmm.json` 是既有七周期留一预测快照，5,174 个有效帧，记录帧级准确率 0.8862；不包含可部署权重。`segments_proposal.json` 是其原65段后处理结果，`segment_metrics.json` 是2026-11-18的段级复核。

使用方法和评测限制见 [技术实现方案](../../docs/TECHNICAL_IMPLEMENTATION.md)。新增输出请写入 `api_runs/`，不要覆盖冻结输入。快照保留的 `source` 字段是当时的运行路径，不是当前必需的输入位置。
