# 21 · 分层架构落定：本地模型管边界与步骤，Qwen 只管片段语义

> 背景：Codex 提出「把 Qwen 放回片段语义补全层，而不是回到每几秒猜六步」。
> 我完全同意，并已落地第一版。本文记录分工、实现与验收方式。
> **一处更正**：本地链路已不是 TCN，而是 MLP+HMM（88.6% > TCN 85.5%）。

---

## 一、最终分工（四个角色，各管一段）

```text
① 本地模型（MLP + HMM + 状态机）
     负责：片段边界、step_id、置信度
     证据：帧级 88.6%（7 周期留一、无泄漏）；段级 F1 0.43

② Qwen-VL（只在片段语义补全时调用）
     负责：action 文字、左右手 {action, holding, target}、state_change、evidence
     不负责：切边界、改 step_id
     成本：每个已确定片段 1 次调用（本片 15 段 → 15 次）

③ 固定 SOP 表（configs/sop_steps.json）
     负责：step_id → 标准 SOP 名称与三要素判据（Qwen 永不生成）

④ 状态机 + 前端
     负责：去抖/合并/unknown 兜底；四卡片 + 三轨时间轴
```

**为什么这样分**（都被实测数据支撑）：

| 做法 | 实测 | 结论 |
|---|---|---|
| Qwen 负责切边界 + 判六步 | clean 18.2% / multi 9.1% | ❌ 已证伪 |
| 本地模型负责切边界 + 判六步 | 帧级 88.6% | ✅ 可行 |
| **Qwen 只描述已切好的片段** | 待验证（本轮实现） | 任务显著简化 |

---

## 二、本轮实现

### 2.1 `scripts/enrich_segments_qwen.py`

对每个已发布 run 里的片段，取段内 2 帧，问 Qwen 五个字段：

```json
{"action":"…",
 "left_hand":{"action":"…","holding":"…","target":"…"},
 "right_hand":{"action":"…","holding":"…","target":"…"},
 "state_change":"none","evidence":"…","consistent":true}
```

**关键设计：`consistent` 字段**
Qwen 被要求判断"画面是否真的像这个 SOP 步骤"。若它说不像 → 自动标 `need_review=true`。
这让 Qwen 从"生成器"变成"复核员"，正好补上本地模型不可解释的短板。

提示词里明确禁止：改 step_id、改边界、编造品牌型号、用英文描述。

**成本（dry-run 实测）**：15 段 → 30 张图 / **15 次调用**。

### 2.2 前端支持结构化左右手

`App.vue` 的左右手卡片现在兼容两种形态：

| 形态 | 来源 | 渲染 |
|---|---|---|
| 字符串 | 本地模型（无左右手语义） | 直接显示，卡片显示"（模型未给出）" |
| `{action, holding, target}` | Qwen 补全后 | 正文=action；小字=`持握：X · 作用于：Y` |

已实测：切到 `p3_mlp_hmm` 时字符串形态正常显示、0 报错；补全后的结构化形态走另一分支。

---

## 三、链路全貌（当前）

```text
离线（建标签与 SOP 表）
  视频 → 10Hz 姿态(body+hands) → 帧级标签（GT 段 + invalid 屏蔽）
       → 训练 MLP（逐帧）→ 统计转移矩阵 → HMM

离线（生成结果）
  MLP+HMM 解码 → 状态机去抖 → segments（step_id + 边界）
       → Qwen 逐段补全（action/左右手/证据）→ segments_enriched.json
       → publish_segments → web/public/data/runs/<name>.json

在线（NX）
  摄像头 → YOLO → RTMW → 153 维特征 → MLP → HMM → 状态机 → segments.json
  （不需要 Qwen，不需要联网）
```

---

## 四、下一步（按 Codex 建议 + 我的实测排序）

| # | 事项 | 状态 | 说明 |
|---|---|---|---|
| 1 | 保留六步唯一契约，不急着并 S6 | ✅ 已按此执行 | S6 仍是独立类，只是 F1=0 |
| 2 | 修 YOLO 召回 / 外观 ROI | ⬜ | 12% 帧无人框；conf 0.25→0.15 待试 |
| 3 | 保存 v3/v4 **概率**再验证集成 | ⬜ | 现在只存 argmax，无法做概率平均 |
| 4 | **Qwen 逐段语义补全** | ✅ **本轮实现** | 15 次调用，需 API Key 实跑 |
| 5 | 左右手 `{action, holding, target}` | ✅ **本轮实现** | 前端已兼容 |
| 6 | 同一份 segments.json 驱动四卡片 | ✅ 已实现 | `p3_mlp_hmm` 已是此结构 |
| 7 | ONNX 导出 + NX 延迟 | ⬜ | MLP 只有 1.4 万参数，预计 <0.1ms/帧 |

---

## 五、待你执行的唯一一步（需要 Key）

```powershell
cd E:\study\SOP_train_model_resource\sop_offline_teacher
$env:DASHSCOPE_API_KEY="你的 Key"

.venv\Scripts\python.exe scripts\enrich_segments_qwen.py `
  --run web/public/data/runs/p3_mlp_hmm.json `
  --frames 2 --out api_runs/p3_enriched --publish-name p3_enriched
```

跑完后网页「结果来源」会多一个 **`p3_enriched`**：
工序轨用本地模型的边界（88.6% 那套），右侧四卡片的文字由 Qwen 补全，
左右手卡片会显示 `持握 / 作用于` 两个新字段——**这就是参考视频的完整形态**。

同时它会输出「多少段被 Qwen 判为与步骤不一致」，这是本地模型的**外部体检指标**。
