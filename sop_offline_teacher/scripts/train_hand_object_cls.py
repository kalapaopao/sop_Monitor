"""在手周 ROI 数据集上微调 MobileNetV3-Small，做「手里是什么」的物件分类。

目的
----
验证一个具体问题：**手周裁图里到底有没有足够信息，把物件分开？**
· 廉价特征的最近质心基线 = 47.9%（随机 20%）——有信息，但弱
· 本脚本用 ImageNet 预训练的 MobileNetV3-Small 微调，看能不能明显拉开。
· 若能，它就是「物件身份」通道，专门解当前最差的两对混淆：S1↔S2、S3↔S4。

口径（不许含糊）
----------------
· **留一周期（leave-one-cycle-out）**：训练集与测试集来自不同装配周期，
  不是随机切帧——否则相邻帧几乎一样，会严重高估。
· 单工人、单产品、单机位。跨工人未验证。

用法
----
    .venv\\Scripts\\python.exe scripts\\train_hand_object_cls.py \
        --data api_runs/hand_objects --out api_runs/obj_cls \
        --weights E:/study/SOP_train_model_resource/project/models/roi_action_20261027/mobilenet_v3_small-047dcff4.pth
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import cv2


class Crops(Dataset):
    def __init__(self, items, root, size, train):
        self.items = items
        self.root = Path(root)
        self.size = size
        if train:
            self.tf = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.85, 1.18)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.25, 0.25, 0.2, 0.05),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        img = cv2.imread(str(self.root / "crops" / it["file"]))
        if img is None:
            img = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.tf(img), it["_y"]


def make_model(n_cls, weights):
    m = mobilenet_v3_small(weights=None)
    if weights and Path(weights).exists():
        sd = torch.load(str(weights), map_location="cpu", weights_only=True)
        try:
            m.load_state_dict(sd, strict=True)
            print(f"已加载 ImageNet 权重：{Path(weights).name}")
        except Exception as e:
            print(f"权重键不匹配，改为部分加载：{e}")
            miss, unexp = m.load_state_dict(sd, strict=False)
            print(f"  缺失 {len(miss)} 键，多余 {len(unexp)} 键")
    else:
        print("没有预训练权重，从零训练（效果会更差，请以此判断下限）")
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, n_cls)
    return m


def evaluate(model, loader, device):
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device))
            preds += out.argmax(1).cpu().tolist()
            gts += y.tolist()
    preds, gts = np.array(preds), np.array(gts)
    return float((preds == gts).mean()), preds, gts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--folds", default="all", help="all 或逗号分隔的周期号，如 0,3")
    a = ap.parse_args()

    idx = json.loads((a.data / "index.json").read_text(encoding="utf-8-sig"))
    items = [x for x in idx["items"] if (a.data / "crops" / x["file"]).exists()]
    classes = sorted({x["obj_class"] for x in items})
    c2i = {c: i for i, c in enumerate(classes)}
    for x in items:
        x["_y"] = c2i[x["obj_class"]]
    cycles = sorted({x["cycle"] for x in items})
    folds = cycles if a.folds == "all" else [int(v) for v in a.folds.split(",")]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"样本 {len(items)} ｜ 类 {classes} ｜ 周期 {cycles} ｜ 设备 {device} ｜ size {a.size}")

    a.out.mkdir(parents=True, exist_ok=True)
    results = []
    for fold in folds:
        tr = [x for x in items if x["cycle"] != fold]
        te = [x for x in items if x["cycle"] == fold]
        if not te or not tr:
            continue
        dl_tr = DataLoader(Crops(tr, a.data, a.size, True), batch_size=a.batch,
                           shuffle=True, num_workers=0, drop_last=False)
        dl_te = DataLoader(Crops(te, a.data, a.size, False), batch_size=a.batch,
                           shuffle=False, num_workers=0)
        model = make_model(len(classes), a.weights).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, a.epochs))
        # 类别不均衡（扎带 57 vs 螺丝刀 820）
        cnt = np.bincount([x["_y"] for x in tr], minlength=len(classes)).astype(np.float32)
        w = torch.tensor((cnt.sum() / (len(classes) * np.maximum(cnt, 1))), dtype=torch.float32).to(device)
        crit = nn.CrossEntropyLoss(weight=w)
        t0 = time.time()
        for ep in range(a.epochs):
            model.train()
            tot = 0.0
            for x, y in dl_tr:
                opt.zero_grad()
                loss = crit(model(x.to(device)), y.to(device))
                loss.backward()
                opt.step()
                tot += float(loss) * len(y)
            sched.step()
            if ep == a.epochs - 1 or ep % 3 == 0:
                acc, _p, _g = evaluate(model, dl_te, device)
                print(f"  fold {fold} ep {ep + 1}/{a.epochs} loss {tot / len(tr):.3f} 测试 {acc:.1%}"
                      f"  {time.time() - t0:.0f}s")
        acc, preds, gts = evaluate(model, dl_te, device)
        per = {}
        for c, i in c2i.items():
            m = gts == i
            if m.sum():
                per[c] = {"n": int(m.sum()), "acc": float((preds[m] == i).mean())}
        results.append({"fold": fold, "n_train": len(tr), "n_test": len(te),
                        "acc": round(acc, 4), "per_class": per})
        print(f"fold {fold} → 准确率 {acc:.1%} ｜ " +
              " ".join(f"{k}:{v['acc']:.0%}({v['n']})" for k, v in per.items()))

    if results:
        mean = float(np.mean([r["acc"] for r in results]))
        chance = 1.0 / len(classes)
        print(f"\n=== 留一周期平均准确率 {mean:.1%}（随机 {chance:.1%}，廉价特征基线 47.9%）===")
        json_doc = {
            "data": str(a.data), "size": a.size, "epochs": a.epochs,
            "classes": classes, "folds": [r["fold"] for r in results],
            "mean_acc": round(mean, 4), "chance": round(chance, 4),
            "cheap_feature_baseline": 0.479,
            "per_fold": results,
            "caveat": "单工人单产品单机位；留一周期不等于跨工人泛化",
        }
        (a.out / "results.json").write_text(json.dumps(json_doc, ensure_ascii=False, indent=1),
                                            encoding="utf-8")
        print(f"写入 {a.out / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
