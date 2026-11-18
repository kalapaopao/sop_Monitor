"""因果膨胀卷积 TCN（第 2 步：替代逐帧 MLP，解决抖动与边界）。

设计要点：
  · **因果**：只看当前帧与过去帧，不用未来帧 → 能上 NX 做实时
  · **膨胀卷积**：dilation 1/2/4，3 层即可覆盖 15 帧（1.5 秒 @10Hz）
  · **残差 + LayerNorm**：小模型也稳定
  · 输出：逐帧 7 类 logits（与 MLP 版本同口径，便于对比）
  · 参数量约 6 万，CPU 推理 < 1 ms/帧
"""
from __future__ import annotations
import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    """因果卷积：右侧补 padding，保证 t 时刻只看到 t 及以前。"""

    def __init__(self, c_in, c_out, k=3, dilation=1):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv = nn.Conv1d(c_in, c_out, k, dilation=dilation, padding=0)

    def forward(self, x):                     # x: (B, C, T)
        x = torch.nn.functional.pad(x, (self.pad, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    def __init__(self, c, k=3, dilation=1, dropout=0.1):
        super().__init__()
        self.conv1 = CausalConv1d(c, c, k, dilation)
        self.norm1 = nn.LayerNorm(c)
        self.conv2 = CausalConv1d(c, c, k, dilation)
        self.norm2 = nn.LayerNorm(c)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x):                     # (B, C, T)
        h = self.conv1(x)
        h = self.norm1(h.transpose(1, 2)).transpose(1, 2)
        h = self.act(h)
        h = self.drop(h)
        h = self.conv2(h)
        h = self.norm2(h.transpose(1, 2)).transpose(1, 2)
        h = self.act(h)
        h = self.drop(h)
        return x + h


class TCN(nn.Module):
    """输入 (B, T, D) → 输出 (B, T, n_classes)。"""

    def __init__(self, d_in, n_classes, channels=64, dilations=(1, 2, 4), dropout=0.1):
        super().__init__()
        self.inp = nn.Conv1d(d_in, channels, 1)
        self.blocks = nn.ModuleList([
            TCNBlock(channels, 3, d, dropout) for d in dilations])
        self.head = nn.Conv1d(channels, n_classes, 1)

    def forward(self, x):                     # (B, T, D)
        h = x.transpose(1, 2)                 # (B, D, T)
        h = self.inp(h)
        for b in self.blocks:
            h = b(h)
        h = self.head(h)                      # (B, C, T)
        return h.transpose(1, 2)              # (B, T, C)

    def receptive_field(self):
        """计算理论感受野（帧）。"""
        rf = 1
        for _b in self.blocks:
            rf += 2 * (3 - 1) * _b.conv1.conv.dilation[0]
        return rf


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


if __name__ == '__main__':
    m = TCN(105, 7)
    print(f'参数量 {count_params(m):,} ｜ 感受野 {m.receptive_field()} 帧 '
          f'（{m.receptive_field() * 0.1:.1f} 秒 @10Hz）')
    x = torch.randn(4, 15, 105)
    print('输出形状:', tuple(m(x).shape))
