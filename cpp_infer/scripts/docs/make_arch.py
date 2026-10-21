# -*- coding: utf-8 -*-
"""drawio 风格的代码框架结构图（白底、细线、直角框、浅色填充）"""
import os

W, H = 1200, 840
FONT = "Helvetica, Arial, Microsoft YaHei, sans-serif"

BLUE = ("#DAE8FC", "#6C8EBF")
GREEN = ("#D5E8D4", "#82B366")
YELLOW = ("#FFF2CC", "#D6B656")
PURPLE = ("#E1D5E7", "#9673A6")
GRAY = ("#F5F5F5", "#999999")

out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">']
out.append(f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>')
out.append(
    '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
    'orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#4D4D4D"/></marker></defs>'
)


def node(x, y, w, h, title, sub, colors, rx=0, dashed=False, size=15):
    fill, stroke = colors
    dash = ' stroke-dasharray="6 4"' if dashed else ""
    out.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="1"{dash}/>'
    )
    ty = y + (h / 2 - 2 if not sub else h / 2 - 11)
    out.append(
        f'<text x="{x + w / 2}" y="{ty}" font-family="{FONT}" font-size="{size}" '
        f'font-weight="bold" fill="#1A1A1A" text-anchor="middle" dominant-baseline="central">{title}</text>'
    )
    if sub:
        out.append(
            f'<text x="{x + w / 2}" y="{y + h / 2 + 13}" font-family="{FONT}" font-size="12.5" '
            f'fill="#4D4D4D" text-anchor="middle" dominant-baseline="central">{sub}</text>'
        )


def group(x, y, w, h, label):
    out.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="none" stroke="#8C8C8C" '
        f'stroke-width="1" stroke-dasharray="7 5"/>'
    )
    out.append(
        f'<text x="{x + 12}" y="{y + 18}" font-family="{FONT}" font-size="13" '
        f'font-weight="bold" fill="#666666">{label}</text>'
    )


def arrow(x1, y1, x2, y2, label="", dashed=False, lx=None, ly=None):
    dash = ' stroke-dasharray="6 4"' if dashed else ""
    out.append(
        f'<path d="M{x1} {y1} L{x2} {y2}" fill="none" stroke="#4D4D4D" stroke-width="1.2"'
        f'{dash} marker-end="url(#a)"/>'
    )
    if label:
        out.append(
            f'<text x="{lx if lx else (x1 + x2) / 2 + 8}" y="{ly if ly else (y1 + y2) / 2 - 4}" '
            f'font-family="{FONT}" font-size="12" fill="#666666">{label}</text>'
        )


# Runtime 组
group(40, 50, 330, 240, "runtime")
node(60, 78, 290, 58, "capture/hardware_capture.hpp", "GStreamer 硬解 · I420 → BGR", GREEN)
node(60, 150, 290, 58, "core/bounded_queue.hpp", "最新帧槽 · 容量 1 · 忙时替换", GREEN)
node(60, 222, 290, 58, "pipeline/pipeline.hpp", "解码 / 推理 / 显示 三队列", GREEN)

# Entry 组
group(400, 50, 500, 240, "entry（6 个可执行目标）")
node(420, 78, 460, 58, "core/main.cpp", "legacy / serial / pipeline / parallel", BLUE)
node(420, 150, 460, 58, "probes/dual_probe.cpp", "1 / 2 / 4 路共享引擎 · 公平轮询", BLUE)
node(420, 222, 460, 58, "probes/batch_probe.cpp", "凑批 5 ms · 末批补齐 · 按 owner 回写", BLUE)

# Inference 组
group(40, 330, 860, 220, "inference（TensorRT FP16）")
node(60, 358, 400, 74, "models/joint_detector.hpp", "4 类联合 OBB：工具×2 / 手 / 人", PURPLE)
node(480, 358, 400, 74, "yoloDetections()", "3 类 OBB 解码 + 每类旋转 NMS", PURPLE)
node(60, 450, 400, 74, "yoloXPeople()", "YOLOX-tiny 人体框（旧三模型链路）", PURPLE)
node(480, 450, 400, 74, "rtmwPose()", "133 关键点 · 192×256 · SimCC", PURPLE)

# Acceleration 组
group(40, 600, 860, 190, "acceleration")
node(60, 628, 260, 76, "silu_plugin.cu", "93 组 SiLU 插件", YELLOW)
node(340, 628, 260, 76, "models/stage2_model.inc", "GPU 端 RTMW 预处理", YELLOW)
node(620, 628, 260, 76, "models/batch_model.inc", "批量 IO / GPU 裁剪", YELLOW)
node(60, 716, 820, 50, "preprocess/ · stage2_kernels.cu · batch_kernels.cu · accel_preprocess.cu", "sm72 + PTX 编译的预处理内核", YELLOW)

# Output 组
group(930, 330, 230, 220, "output")
node(946, 358, 198, 74, "draw()", "框 · 骨架 · 小字 HUD", GRAY)
node(946, 450, 198, 74, "--metrics CSV", "任务数 / 年龄 / 指纹", GRAY)

# 连线
arrow(370, 145, 400, 145)
arrow(205, 290, 205, 330, "帧")
arrow(650, 290, 650, 330, "任务")
arrow(900, 440, 930, 440)
arrow(470, 550, 470, 600, "GPU 支撑", dashed=True, lx=480, ly=576)

out.append("</svg>")

base = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "docs", "media", "architecture")
with open(base + ".svg", "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(out))

html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>html,body{{margin:0;padding:0;background:#fff}}</style></head>
<body><img src="architecture.svg" width="{W}" height="{H}" style="display:block"></body></html>"""
with open(base + ".html", "w", encoding="utf-8", newline="\n") as f:
    f.write(html)
print("written", base, os.path.getsize(base + ".svg"))
