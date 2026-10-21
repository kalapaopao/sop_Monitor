# 四路计时与复现

本入口只用于分析。当前普通四路程序仍为 `sop_nx_stage2_probe`。

## 编译

在NX中：

```bash
cd /home/nvidia/Music/project/sop_Monitor/cpp_infer
cmake -S . -B build -DSOP_BUILD_ACCEL=ON
cmake --build build --target sop_nx_profile_probe -j2
```

## 一次40秒计时

以下示例会创建单独的手动测试目录。再次运行时换用新目录名，避免覆盖已有数据。

```bash
mkdir -p output/profile_manual
SOP_PROFILE_CSV=output/profile_manual/phases.csv \
./build/sop_nx_profile_probe \
  --source ../test_vedio/03.mp4 \
  --second-source ../test_vedio/03.mp4 \
  --third-source ../test_vedio/07.mp4 \
  --fourth-source ../test_vedio/07.mp4 \
  --duration 40 \
  --metrics output/profile_manual/metrics.csv
```

加 `--headless` 运行无画面对照。没有设置SOP_PROFILE_CSV时不写阶段CSV；分析目标内仍有少量CPU时钟读取，吞吐基准优先使用原stage2目标。
程序沿用原入口的桌面环境准备逻辑；没有接入可用桌面时先用headless，不改变系统桌面配置。

## 读数据

- phases.csv：lane、frame、start_ms、domain、model、stage、duration_ms。
- CPU domain记录主机阶段耗时；GPU domain记录同stream事件区间。
- start_ms小于5000的记录属于预热，稳态统计应排除。
- capture记录覆盖每路实际解码帧；推理记录只覆盖真正处理的帧。不同阶段样本数不同是正常现象。
- pose可以每任务调用0次或多次，取决于检测到的人数。比较骨架耗时时需同时查看person_count/pose_calls。
- CPU enqueue、CPU完成等待与GPU执行相互重叠；阶段总计也包含子项，禁止全部相加。
- 引擎GPU区间包括内部算子、转换与调度间隙；若需要纯内核占用，参考Nsight并考虑采集开销。
- GUI的compose包含四路缩放、叠画和指标记录，present_wait包含imshow/waitKey，不包括主循环定时sleep。
- 端到端结果与同帧指纹见metrics.csv；不要用四宫格播放FPS替代每路完整识别次数。

## 已保存实验

```bash
python3 analyze_profile.py output/profile_20261020
python3 analyze_nsys.py output/profile_20261020
```

前者读取已保存的数据和审计文件，重新生成analysis.json并将摘要打印到终端；后者读取现有SQLite生成nsys_analysis.json。
`profile_experiments.py`会顺序重新运行全部实验、覆盖同名日志并打开四路画面，总计约数分钟。保存旧结果或修改输出目录后再运行。

Nsight原始报告位于：

```text
/home/nvidia/Music/project/sop_Monitor/cpp_infer/output/profile_20261020/four_headless.nsys-rep
```

该实验使用现有工具，不需要安装新SDK或修改系统设置。

## 测试与回退

设备CTest版本较旧，请显式进入构建目录：

```bash
cd /home/nvidia/Music/project/sop_Monitor/cpp_infer/build
ctest --output-on-failure
```

本轮4项测试实际通过。`ctest --test-dir build`在此设备曾导致未找到测试，不能把退出0当作测试已通过。

回退原文件的备份在 `output/profile_20261020/before`。恢复前检查后续工作区修改，只恢复本轮涉及的7个文件并重新编译。具体范围和限制见PROFILE_REPORT_20261020.md，不要恢复或覆盖用户原有.gitignore修改。
