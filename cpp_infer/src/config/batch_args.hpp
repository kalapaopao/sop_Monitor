#pragma once
// batch_probe 专属参数。不再继承 main.cpp 的 22 字段 Args。
//
// 优先级（高到低）：位置参数/命令行选项 > --preset 的 YAML > 内置默认值
#include <yaml-cpp/yaml.h>

#include <cmath>
#include <cstdlib>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct BatchArgs {
    std::vector<std::string> sources;

    std::string modelDir = "models";
    std::string yoloEngine;
    std::string rtmwEngine;
    std::string preset;
    std::string metrics;      // 逐帧结果年龄 CSV
    std::string batchMetrics; // 逐组耗时 CSV

    double duration = 0;
    double batchWaitMs = 5;
    double maxResultAgeMs = 200;

    int detWidth = 640;
    int detHeight = 384;
    int detBatch = 4;
    int poseBatch = 4;
    int displayWidth = 1100;
    int displayHeight = 620;
    int startSec = 0;

    bool headless = false;
    bool singleSource = false;

    std::string decoder = "gst-h265";
    std::string preprocessing = "cuda";
    std::string posePreprocessing = "cuda";
    std::string reusableIO = "on";

    // 诊断/校验模式
    std::string sampleDir;
    std::string validationDir;
};

namespace sopcfg {

inline void applyPreset(BatchArgs& a, const std::string& name) {
    // 程序约定从 build/ 启动，但用户可能在 cpp_infer/ 下运行。
    // 依次尝试：绝对/相对原样 → ../config/ → config/
    std::vector<std::string> candidates;
    if (name.find('/') != std::string::npos || name[0] == '/') {
        candidates.push_back(name);
    } else {
        candidates.push_back("../config/" + name + ".yaml");  // 从 build/ 启动
        candidates.push_back("config/" + name + ".yaml");      // 从 cpp_infer/ 启动
        candidates.push_back(name + ".yaml");                  // 当前目录
    }

    std::string path, lastError;
    for (const auto& c : candidates) {
        try {
            YAML::LoadFile(c);  // 只验证可读，真正解析在下面
            path = c;
            break;
        } catch (const YAML::Exception&) {
            lastError = c;
        }
    }
    if (path.empty()) {
        std::ostringstream tried;
        for (const auto& c : candidates) tried << "\n  " << c;
        throw std::runtime_error("Cannot find preset '" + name + "'. Tried:" + tried.str());
    }

    YAML::Node root;
    try {
        root = YAML::LoadFile(path);
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("Cannot read preset " + path + ": " + e.what());
    }
    a.preset = path;

    if (root["sources"] && root["sources"].IsSequence()) {
        a.sources.clear();
        for (const auto& s : root["sources"]) a.sources.push_back(s.Scalar());
    }
    auto num = [&](const char* k, double& v) { if (root[k]) v = root[k].as<double>(); };
    auto ini = [&](const char* k, int& v) { if (root[k]) v = root[k].as<int>(); };
    auto str = [&](const char* k, std::string& v) { if (root[k]) v = root[k].Scalar(); };
    auto bln = [&](const char* k, bool& v) { if (root[k]) v = root[k].as<bool>(); };

    ini("det_width", a.detWidth);
    ini("det_height", a.detHeight);
    ini("det_batch", a.detBatch);
    ini("pose_batch", a.poseBatch);
    num("batch_wait_ms", a.batchWaitMs);
    num("duration", a.duration);
    num("max_result_age_ms", a.maxResultAgeMs);
    ini("display_width", a.displayWidth);
    ini("display_height", a.displayHeight);
    ini("start_sec", a.startSec);
    bln("headless", a.headless);
    str("model_dir", a.modelDir);
    str("yolo_engine", a.yoloEngine);
    str("rtmw_engine", a.rtmwEngine);
    str("metrics", a.metrics);
    str("batch_metrics", a.batchMetrics);
    str("decoder", a.decoder);
    str("preprocessing", a.preprocessing);
    str("pose_preprocessing", a.posePreprocessing);
    str("reuse_io", a.reusableIO);
}

inline BatchArgs parseBatchArgs(int argc, char** argv) {
    BatchArgs a;

    // ── 第一遍：先找 --preset，让 YAML 值成为后续的基底 ──
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--preset") {
            if (++i >= argc) throw std::runtime_error("Missing --preset value");
            applyPreset(a, argv[i]);
        }
    }

    // 记录哪些字段被命令行显式设置过（命令行优先于 YAML）
    std::vector<std::string> positional;

    // ── 第二遍：命令行覆盖 ──
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto value = [&]() {
            if (i + 1 >= argc) throw std::runtime_error("Missing value for " + key);
            return std::string(argv[++i]);
        };

        if (key == "--preset") { ++i; continue; }  // 第一遍已处理

        if (key == "--source") { a.sources = {value()}; a.singleSource = true; }
        else if (key == "--second-source") {
            if (a.sources.size() < 1) a.sources.resize(1);
            a.sources.resize(2); a.sources[1] = value(); a.singleSource = false;
        }
        else if (key == "--third-source") {
            if (a.sources.size() < 2) throw std::runtime_error("--third-source requires --source and --second-source");
            a.sources.resize(3); a.sources[2] = value();
        }
        else if (key == "--fourth-source") {
            if (a.sources.size() < 3) throw std::runtime_error("--fourth-source requires three prior sources");
            a.sources.resize(4); a.sources[3] = value();
        }
        else if (key == "--single-source") a.singleSource = true;
        else if (key == "--model-dir") a.modelDir = value();
        else if (key == "--yolo-engine") a.yoloEngine = value();
        else if (key == "--rtmw-engine") a.rtmwEngine = value();
        else if (key == "--duration") a.duration = std::stod(value());
        else if (key == "--start-sec") a.startSec = std::stoi(value());
        else if (key == "--det-width") a.detWidth = std::stoi(value());
        else if (key == "--det-height") a.detHeight = std::stoi(value());
        else if (key == "--det-batch") a.detBatch = std::stoi(value());
        else if (key == "--pose-batch") a.poseBatch = std::stoi(value());
        else if (key == "--batch-wait-ms") a.batchWaitMs = std::stod(value());
        else if (key == "--max-result-age-ms") a.maxResultAgeMs = std::stod(value());
        else if (key == "--display-width") a.displayWidth = std::stoi(value());
        else if (key == "--display-height") a.displayHeight = std::stoi(value());
        else if (key == "--metrics") a.metrics = value();
        else if (key == "--batch-metrics") a.batchMetrics = value();
        else if (key == "--sample-dir") a.sampleDir = value();
        else if (key == "--validation-dir") a.validationDir = value();
        else if (key == "--decoder") a.decoder = value();
        else if (key == "--preprocess") a.preprocessing = value();
        else if (key == "--pose-preprocess") a.posePreprocessing = value();
        else if (key == "--reuse-io") a.reusableIO = value();
        else if (key == "--headless") a.headless = true;
        else if (key == "--help" || key == "-h") {
            std::cout << "用法:\n"
                      << "  sop_nx_batch_probe VIDEO [VIDEO2 [VIDEO3 [VIDEO4]]] [选项]\n"
                      << "  sop_nx_batch_probe --preset 4lane [选项]\n\n"
                      << "位置参数（最常见，直接给视频或 RTSP）:\n"
                      << "  VIDEO...                 1-4 路视频文件或 rtsp:// 地址\n\n"
                      << "预设:\n"
                      << "  --preset NAME            读取 config/NAME.yaml\n\n"
                      << "必要选项:\n"
                      << "  --duration SEC           运行秒数，必须 > 5（前 5 秒预热）\n"
                      << "  --model-dir DIR          模型目录，默认 models\n\n"
                      << "性能选项（默认已是实测最优）:\n"
                      << "  --det-height 384|640     检测输入高度，默认 384\n"
                      << "  --det-batch 1|2|4        检测批大小，默认 4\n"
                      << "  --pose-batch 1|2|4       姿态批大小，默认 4\n"
                      << "  --batch-wait-ms MS       攒批等待，默认 5\n\n"
                      << "显示选项:\n"
                      << "  --headless               不显示画面（生产默认）\n"
                      << "  --display-width/height   窗口尺寸，默认 1100x620\n"
                      << "  --max-result-age-ms MS   超过该年龄的骨架不画，默认 200\n\n"
                      << "输出选项:\n"
                      << "  --metrics CSV            逐帧结果年龄明细\n"
                      << "  --batch-metrics CSV      逐组耗时明细\n\n"
                      << "示例:\n"
                      << "  ./sop_nx_batch_probe 03.mp4 03.mp4 07.mp4 07.mp4 --duration 40 --headless\n"
                      << "  ./sop_nx_batch_probe --preset 4lane --duration 10\n";
            std::exit(0);
        }
        else if (key.size() > 1 && key[0] == '-') throw std::runtime_error("Unknown argument: " + key);
        else positional.push_back(key);
    }

    // ── 位置参数：覆盖 sources（最人性化的一条路径）──
    if (!positional.empty()) {
        a.sources = positional;
        a.singleSource = false;
    }

    // ── 归一化与校验 ──
    if (a.sources.empty()) a.sources = {"../test_vedio/02.mp4"};
    if (a.sources.size() > 4) throw std::runtime_error("At most 4 sources are supported");
    if (a.sources.size() == 1) a.singleSource = true;

    if (a.modelDir.empty()) a.modelDir = "models";
    // 引擎默认路径随批大小选择
    if (a.yoloEngine.empty())
        a.yoloEngine = a.modelDir + "/batch_20261020/obb_640x" + std::to_string(a.detHeight) + "_b1_4.plan";
    if (a.rtmwEngine.empty())
        a.rtmwEngine = a.poseBatch == 1 ? a.modelDir + "/../build/rtmw.plan"
                                        : a.modelDir + "/batch_20261020/rtmw_b" + std::to_string(a.poseBatch) + ".plan";

    if (a.duration <= 5 || !std::isfinite(a.duration))
        throw std::runtime_error("--duration must be greater than 5 seconds (first 5s warmup)");
    if ((a.detBatch != 1 && a.detBatch != 2 && a.detBatch != 4) ||
        (a.poseBatch != 1 && a.poseBatch != 2 && a.poseBatch != 4))
        throw std::runtime_error("--det-batch/--pose-batch must be 1, 2 or 4");
    if (a.detWidth != 640 || (a.detHeight != 384 && a.detHeight != 640))
        throw std::runtime_error("--det-width must be 640 and --det-height 384 or 640");
    if (!std::isfinite(a.batchWaitMs) || a.batchWaitMs < 0 || a.batchWaitMs > 20)
        throw std::runtime_error("--batch-wait-ms must be 0..20");
    if (a.displayWidth < 160 || a.displayHeight < 120)
        throw std::runtime_error("Display dimensions are too small");
    if (!std::isfinite(a.maxResultAgeMs) || a.maxResultAgeMs <= 0)
        throw std::runtime_error("--max-result-age-ms must be positive");
    if (a.preprocessing != "cpu" && a.preprocessing != "cuda")
        throw std::runtime_error("--preprocess must be cpu or cuda");
    if (a.posePreprocessing != "cpu" && a.posePreprocessing != "cuda")
        throw std::runtime_error("--pose-preprocess must be cpu or cuda");
    if (a.reusableIO != "on" && a.reusableIO != "off")
        throw std::runtime_error("--reuse-io must be on or off");
    if (a.sampleDir.empty() != a.validationDir.empty())
        throw std::runtime_error("--sample-dir and --validation-dir must be supplied together");

    return a;
}

}  // namespace sopcfg
