#pragma once
// 各探针共享的配置字段。
// 只放「多个探针都要用」的项；探针专属字段留在各自的 Arg 结构体里。
#include <string>

struct CommonConfig {
    std::string modelDir = "models";
    double duration = 0;
    bool headless = false;
    double maxResultAgeMs = 200;
    int displayWidth = 1100;
    int displayHeight = 620;
    std::string metrics;
};

struct EngineConfig {
    std::string yoloEngine;   // 旧三模型入口的 OBB 引擎
    std::string yoloxEngine;  // 旧三模型入口的人体检测引擎
    std::string rtmwEngine;   // 姿态引擎
};
