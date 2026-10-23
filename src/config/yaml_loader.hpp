#pragma once
// 极简 YAML 读取：只支持本项目配置文件用到的子集
//   key: value
//   key: [a, b, c]
//   # 注释
// 不支持嵌套、锚点、多文档。yaml-cpp 已装在 NX 上，这里只做标量取值。
#include <yaml-cpp/yaml.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace sopcfg {

inline YAML::Node loadFile(const std::string& path) {
    try {
        return YAML::LoadFile(path);
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("Cannot read config " + path + ": " + e.what());
    }
}

// 读取可选字段：存在则覆盖，不存在保持原值
template <typename T>
inline void read(const YAML::Node& root, const char* key, T& out) {
    if (!root[key]) return;
    try {
        out = root[key].as<T>();
    } catch (const YAML::Exception& e) {
        throw std::runtime_error(std::string("Bad config key '") + key + "': " + e.what());
    }
}

inline void readString(const YAML::Node& root, const char* key, std::string& out) {
    if (!root[key]) return;
    out = root[key].Scalar();
}

inline void readSources(const YAML::Node& root, std::vector<std::string>& out) {
    if (!root["sources"]) return;
    if (!root["sources"].IsSequence())
        throw std::runtime_error("Config key 'sources' must be a list");
    out.clear();
    for (const auto& item : root["sources"]) out.push_back(item.Scalar());
}

}  // namespace sopcfg
