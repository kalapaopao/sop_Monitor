#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <mutex>
#include <thread>
#include <algorithm>
#include "bounded_queue.hpp"
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef SOP_ACCEL
#include "../preprocess/accel_preprocess.hpp"
#endif
#ifdef SOP_STAGE2
#include "../preprocess/stage2_kernels.hpp"
#endif

#ifdef __linux__
#include <unistd.h>
#endif

namespace {

class Logger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) std::cerr << "[TensorRT] " << message << '\n';
    }
};

Logger gLogger;

void prepareDesktopDisplay() {
#ifdef __linux__
    const char* display = std::getenv("DISPLAY");
    const bool missingDisplay = !display || !*display;
    const char* authority = std::getenv("XAUTHORITY");
    const std::string desktopAuthority = "/run/user/" + std::to_string(getuid()) + "/gdm/Xauthority";
    // SSH does not inherit the logged-in user's GNOME display environment.
    // Only fall back to this user's local :0 desktop; preserve explicit displays.
    if (missingDisplay) {
        if (access("/tmp/.X11-unix/X0", F_OK) != 0 ||
            access(desktopAuthority.c_str(), R_OK) != 0) {
            throw std::runtime_error("No accessible local GNOME desktop found. Run in the NX desktop terminal, set DISPLAY/XAUTHORITY explicitly, or use --headless for testing.");
        }
        if (setenv("DISPLAY", ":0", 1) != 0) throw std::runtime_error("Cannot set DISPLAY");
        display = std::getenv("DISPLAY");
    }
    if ((!authority || !*authority) && display &&
        (std::strcmp(display, ":0") == 0 || std::strcmp(display, ":0.0") == 0) &&
        access(desktopAuthority.c_str(), R_OK) == 0) {
        if (setenv("XAUTHORITY", desktopAuthority.c_str(), 1) != 0)
            throw std::runtime_error("Cannot set XAUTHORITY");
    }
    std::cerr << "[display] Using " << (display ? display : "") << '\n';
#endif
}

void checkCuda(cudaError_t error, const char* what) {
    if (error != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(error));
}

size_t volume(const nvinfer1::Dims& dims) {
    size_t result = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] < 0) throw std::runtime_error("Dynamic output dimension was not resolved");
        result *= static_cast<size_t>(dims.d[i]);
    }
    return result;
}

size_t typeSize(nvinfer1::DataType type) {
    switch (type) {
    case nvinfer1::DataType::kFLOAT: return sizeof(float);
    case nvinfer1::DataType::kHALF: return sizeof(uint16_t);
    case nvinfer1::DataType::kINT8: return sizeof(int8_t);
    case nvinfer1::DataType::kINT32: return sizeof(int32_t);
    case nvinfer1::DataType::kBOOL: return sizeof(bool);
    default: throw std::runtime_error("Unsupported TensorRT tensor type");
    }
}

float halfToFloat(uint16_t value) {
    const uint32_t sign = (value & 0x8000u) << 16;
    const uint32_t exponent = (value >> 10) & 0x1fu;
    const uint32_t fraction = value & 0x3ffu;
    uint32_t result;
    if (exponent == 0) {
        if (fraction == 0) result = sign;
        else {
            float f = std::ldexp(static_cast<float>(fraction), -24);
            std::memcpy(&result, &f, sizeof(result));
            result = (result & 0x7fffffffu) | sign;
        }
    } else if (exponent == 31) {
        result = sign | 0x7f800000u | (fraction << 13);
    } else {
        result = sign | ((exponent + 112u) << 23) | (fraction << 13);
    }
    float output;
    std::memcpy(&output, &result, sizeof(output));
    return output;
}

uint16_t floatToHalf(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const uint32_t exponent = (bits >> 23) & 0xffu;
    const uint32_t fraction = bits & 0x7fffffu;
    if (exponent == 0xffu) return static_cast<uint16_t>(sign | 0x7c00u | (fraction ? 0x0200u : 0u));
    const int e = static_cast<int>(exponent) - 127 + 15;
    if (e >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    if (e <= 0) {
        if (e < -10) return static_cast<uint16_t>(sign);
        const uint32_t mantissa = fraction | 0x800000u;
        return static_cast<uint16_t>(sign | ((mantissa >> (14 - e)) + ((mantissa >> (13 - e)) & 1u)));
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(e) << 10) | (fraction >> 13));
}

struct TensorResult {
    std::vector<float> floats;
    std::vector<int32_t> ints;
    nvinfer1::Dims dims{};
    nvinfer1::DataType type{};

    float value(size_t index) const {
        if (type == nvinfer1::DataType::kFLOAT) return floats.at(index);
        if (type == nvinfer1::DataType::kHALF) return halfToFloat(static_cast<uint16_t>(ints.at(index)));
        if (type == nvinfer1::DataType::kINT32) return static_cast<float>(ints.at(index));
        throw std::runtime_error("Tensor result is not numeric float/int32");
    }
    size_t size() const { return type == nvinfer1::DataType::kFLOAT ? floats.size() : ints.size(); }
};

#ifdef SOP_STAGE2
#include "../preprocess/stage2_io.hpp"
using ModelOutputs = const std::vector<TensorResult>&;
#else
using ModelOutputs = std::vector<TensorResult>;
#endif
class TrtModel {
public:
    explicit TrtModel(const std::string& enginePath) : path_(enginePath) {
        std::ifstream file(enginePath, std::ios::binary | std::ios::ate);
        if (!file) throw std::runtime_error("Cannot open TensorRT engine: " + enginePath);
        const auto size = file.tellg(); file.seekg(0);
        std::vector<char> bytes(static_cast<size_t>(size));
        file.read(bytes.data(), size);
        runtime_.reset(nvinfer1::createInferRuntime(gLogger));
        if (!runtime_) throw std::runtime_error("createInferRuntime failed");
        engine_.reset(runtime_->deserializeCudaEngine(bytes.data(), bytes.size()), DestroyEngine{});
        if (!engine_) throw std::runtime_error("deserializeCudaEngine failed: " + enginePath);
        context_.reset(engine_->createExecutionContext());
        if (!context_) throw std::runtime_error("createExecutionContext failed: " + enginePath);
        checkCuda(cudaStreamCreate(&stream_), "cudaStreamCreate");
        for (int i = 0; i < engine_->getNbBindings(); ++i) {
            const std::string name = engine_->getBindingName(i);
            if (engine_->bindingIsInput(i)) inputIndex_ = i;
            else outputIndices_.push_back(i);
            std::cerr << "[engine] " << path_ << " binding " << i << " " << name << "\n";
        }
        if (inputIndex_ < 0) throw std::runtime_error("Engine has no input: " + enginePath);
    }

#ifdef SOP_ACCEL
    // Each worker shares weights but owns its context, stream and mutable buffers.
    explicit TrtModel(std::shared_ptr<TrtModel> prototype) : owner_(prototype), engine_(prototype->engine_) {
        context_.reset(engine_->createExecutionContext());
        if (!context_) throw std::runtime_error("create worker context failed");
        checkCuda(cudaStreamCreate(&stream_), "create worker stream");
        for (int i=0; i<engine_->getNbBindings(); ++i) {
            if (engine_->bindingIsInput(i)) inputIndex_=i;
            else outputIndices_.push_back(i);
        }
    }
    bool gpuPreprocess = false;
#ifdef SOP_STAGE2
#include "../models/stage2_model.inc"
#ifdef SOP_BATCH
#include "../models/batch_model.inc"
#endif
#endif
    ModelOutputs inferLetterbox(const cv::Mat& image, int side=640) {
        if (image.empty() || image.type()!=CV_8UC3) throw std::runtime_error("Invalid preprocessing image");
        if (!context_->setBindingDimensions(inputIndex_,nvinfer1::Dims4{1,3,side,side}))
            throw std::runtime_error("Invalid GPU input dimensions");
        resizeBuffers();
        const auto type=engine_->getBindingDataType(inputIndex_);
        if (type!=nvinfer1::DataType::kFLOAT && type!=nvinfer1::DataType::kHALF)
            throw std::runtime_error("GPU preprocessing requires FP32/FP16 input");
#ifdef SOP_STAGE2
        if(!io_.prepared) beginFrame(image);
        io_.prepared=false;
        const size_t row=size_t(image.cols)*3;
#else
        const size_t row=size_t(image.cols)*3, bytes=row*image.rows;
        if (bytes>sourceBytes_) {
            if (sourceDevice_) checkCuda(cudaFree(sourceDevice_),"free source");
            sourceDevice_=nullptr; sourceBytes_=0;
            checkCuda(cudaMalloc(&sourceDevice_,bytes),"allocate source"); sourceBytes_=bytes;
        }
        checkCuda(cudaMemcpy2DAsync(sourceDevice_,row,image.data,image.step,row,image.rows,
                    cudaMemcpyHostToDevice,stream_),"upload source");
#endif
        {
#ifdef SOP_PROFILE
        sop_profile::CpuScope preHost("preprocess_host_submit");
        sop_profile::GpuScope preGpu("preprocess",stream_);
#endif
        const float ratio=std::min(float(side)/image.cols,float(side)/image.rows);
        const int w=std::max(1,int(std::round(image.cols*ratio))), h=std::max(1,int(std::round(image.rows*ratio)));
        checkCuda(sopLetterbox(static_cast<unsigned char*>(sourceDevice_),image.cols,image.rows,row,
                    deviceBuffers_[inputIndex_],type==nvinfer1::DataType::kHALF,side,w,h,(side-w)/2,(side-h)/2,stream_),
                    "GPU letterbox");
        }
        return execute();
    }
#endif
    ~TrtModel() {
#ifdef SOP_ACCEL
        if (sourceDevice_) cudaFree(sourceDevice_);
#endif
        for (void* ptr : deviceBuffers_) if (ptr) cudaFree(ptr);
        if (stream_) cudaStreamDestroy(stream_);
    }

    ModelOutputs infer(const std::vector<float>& input, const nvinfer1::Dims& inputDims) {
        if (!context_->setBindingDimensions(inputIndex_, inputDims)) throw std::runtime_error("setBindingDimensions failed");
        resizeBuffers();
        const auto inputType = engine_->getBindingDataType(inputIndex_);
#ifdef SOP_STAGE2
        io_.uploadBytes+=input.size()*typeSize(inputType);
#endif
        if (inputType == nvinfer1::DataType::kFLOAT) {
#ifdef SOP_STAGE2
            if(reuseIO) {
                void* pinned=io_.input.ensure(input.size()*sizeof(float));
                std::memcpy(pinned,input.data(),input.size()*sizeof(float));
                checkCuda(cudaMemcpyAsync(deviceBuffers_[inputIndex_],pinned,input.size()*sizeof(float),cudaMemcpyHostToDevice,stream_),"copy pinned tensor");
            } else
#endif
            checkCuda(cudaMemcpyAsync(deviceBuffers_[inputIndex_], input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice, stream_), "copy FP32 input");
        } else if (inputType == nvinfer1::DataType::kHALF) {
            std::vector<uint16_t> halfInput(input.size());
            for (size_t i = 0; i < input.size(); ++i) halfInput[i] = floatToHalf(input[i]);
            checkCuda(cudaMemcpyAsync(deviceBuffers_[inputIndex_], halfInput.data(), halfInput.size() * sizeof(uint16_t), cudaMemcpyHostToDevice, stream_), "copy FP16 input");
        } else {
            throw std::runtime_error("Unsupported input type in " + path_);
        }
        return execute();
    }

    ModelOutputs execute() {
#ifdef SOP_STAGE2
        return executeStage2();
#else
        if (!context_->enqueueV2(deviceBuffers_.data(), stream_, nullptr)) throw std::runtime_error("TensorRT enqueue failed");
        std::vector<TensorResult> results;
        for (int index : outputIndices_) {
            TensorResult result;
            result.dims = context_->getBindingDimensions(index);
            result.type = engine_->getBindingDataType(index);
            const size_t count = volume(result.dims);
            const size_t bytes = count * typeSize(result.type);
            if (result.type == nvinfer1::DataType::kFLOAT) result.floats.resize(count);
            else if (result.type == nvinfer1::DataType::kINT32) result.ints.resize(count);
            else if (result.type == nvinfer1::DataType::kHALF) result.ints.resize(count);
            else throw std::runtime_error("Unsupported output type in " + path_);
            void* destination = result.type == nvinfer1::DataType::kFLOAT ? static_cast<void*>(result.floats.data()) : static_cast<void*>(result.ints.data());
            checkCuda(cudaMemcpyAsync(destination, deviceBuffers_[index], bytes, cudaMemcpyDeviceToHost, stream_), "copy output");
            results.push_back(std::move(result));
        }
        checkCuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
        return results;
#endif
    }

    std::string name(int index) const { return engine_->getBindingName(index); }

private:
    void resizeBuffers() {
        if (deviceBuffers_.size() != static_cast<size_t>(engine_->getNbBindings())) deviceBuffers_.assign(engine_->getNbBindings(), nullptr);
        for (int i = 0; i < engine_->getNbBindings(); ++i) {
            const auto dims = context_->getBindingDimensions(i);
            const size_t bytes = volume(dims) * typeSize(engine_->getBindingDataType(i));
            if (bytes > bufferSizes_[i]) {
                if (deviceBuffers_[i]) checkCuda(cudaFree(deviceBuffers_[i]), "cudaFree old buffer");
                checkCuda(cudaMalloc(&deviceBuffers_[i], bytes), "cudaMalloc");
                bufferSizes_[i] = bytes;
            }
        }
    }

    struct DestroyRuntime { void operator()(nvinfer1::IRuntime* p) const { if (p) p->destroy(); } };
    struct DestroyEngine { void operator()(nvinfer1::ICudaEngine* p) const { if (p) p->destroy(); } };
    struct DestroyContext { void operator()(nvinfer1::IExecutionContext* p) const { if (p) p->destroy(); } };
    std::string path_;
#ifdef SOP_ACCEL
    std::shared_ptr<TrtModel> owner_;
    void* sourceDevice_ = nullptr;
    size_t sourceBytes_ = 0;
#ifdef SOP_STAGE2
    SopStage2IO io_;
#endif
#endif
    std::unique_ptr<nvinfer1::IRuntime, DestroyRuntime> runtime_;
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext, DestroyContext> context_;
    cudaStream_t stream_{};
    int inputIndex_ = -1;
    std::vector<int> outputIndices_;
    std::vector<void*> deviceBuffers_;
    std::array<size_t, 16> bufferSizes_{};
};

struct Detection { std::string label; float confidence{}; cv::RotatedRect rect; std::array<cv::Point2f, 4> points{}; };
struct PoseResult { std::array<cv::Point2f, 134> keypoints{}; std::array<float, 134> scores{}; cv::Rect2f bbox; bool valid = false; };

cv::Mat letterboxCentered(const cv::Mat& frame, int size, float& ratio, int& left, int& top) {
    ratio = std::min(static_cast<float>(size) / frame.cols, static_cast<float>(size) / frame.rows);
    const int width = std::lround(frame.cols * ratio), height = std::lround(frame.rows * ratio);
    left = (size - width) / 2; top = (size - height) / 2;
    cv::Mat canvas(size, size, CV_8UC3, cv::Scalar(114, 114, 114));
    cv::Mat resized; cv::resize(frame, resized, cv::Size(width, height));
    resized.copyTo(canvas(cv::Rect(left, top, width, height))); return canvas;
}

std::vector<float> chwFloat(const cv::Mat& image, bool normalize, const cv::Scalar& mean = {}, const cv::Scalar& stddev = {}) {
    cv::Mat floatImage; image.convertTo(floatImage, CV_32FC3);
    if (normalize) { for (int y = 0; y < floatImage.rows; ++y) for (int x = 0; x < floatImage.cols; ++x) { auto& p = floatImage.at<cv::Vec3f>(y, x); for (int c = 0; c < 3; ++c) p[c] = (p[c] - mean[c]) / stddev[c]; } }
    std::vector<float> result(static_cast<size_t>(3 * image.rows * image.cols));
    for (int y = 0; y < image.rows; ++y) for (int x = 0; x < image.cols; ++x) { const auto p = floatImage.at<cv::Vec3f>(y, x); for (int c = 0; c < 3; ++c) result[static_cast<size_t>(c * image.rows * image.cols + y * image.cols + x)] = p[c]; }
    return result;
}

float polygonIou(const std::array<cv::Point2f, 4>& a, const std::array<cv::Point2f, 4>& b) {
    std::vector<cv::Point2f> va(a.begin(), a.end()), vb(b.begin(), b.end()), inter;
    const float aa = std::fabs(static_cast<float>(cv::contourArea(va))), ab = std::fabs(static_cast<float>(cv::contourArea(vb)));
    if (aa <= 0 || ab <= 0) return 0;
    const float ai = cv::intersectConvexConvex(va, vb, inter);
    return ai / (aa + ab - ai);
}

std::vector<Detection> yoloDetections(TrtModel& model, const cv::Mat& frame, float threshold) {
    float ratio; int left, top; const cv::Mat canvas = letterboxCentered(frame, 640, ratio, left, top);
    cv::Mat rgb; cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);
    auto input = chwFloat(rgb, true, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255)); auto outputs = model.infer(input, nvinfer1::Dims4{1, 3, 640, 640});
    if (outputs.empty() || outputs[0].size() != 8400u * 8u) throw std::runtime_error("Unexpected YOLO raw output shape");
    const auto& raw = outputs[0]; const std::array<std::string, 3> labels{"screwdriver1", "screwdriver2", "hand"};
    struct Candidate { int anchor; int cls; float score; };
    std::vector<std::pair<float, int>> best; best.reserve(8400);
    for (int i = 0; i < 8400; ++i) best.emplace_back(std::max({raw.value(i * 8 + 4), raw.value(i * 8 + 5), raw.value(i * 8 + 6)}), i);
    const size_t keepAnchors = std::min<size_t>(300, best.size()); std::partial_sort(best.begin(), best.begin() + keepAnchors, best.end(), std::greater<>());
    std::vector<Candidate> candidates;
    for (size_t k = 0; k < keepAnchors; ++k) { const int i = best[k].second; for (int cls = 0; cls < 3; ++cls) if (raw.value(i * 8 + 4 + cls) >= threshold) candidates.push_back({i, cls, raw.value(i * 8 + 4 + cls)}); }
    if (candidates.size() > 300) { std::partial_sort(candidates.begin(), candidates.begin() + 300, candidates.end(), [](const Candidate& a, const Candidate& b) { return a.score > b.score; }); candidates.resize(300); }
    std::vector<Detection> detections;
    for (const auto& c : candidates) {
        const float cx = (raw.value(c.anchor * 8) - left) / ratio, cy = (raw.value(c.anchor * 8 + 1) - top) / ratio, w = raw.value(c.anchor * 8 + 2) / ratio, h = raw.value(c.anchor * 8 + 3) / ratio;
        cv::RotatedRect rect(cv::Point2f(cx, cy), cv::Size2f(w, h), raw.value(c.anchor * 8 + 7) * 180.0f / static_cast<float>(CV_PI));
        std::array<cv::Point2f, 4> points; rect.points(points.data()); detections.push_back({labels[c.cls], c.score, rect, points});
    }
    std::vector<Detection> kept;
    for (const auto& label : labels) {
        std::vector<Detection> same; for (const auto& d : detections) if (d.label == label) same.push_back(d);
        std::sort(same.begin(), same.end(), [](const Detection& a, const Detection& b) { return a.confidence > b.confidence; });
        for (const auto& d : same) { bool accept = true; for (const auto& prior : kept) if (prior.label == d.label && polygonIou(prior.points, d.points) >= 0.55f) { accept = false; break; } if (accept) kept.push_back(d); }
    }
    return kept;
}

std::vector<cv::Rect2f> yoloXPeople(TrtModel& model, const cv::Mat& frame) {
    const float ratio = std::min(416.0f / frame.rows, 416.0f / frame.cols); const int width = static_cast<int>(frame.cols * ratio), height = static_cast<int>(frame.rows * ratio);
    cv::Mat padded(416, 416, CV_8UC3, cv::Scalar(114, 114, 114)), resized; cv::resize(frame, resized, cv::Size(width, height)); resized.copyTo(padded(cv::Rect(0, 0, width, height)));
    auto outputs = model.infer(chwFloat(padded, false), nvinfer1::Dims4{1, 3, 416, 416}); if (outputs.size() < 2) throw std::runtime_error("Unexpected YOLOX outputs");
    // The original HumanArt export contains a dynamic TopK/NMS tail that TRT 8.5 cannot parse.
    // The migrated engine exposes decoded boxes [1,3549,4] and class scores [1,3549,80].
    if (outputs[0].size() != 3549u * 4u || outputs[1].size() != 3549u * 80u) throw std::runtime_error("Unexpected raw YOLOX output shape");
    struct Candidate { cv::Rect2f box; float score; };
    std::vector<Candidate> candidates;
    for (size_t i = 0; i < 3549; ++i) {
        const float score = outputs[1].value(i * 80); // HumanArt class 0 is person.
        // RTMLib's default HumanArt person detector threshold is 0.3.
        if (score < 0.3f) continue;
        candidates.push_back({cv::Rect2f(cv::Point2f(outputs[0].value(i * 4) / ratio, outputs[0].value(i * 4 + 1) / ratio), cv::Point2f(outputs[0].value(i * 4 + 2) / ratio, outputs[0].value(i * 4 + 3) / ratio)), score});
    }
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) { return a.score > b.score; });
    std::vector<cv::Rect2f> boxes;
    for (const auto& candidate : candidates) {
        bool keep = true;
        for (const auto& prior : boxes) {
            const float intersection = static_cast<float>((candidate.box & prior).area());
            const float unionArea = candidate.box.area() + prior.area() - intersection;
            if (unionArea > 0 && intersection / unionArea > 0.45f) { keep = false; break; }
        }
        if (keep) boxes.push_back(candidate.box);
    }
    return boxes;
}

cv::Mat poseAffine(const cv::Rect2f& box, cv::Point2f& center, cv::Size2f& scale) {
    center = cv::Point2f((box.x + box.x + box.width) * .5f, (box.y + box.y + box.height) * .5f); scale = cv::Size2f(box.width * 1.25f, box.height * 1.25f);
    const float aspect = 192.0f / 256.0f; if (scale.width > scale.height * aspect) scale.height = scale.width / aspect; else scale.width = scale.height * aspect;
    const cv::Point2f src0 = center, src1 = center + cv::Point2f(0, -scale.width * .5f), src2 = src1 + cv::Point2f(-(src0.y - src1.y), src0.x - src1.x);
    const cv::Point2f dst0(96, 128), dst1(96, 128 - 96), dst2(dst1 + cv::Point2f(-(dst0.y - dst1.y), dst0.x - dst1.x));
    return cv::getAffineTransform(std::vector<cv::Point2f>{src0, src1, src2}, std::vector<cv::Point2f>{dst0, dst1, dst2});
}
cv::Mat poseWarp(const cv::Mat& image, const cv::Rect2f& box, cv::Point2f& center, cv::Size2f& scale) {
    const cv::Mat warp=poseAffine(box,center,scale);
    cv::Mat out; cv::warpAffine(image,out,warp,cv::Size(192,256)); return out;
}

PoseResult decodeRtmw(const std::vector<TensorResult>& outputs,const cv::Rect2f& box,
    const cv::Point2f& center,const cv::Size2f& scale) {
#ifdef SOP_PROFILE
    sop_profile::CpuScope decodeHost("simcc_decode_host");
#endif
    if(outputs.size()<2) throw std::runtime_error("Unexpected RTMW outputs");
    const auto& sx = outputs[0]; const auto& sy = outputs[1]; const size_t keypointCount = sx.size() / 384u; if (keypointCount == 0 || keypointCount > 134 || sx.size() != keypointCount * 384u || sy.size() != keypointCount * 512u) { std::cerr << "[rtmw] output sizes " << sx.size() << "," << sy.size() << " types " << static_cast<int>(sx.type) << "," << static_cast<int>(sy.type) << "\n"; throw std::runtime_error("Unexpected RTMW output shape"); } PoseResult result{}; result.bbox = box; result.valid = true;
    for (size_t k = 0; k < keypointCount; ++k) { size_t ix = 0, iy = 0; float maxx = -std::numeric_limits<float>::infinity(), maxy = -std::numeric_limits<float>::infinity(); for (size_t x = 0; x < 384; ++x) if (sx.value(k * 384 + x) > maxx) maxx = sx.value(k * 384 + x), ix = x; for (size_t y = 0; y < 512; ++y) if (sy.value(k * 512 + y) > maxy) maxy = sy.value(k * 512 + y), iy = y; result.scores[k] = (maxx + maxy) * .5f; const float px = static_cast<float>(ix) / 2.0f, py = static_cast<float>(iy) / 2.0f; result.keypoints[k] = cv::Point2f(px / 192.0f * scale.width + center.x - scale.width * .5f, py / 256.0f * scale.height + center.y - scale.height * .5f); }
    return result;
}
PoseResult rtmwPose(TrtModel& model,const cv::Mat& frame,const cv::Rect2f& box) {
    cv::Point2f center; cv::Size2f scale; const auto crop=poseWarp(frame,box,center,scale);
    auto input=chwFloat(crop,true,cv::Scalar(123.675,116.28,103.53),cv::Scalar(58.395,57.12,57.375));
    const auto& outputs=model.infer(input,nvinfer1::Dims4{1,3,256,192});
    return decodeRtmw(outputs,box,center,scale);
}
#ifdef SOP_STAGE2
PoseResult rtmwPoseGpu(TrtModel& model,const TrtModel& source,const cv::Mat& frame,const cv::Rect2f& box) {
    cv::Point2f center; cv::Size2f scale; cv::Mat inverse;
    {
#ifdef SOP_PROFILE
    sop_profile::CpuScope affineHost("affine_host");
#endif
    const auto affine=poseAffine(box,center,scale);
    cv::invertAffineTransform(affine,inverse);
    }
    const auto& outputs=model.inferWarp(source,frame.size(),inverse);
    return decodeRtmw(outputs,box,center,scale);
}
#endif

void draw(cv::Mat& frame, const std::vector<Detection>& detections, const PoseResult* pose, double fps, double latencyMs) {
    // The pose box is the single human box shown on screen. The OBB detector
    // still returns the person class for ROI selection, but drawing it here
    // would duplicate the pose box.
    for (const auto& d : detections) {
        if (d.label == "person") continue;
        std::vector<cv::Point> polygon;
        for (const auto& p : d.points) polygon.emplace_back(cv::Point(cvRound(p.x), cvRound(p.y)));
        const cv::Scalar color = d.label == "hand" ? cv::Scalar(255, 80, 200) : d.label == "screwdriver1" ? cv::Scalar(0, 220, 255) : cv::Scalar(80, 230, 80);
        cv::polylines(frame, polygon, true, color, 1);
        cv::putText(frame, d.label + " " + cv::format("%.2f", d.confidence), polygon[0], cv::FONT_HERSHEY_SIMPLEX, .65, color, 1);
    }
    if (pose && pose->valid) {
        auto valid = [&](int i) { return pose->scores[i] >= .35f && pose->keypoints[i].x >= 0 && pose->keypoints[i].y >= 0; };
        cv::rectangle(frame, pose->bbox, cv::Scalar(255, 180, 40), 1);
        const std::array<std::pair<int, int>, 8> body{{{5, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10}, {5, 11}, {6, 12}, {11, 12}}};
        for (auto [a, b] : body) if (valid(a) && valid(b)) cv::line(frame, pose->keypoints[a], pose->keypoints[b], cv::Scalar(255, 220, 60), 1);
        const std::array<std::pair<int, int>, 21> hand{{{0,1},{1,2},{2,3},{3,4},{0,5},{5,6},{6,7},{7,8},{5,9},{9,10},{10,11},{11,12},{9,13},{13,14},{14,15},{15,16},{13,17},{17,18},{18,19},{19,20},{0,17}}};
        for (int start : {91, 112}) {
            for (auto [a, b] : hand) if (valid(start + a) && valid(start + b)) cv::line(frame, pose->keypoints[start + a], pose->keypoints[start + b], cv::Scalar(255, 255, 0), 1);
            for (int i = 0; i < 21; ++i) if (valid(start + i)) cv::circle(frame, pose->keypoints[start + i], 2, cv::Scalar(255, 255, 0), -1);
        }
    }
    const std::string hud = "FPS " + cv::format("%.1f", fps) + " | " + cv::format("%.0f", latencyMs) + " ms";
    cv::putText(frame, hud, cv::Point(10, 27), cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar(12, 12, 12), 4);
    cv::putText(frame, hud, cv::Point(10, 27), cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar(255, 255, 255), 1);
}

#ifndef SOP_BATCH  // batch_probe defines its own BatchArgs; skip the shared god-object
struct Args { std::string source; std::string modelDir; std::string yoloEngine, yoloxEngine, rtmwEngine, record; bool headless = false; double duration = 0; int startSec = 0; int obbStride = 1; int personStride = 1; int poseStride = 1; int displayWidth = 1100; int displayHeight = 620; std::string mode = "legacy", metrics; int maxFrames = 0, warmupFrames = 0; bool realtime = false, previewFirst = false; double maxResultAgeMs = 200; };
Args parseArgs(int argc, char** argv) { Args a; for (int i = 1; i < argc; ++i) { const std::string key = argv[i]; auto value = [&]() { if (i + 1 >= argc) throw std::runtime_error("Missing value for " + key); return std::string(argv[++i]); }; if (key == "--source") a.source = value(); else if (key == "--model-dir") a.modelDir = value(); else if (key == "--yolo-engine") a.yoloEngine = value(); else if (key == "--yolox-engine") a.yoloxEngine = value(); else if (key == "--rtmw-engine") a.rtmwEngine = value(); else if (key == "--record") a.record = value(); else if (key == "--headless") a.headless = true; else if (key == "--duration") a.duration = std::stod(value()); else if (key == "--start-sec") a.startSec = std::stoi(value()); else if (key == "--obb-stride") a.obbStride = std::stoi(value()); else if (key == "--person-stride") a.personStride = std::stoi(value()); else if (key == "--pose-stride") a.poseStride = std::stoi(value()); else if (key == "--display-width") a.displayWidth = std::stoi(value()); else if (key == "--display-height") a.displayHeight = std::stoi(value()); else if (key == "--mode") a.mode = value(); else if (key == "--max-frames") a.maxFrames = std::stoi(value()); else if (key == "--warmup-frames") a.warmupFrames = std::stoi(value()); else if (key == "--metrics") a.metrics = value(); else if (key == "--realtime") a.realtime = true; else if (key == "--preview-first") a.previewFirst = true; else if (key == "--max-result-age-ms") a.maxResultAgeMs = std::stod(value()); else if (key == "--help" || key == "-h") { std::cout << "--source VIDEO --model-dir DIR [--headless] [--record MP4] [--duration SEC] [--start-sec SEC] [--obb-stride N] [--person-stride N] [--pose-stride N] [--display-width PX] [--display-height PX] [--mode legacy|serial|pipeline|parallel] [--max-frames N] [--warmup-frames N] [--metrics CSV] [--realtime] [--preview-first] [--max-result-age-ms MS]\n"; std::exit(0); } else throw std::runtime_error("Unknown argument: " + key); } if (a.modelDir.empty()) a.modelDir = "models"; if (a.source.empty()) a.source = "../test_vedio/02.mp4"; if (a.yoloEngine.empty()) a.yoloEngine = a.modelDir + "/../build/yolo.plan"; if (a.yoloxEngine.empty()) a.yoloxEngine = a.modelDir + "/../build/yolox.plan"; if (a.rtmwEngine.empty()) a.rtmwEngine = a.modelDir + "/../build/rtmw.plan"; if (a.obbStride < 1 || a.personStride < 1 || a.poseStride < 1) throw std::runtime_error("Stride values must be >= 1"); if (a.displayWidth < 160 || a.displayHeight < 120) throw std::runtime_error("Display dimensions are too small"); if (a.mode != "legacy" && a.mode != "serial" && a.mode != "pipeline" && a.mode != "parallel") throw std::runtime_error("Unknown execution mode"); if (a.maxFrames < 0 || a.warmupFrames < 0 || !std::isfinite(a.maxResultAgeMs) || a.maxResultAgeMs <= 0) throw std::runtime_error("Invalid frame limit or result age"); if (a.realtime && (a.mode == "legacy" || a.mode == "serial")) throw std::runtime_error("--realtime requires --mode pipeline or parallel"); if (a.mode == "legacy" && (a.maxFrames || a.warmupFrames || !a.metrics.empty())) throw std::runtime_error("Benchmark options require --mode serial, pipeline or parallel"); if (a.previewFirst && (a.mode == "legacy" || a.headless || !a.record.empty())) throw std::runtime_error("--preview-first requires a non-legacy GUI mode without recording"); return a; }
#endif // !SOP_BATCH  (batch_probe supplies its own BatchArgs)

#include "../pipeline/pipeline.hpp"

} // namespace

#ifndef SOP_INFER_EMBEDDED
int main(int argc, char** argv) {
    try {
        const Args args = parseArgs(argc, argv);
        if (!args.headless) prepareDesktopDisplay();
        if (args.mode != "legacy") return runPipeline(args);
        TrtModel yolo(args.yoloEngine), yolox(args.yoloxEngine), rtmw(args.rtmwEngine); cv::VideoCapture cap(args.source); if (!cap.isOpened()) throw std::runtime_error("Cannot open source: " + args.source); const double sourceFps = cap.get(cv::CAP_PROP_FPS) > 0 ? cap.get(cv::CAP_PROP_FPS) : 25.0; if (args.startSec > 0) cap.set(cv::CAP_PROP_POS_MSEC, args.startSec * 1000.0);
        cv::VideoWriter writer;
        if (!args.headless) { cv::namedWindow("SOP C++ TensorRT", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO); cv::resizeWindow("SOP C++ TensorRT", args.displayWidth, args.displayHeight); }
        cv::Mat frame; uint64_t frames = 0, analyzed = 0, detectionFrames = 0, poseFrames = 0; double totalMs = 0; auto started = std::chrono::steady_clock::now();
        std::vector<Detection> detections; std::vector<cv::Rect2f> boxes; PoseResult bestPose; float bestScore = -1;
        while (cap.read(frame)) { if (!args.record.empty() && !writer.isOpened()) { writer.open(args.record, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), sourceFps, frame.size()); if (!writer.isOpened()) throw std::runtime_error("Cannot open record output: " + args.record); } const auto frameStart = std::chrono::steady_clock::now(); const bool refreshObb = frames == 0 || frames % static_cast<uint64_t>(args.obbStride) == 0; const bool refreshPerson = frames == 0 || frames % static_cast<uint64_t>(args.personStride) == 0; const bool refreshPose = frames == 0 || frames % static_cast<uint64_t>(args.poseStride) == 0; if (refreshObb) detections = yoloDetections(yolo, frame, .25f); if (refreshPerson) { boxes = yoloXPeople(yolox, frame); if (boxes.empty()) bestScore = -1; } if (boxes.empty()) bestScore = -1; else if (refreshPose) { bestScore = -1; for (const auto& box : boxes) { auto pose = rtmwPose(rtmw, frame, box); const float score = std::accumulate(pose.scores.begin() + 5, pose.scores.begin() + 11, 0.0f) / 6.0f; if (score > bestScore) bestPose = pose, bestScore = score; } } const double latency = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - frameStart).count(); totalMs += latency; ++analyzed; ++frames; detectionFrames += !detections.empty(); poseFrames += bestScore >= 0; const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count(); const double fps = analyzed / std::max(elapsed, 1e-6); draw(frame, detections, bestScore >= 0 ? &bestPose : nullptr, fps, latency); if (writer.isOpened()) writer.write(frame); if (!args.headless) { cv::Mat display = frame; const double scale = std::min({1.0, static_cast<double>(args.displayWidth) / frame.cols, static_cast<double>(args.displayHeight) / frame.rows}); if (scale < 1.0) cv::resize(frame, display, cv::Size(cvRound(frame.cols * scale), cvRound(frame.rows * scale)), 0, 0, cv::INTER_AREA); cv::imshow("SOP C++ TensorRT", display); const int key = cv::waitKey(1); if (key == 27 || key == 'q' || key == 'Q') break; } if (args.duration > 0 && elapsed >= args.duration) break; if (frames % 30 == 0) std::cout << "frames=" << frames << " analyzed=" << analyzed << " fps=" << fps << " latency_ms=" << latency << " yolo_frames=" << detectionFrames << " pose_frames=" << poseFrames << " source_fps=" << sourceFps << " obb_stride=" << args.obbStride << " person_stride=" << args.personStride << " pose_stride=" << args.poseStride << '\n'; }
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count(); if (writer.isOpened()) writer.release(); std::cout << std::fixed << std::setprecision(3) << "status=complete frames=" << frames << " analyzed=" << analyzed << " elapsed_s=" << elapsed << " inference_fps=" << analyzed / std::max(elapsed, 1e-6) << " latency_mean_ms=" << totalMs / std::max<uint64_t>(analyzed, 1) << " yolo_frames=" << detectionFrames << " pose_frames=" << poseFrames << " source_fps=" << sourceFps << '\n'; cv::destroyAllWindows(); return 0;
    } catch (const std::exception& error) { std::cerr << "status=failed error=" << error.what() << '\n'; return 1; }
}


#endif // SOP_INFER_EMBEDDED
