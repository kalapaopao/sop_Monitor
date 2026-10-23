#include <NvInfer.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <new>
#include <string>

namespace {
template<class T> __global__ void silu(const T* x, T* y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { const float v = static_cast<float>(x[i]); y[i] = static_cast<T>(v / (1.f + expf(-v))); }
}
class SopSiLU final : public nvinfer1::IPluginV2DynamicExt {
public:
    const char* getPluginType() const noexcept override { return "SopSiLU"; }
    const char* getPluginVersion() const noexcept override { return "1"; }
    int getNbOutputs() const noexcept override { return 1; }
    int initialize() noexcept override { return 0; }
    void terminate() noexcept override {}
    size_t getSerializationSize() const noexcept override { return 0; }
    void serialize(void*) const noexcept override {}
    void destroy() noexcept override { delete this; }
    SopSiLU* clone() const noexcept override { return new(std::nothrow) SopSiLU(*this); }
    void setPluginNamespace(const char* ns) noexcept override { ns_ = ns ? ns : ""; }
    const char* getPluginNamespace() const noexcept override { return ns_.c_str(); }
    nvinfer1::DataType getOutputDataType(int, const nvinfer1::DataType* types, int) const noexcept override { return types[0]; }
    nvinfer1::DimsExprs getOutputDimensions(int, const nvinfer1::DimsExprs* inputs, int, nvinfer1::IExprBuilder&) noexcept override { return inputs[0]; }
    bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* io, int, int) noexcept override {
        if (io[pos].format != nvinfer1::TensorFormat::kLINEAR) return false;
        if (pos == 0) return io[0].type == nvinfer1::DataType::kFLOAT || io[0].type == nvinfer1::DataType::kHALF;
        return io[pos].type == io[0].type;
    }
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc*,int,const nvinfer1::DynamicPluginTensorDesc*,int) noexcept override {}
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc*,int,const nvinfer1::PluginTensorDesc*,int) const noexcept override { return 0; }
    int enqueue(const nvinfer1::PluginTensorDesc* input, const nvinfer1::PluginTensorDesc*,
                const void* const* inputs, void* const* outputs, void*, cudaStream_t stream) noexcept override {
        int n = 1;
        for (int i=0;i<input[0].dims.nbDims;++i) n *= input[0].dims.d[i];
        if (input[0].type == nvinfer1::DataType::kFLOAT)
            silu<<<(n+255)/256,256,0,stream>>>(static_cast<const float*>(inputs[0]),static_cast<float*>(outputs[0]),n);
        else silu<<<(n+255)/256,256,0,stream>>>(static_cast<const __half*>(inputs[0]),static_cast<__half*>(outputs[0]),n);
        return cudaPeekAtLastError() == cudaSuccess ? 0 : 1;
    }
private:
    std::string ns_;
};
class SopSiLUCreator final : public nvinfer1::IPluginCreator {
public:
    const char* getPluginName() const noexcept override { return "SopSiLU"; }
    const char* getPluginVersion() const noexcept override { return "1"; }
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override { return &fields_; }
    nvinfer1::IPluginV2* createPlugin(const char*, const nvinfer1::PluginFieldCollection*) noexcept override {
        auto* p = new(std::nothrow) SopSiLU; if (p) p->setPluginNamespace(ns_.c_str()); return p;
    }
    nvinfer1::IPluginV2* deserializePlugin(const char*, const void*, size_t length) noexcept override {
        if (length != 0) return nullptr;
        return createPlugin(nullptr,nullptr);
    }
    void setPluginNamespace(const char* ns) noexcept override { ns_ = ns ? ns : ""; }
    const char* getPluginNamespace() const noexcept override { return ns_.c_str(); }
private:
    nvinfer1::PluginFieldCollection fields_{0,nullptr};
    std::string ns_;
};
REGISTER_TENSORRT_PLUGIN(SopSiLUCreator);
}
extern "C" int sopRegisterSiLU() { return 1; }
