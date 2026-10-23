#include <NvInfer.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
extern "C" int sopRegisterSiLU();
int main() {
    if (sopRegisterSiLU()!=1) return 1;
    auto* creator = getPluginRegistry()->getPluginCreator("SopSiLU","1","");
    if (!creator) return 1;
    auto* base = creator->createPlugin("test",nullptr);
    auto* plugin = dynamic_cast<nvinfer1::IPluginV2DynamicExt*>(base);
    if (!plugin) return 1;
    constexpr int n=1027;
    std::vector<float> values(n), floatOut(n);
    std::vector<__half> halfIn(n), halfOut(n);
    for (int i=0;i<n;++i) { values[i]=(i-513)*.04f; halfIn[i]=__float2half(values[i]); }
    void *x=nullptr,*y=nullptr;cudaStream_t stream{};
    auto ok=[](cudaError_t e) { if(e!=cudaSuccess) std::cerr<<cudaGetErrorString(e)<<'\n';return e==cudaSuccess; };
    if(!ok(cudaMalloc(&x,n*sizeof(float))) || !ok(cudaMalloc(&y,n*sizeof(float))) || !ok(cudaStreamCreate(&stream)))return 1;
    bool passed=true;
    for (bool half : {false,true}) {
        nvinfer1::PluginTensorDesc desc{};
        desc.dims=nvinfer1::Dims4{1,1,1,n};desc.format=nvinfer1::TensorFormat::kLINEAR;
        desc.type=half?nvinfer1::DataType::kHALF:nvinfer1::DataType::kFLOAT;
        const size_t bytes=n*(half?sizeof(__half):sizeof(float));
        const void* input=half?static_cast<const void*>(halfIn.data()):static_cast<const void*>(values.data());
        void* output=half?static_cast<void*>(halfOut.data()):static_cast<void*>(floatOut.data());
        if(!ok(cudaMemcpyAsync(x,input,bytes,cudaMemcpyHostToDevice,stream)))return 1;
        const void* inputs[]{x};void* outputs[]{y};
        if(plugin->enqueue(&desc,&desc,inputs,outputs,nullptr,stream)!=0)return 1;
        if(!ok(cudaMemcpyAsync(output,y,bytes,cudaMemcpyDeviceToHost,stream)) || !ok(cudaStreamSynchronize(stream)))return 1;
        float maxError=0;
        for(int i=0;i<n;++i) {
            const float v=half?__half2float(halfIn[i]):values[i];
            const float reference=v/(1.f+std::exp(-v));
            const float actual=half?__half2float(halfOut[i]):floatOut[i];
            const float error=std::abs(actual-reference);maxError=std::max(maxError,error);
            if(!std::isfinite(actual) || error>(half?.001f:1e-6f)*std::max(1.f,std::abs(reference)))passed=false;
        }
        std::cout<<(half?"FP16":"FP32")<<" max_abs_error="<<maxError<<'\n';
    }
    cudaStreamDestroy(stream);cudaFree(x);cudaFree(y);plugin->destroy();
    return passed?0:1;
}
