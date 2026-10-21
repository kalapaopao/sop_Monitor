#include "batch_kernels.hpp"
#include <cuda_fp16.h>
#include <cmath>
template<class T> __device__ T convertedBatch(float x);
template<> __device__ float convertedBatch<float>(float x) { return x; }
template<> __device__ __half convertedBatch<__half>(float x) { return __float2half(x); }
template<class T> __global__ void rectLetterbox(SopBatchView src,T* dst,int dw,int dh,int rw,int rh,int left,int top) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=dw*dh)return;
    const int x=i%dw,y=i/dw;
    for(int c=0;c<3;++c) {
        float v=114.f;
        if(x>=left && x<left+rw && y>=top && y<top+rh) {
            const float fx=fmaxf(0.f,fminf((x-left+.5f)*src.width/rw-.5f,src.width-1.f));
            const float fy=fmaxf(0.f,fminf((y-top+.5f)*src.height/rh-.5f,src.height-1.f));
            const int x0=int(floorf(fx)),y0=int(floorf(fy)),x1=min(x0+1,src.width-1),y1=min(y0+1,src.height-1);
            const float ax=fx-x0,ay=fy-y0; const int bgr=2-c;
            const float a=src.pixels[y0*src.pitch+3*x0+bgr]*(1-ax)+src.pixels[y0*src.pitch+3*x1+bgr]*ax;
            const float b=src.pixels[y1*src.pitch+3*x0+bgr]*(1-ax)+src.pixels[y1*src.pitch+3*x1+bgr]*ax;
            v=floorf(a*(1-ay)+b*ay+.5f);
        }
        dst[c*dw*dh+i]=convertedBatch<T>(v/255.f);
    }
}
extern "C" cudaError_t sopRectLetterbox(const SopBatchView& src,void* dst,bool half,int dw,int dh,cudaStream_t stream) {
    if(!src.pixels || !dst || src.width<=0 || src.height<=0 || dw<=0 || dh<=0)return cudaErrorInvalidValue;
    const float ratio=fminf(float(dw)/src.width,float(dh)/src.height);
    const int rw=int(std::round(src.width*ratio)),rh=int(std::round(src.height*ratio));
    const int blocks=(dw*dh+255)/256;
    if(half)rectLetterbox<<<blocks,256,0,stream>>>(src,static_cast<__half*>(dst),dw,dh,rw,rh,(dw-rw)/2,(dh-rh)/2);
    else rectLetterbox<<<blocks,256,0,stream>>>(src,static_cast<float*>(dst),dw,dh,rw,rh,(dw-rw)/2,(dh-rh)/2);
    return cudaGetLastError();
}
