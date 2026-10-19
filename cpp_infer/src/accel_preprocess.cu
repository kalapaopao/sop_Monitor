#include "accel_preprocess.hpp"
#include <cuda_fp16.h>

template<typename T> __device__ T converted(float x);
template<> __device__ float converted<float>(float x) { return x; }
template<> __device__ __half converted<__half>(float x) { return __float2half(x); }

template<typename T> __global__ void letterbox(const unsigned char* src,
    int w, int h, size_t pitch, T* dst, int side, int rw, int rh, int left, int top) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= side * side) return;
    const int x = i % side, y = i / side;
    for (int c = 0; c < 3; ++c) {
        float v = 114.f;
        if (x >= left && x < left + rw && y >= top && y < top + rh) {
            const float fx = fmaxf(0.f, fminf((x-left+.5f)*w/rw-.5f, w-1.f));
            const float fy = fmaxf(0.f, fminf((y-top+.5f)*h/rh-.5f, h-1.f));
            const int x0 = int(floorf(fx)), y0 = int(floorf(fy));
            const int x1 = min(x0+1,w-1), y1 = min(y0+1,h-1);
            const float ax = fx-x0, ay = fy-y0;
            const int bgr = 2-c;
            const float a = src[y0*pitch+3*x0+bgr]*(1-ax)+src[y0*pitch+3*x1+bgr]*ax;
            const float b = src[y1*pitch+3*x0+bgr]*(1-ax)+src[y1*pitch+3*x1+bgr]*ax;
            // CPU cv::resize produces an 8-bit image before normalization.
            v = floorf(a*(1-ay)+b*ay+.5f);
        }
        dst[c*side*side+i] = converted<T>(v/255.f);
    }
}

extern "C" cudaError_t sopLetterbox(const unsigned char* src, int w, int h,
    size_t pitch, void* dst, bool halfOutput, int side, int rw, int rh,
    int left, int top, cudaStream_t stream) {
    if (!src || !dst || w<=0 || h<=0 || side<=0 || rw<=0 || rh<=0 || pitch<size_t(w)*3)
        return cudaErrorInvalidValue;
    const int blocks = (side*side+255)/256;
    if (halfOutput) letterbox<<<blocks,256,0,stream>>>(src,w,h,pitch,(__half*)dst,side,rw,rh,left,top);
    else letterbox<<<blocks,256,0,stream>>>(src,w,h,pitch,(float*)dst,side,rw,rh,left,top);
    return cudaGetLastError();
}
