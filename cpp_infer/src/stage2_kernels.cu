#include "stage2_kernels.hpp"
#include <cuda_fp16.h>
template<typename T> __device__ T poseValue(float value);
template<> __device__ float poseValue<float>(float value) { return value; }
template<> __device__ __half poseValue<__half>(float value) { return __float2half(value); }
__device__ int at(const unsigned char* p,int w,int h,size_t pitch,int x,int y,int c) {
    return x>=0 && y>=0 && x<w && y<h ? p[y*pitch+x*3+c] : 0;
}
template<typename T> __global__ void poseWarp(const unsigned char* src,int w,int h,
    size_t pitch,T* dst,SopInverseAffine inv) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=192*256) return;
    const int x=i%192,y=i/192;
    // Match warpAffine's AB_BITS=10 mapping and 1/32 interpolation table.
    const int X=(__double2int_rn(inv.m[0]*x*1024.)+__double2int_rn((inv.m[1]*y+inv.m[2])*1024.)+16)>>5;
    const int Y=(__double2int_rn(inv.m[3]*x*1024.)+__double2int_rn((inv.m[4]*y+inv.m[5])*1024.)+16)>>5;
    const int ix=X>>5,iy=Y>>5,fx=X&31,fy=Y&31;
    const double mean[3]={123.675,116.28,103.53}, stddev[3]={58.395,57.12,57.375};
    for(int c=0;c<3;++c) {
        const int sum=at(src,w,h,pitch,ix,iy,c)*(32-fx)*(32-fy)+at(src,w,h,pitch,ix+1,iy,c)*fx*(32-fy)
            +at(src,w,h,pitch,ix,iy+1,c)*(32-fx)*fy+at(src,w,h,pitch,ix+1,iy+1,c)*fx*fy;
        const int pixel=(sum+512)>>10;
        // Preserve the established RTMW BGR channel order. No silent RGB change.
        dst[c*192*256+i]=poseValue<T>(__double2float_rn((double(pixel)-mean[c])/stddev[c]));
    }
}
__device__ unsigned char clipped(float x) { return (unsigned char)fminf(255.f,fmaxf(0.f,roundf(x))); }
__global__ void i420ToBgr(const unsigned char* src,int w,int h,unsigned char* dst) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=w*h) return;
    const int x=i%w,y=i/w,uv=(y/2)*(w/2)+x/2;
    const float Y=1.16438356f*(int(src[i])-16), U=int(src[w*h+uv])-128, V=int(src[w*h*5/4+uv])-128;
    dst[3*i]=clipped(Y+2.01723214f*U);
    dst[3*i+1]=clipped(Y-.39176229f*U-.81296765f*V);
    dst[3*i+2]=clipped(Y+1.59602679f*V);
}
extern "C" cudaError_t sopPoseWarp(const unsigned char* src,int w,int h,size_t pitch,
    void* dst,bool half,SopInverseAffine inv,cudaStream_t stream) {
    if(!src || !dst || w<=0 || h<=0 || pitch<size_t(w)*3) return cudaErrorInvalidValue;
    if(half) poseWarp<<<192,256,0,stream>>>(src,w,h,pitch,(__half*)dst,inv);
    else poseWarp<<<192,256,0,stream>>>(src,w,h,pitch,(float*)dst,inv);
    return cudaGetLastError();
}
extern "C" cudaError_t sopI420ToBgr(const unsigned char* src,int w,int h,unsigned char* dst,cudaStream_t stream) {
    if(!src || !dst || w<=0 || h<=0 || w%2 || h%2) return cudaErrorInvalidValue;
    i420ToBgr<<<(w*h+255)/256,256,0,stream>>>(src,w,h,dst);
    return cudaGetLastError();
}
