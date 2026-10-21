#pragma once
#include <cuda_runtime_api.h>
#include <cstddef>
struct SopInverseAffine { double m[6]; };
extern "C" cudaError_t sopPoseWarp(const unsigned char* bgr,int width,int height,
    size_t pitch,void* output,bool halfOutput,SopInverseAffine inverse,cudaStream_t stream);
extern "C" cudaError_t sopI420ToBgr(const unsigned char* i420,int width,int height,
    unsigned char* bgr,cudaStream_t stream);
