#pragma once
#include <cuda_runtime_api.h>
#include <cstddef>
struct SopBatchView { const unsigned char* pixels; int width,height; size_t pitch; };
extern "C" cudaError_t sopRectLetterbox(const SopBatchView&,void*,bool,int,int,cudaStream_t);
