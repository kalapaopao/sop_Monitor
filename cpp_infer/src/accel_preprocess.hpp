#pragma once
#include <cuda_runtime_api.h>
#include <cstddef>

// Writes centered letterbox RGB/CHW /255, with the same integer resize geometry
// as the CPU path. Each caller owns source/destination storage and its stream.
extern "C" cudaError_t sopLetterbox(const unsigned char* source, int width,
    int height, size_t pitch, void* output, bool halfOutput, int side,
    int resizedWidth, int resizedHeight, int left, int top, cudaStream_t stream);
