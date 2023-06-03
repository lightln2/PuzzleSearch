#pragma once

#include <cstdint>

using CuStream = void*;

uint64_t* CreateGPUBuffer(int count);
void DestroyGPUBuffer(uint64_t* gpuBuffer);
CuStream CreateCudaStream();
void DestroyCudaStream(CuStream stream);

void CopyToGpu(uint64_t* buffer, uint64_t* gpuBuffer, size_t count, CuStream stream);
void CopyFromGpu(uint64_t* gpuBuffer, uint64_t* buffer, size_t count, CuStream stream);

void GpuSlidingTilePuzzleSimpleExpand(
    uint64_t* gpuIndexes,
    uint64_t* gpuExpanded,
    int width,
    int size,
    uint64_t count,
    CuStream stream);
