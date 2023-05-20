#pragma once

#include <cstdint>

uint64_t* CreateGPUBuffer(int count);
void DestroyGPUBuffer(uint64_t* gpuBuffer);
void CopyToGpu(uint64_t* buffer, uint64_t* gpuBuffer, size_t count);
void CopyFromGpu(uint64_t* gpuBuffer, uint64_t* buffer, size_t count);

void GpuSlidingTilePuzzleSimpleExpand(
    uint64_t* gpuIndexes,
    uint64_t* gpuExpanded,
    int width,
    int size,
    uint64_t count);
