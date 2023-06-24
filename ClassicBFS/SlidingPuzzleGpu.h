#pragma once

#include "GPU.h"

void GpuSlidingTilePuzzleSimpleExpand(
    uint64_t* gpuIndexes,
    uint64_t* gpuExpanded,
    int width,
    int size,
    uint64_t count,
    CuStream stream);

template<int width, int height>
void GpuSlidingTilePuzzleOptimizedExpand(
    uint64_t* gpuIndexes,
    uint64_t* gpuExpanded,
    uint64_t count,
    CuStream stream);
