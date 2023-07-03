#pragma once

#include "GPU.h"

void GpuHanoiTowersExpand(
    uint64_t* gpuIndexes,
    uint64_t* gpuExpanded,
    int size,
    bool useSymmetry,
    uint64_t count,
    CuStream stream);

void GpuHanoiTowersExpandOptimized(
    uint64_t* gpuIndexes,
    uint64_t* gpuExpanded,
    int size,
    bool useSymmetry,
    uint64_t count,
    CuStream stream);
