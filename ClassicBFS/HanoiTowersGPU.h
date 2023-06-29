#pragma once

#include "GPU.h"

template<int size, bool useSymmetry>
void GpuHanoiTowersExpand(
    uint64_t* gpuIndexes,
    uint64_t* gpuExpanded,
    uint64_t count,
    CuStream stream);
