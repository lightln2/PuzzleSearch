#pragma once

#include "GPU.h"

void PancakeExpand(
    uint64_t* gpuIndexes,
    uint64_t* gpuExpanded,
    int size,
    bool invIndex,
    uint64_t count,
    CuStream stream);

void PancakeExpandInSegment(
    uint64_t* gpuIndexes,
    uint64_t* gpuExpanded,
    int size,
    bool invIndex,
    uint64_t count,
    CuStream stream);

void PancakeExpandCrossSegment(
    uint64_t* gpuIndexes,
    uint64_t* gpuExpanded,
    int size,
    bool invIndex,
    uint64_t count,
    CuStream stream);
