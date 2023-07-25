#pragma once

#include "../Puzzles/GPU.h"
#include "../Common/Util.h"


struct GpuBuffer {
    static constexpr size_t SRCBUFSIZE = 16 * 1024 * 1024;
    static constexpr size_t DSTBUFSIZE = 16 * 1024 * 1024;

    CuStream Stream;
    uint32_t* Indexes;
    uint32_t* Children;

    GpuBuffer();
    ~GpuBuffer();
};

