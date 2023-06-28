#include "FourPegHanoiGPU.h"
#include "HanoiTowersGPU.h"

void FourPegHanoiGPU::ExpandGpu(
    uint64_t* gpuIndexes,
    uint64_t* gpuExpanded,
    uint64_t count,
    CuStream stream)
{
    GpuHanoiTowersExpand(
        gpuIndexes,
        gpuExpanded,
        m_SimplePuzzle.Size(),
        m_SimplePuzzle.UseSymmetry(),
        count,
        stream);
}
