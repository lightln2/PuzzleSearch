#include "FourPegHanoiGPU.h"
#include "HanoiTowersGPU.h"

std::string FourPegHanoiGPU::Name() const {
    std::ostringstream stream;
    stream
        << "Four-Peg Hanoi Towers (GPU), size=" << m_SimplePuzzle.Size()
        << "; symmetry=" << m_SimplePuzzle.UseSymmetry();
    return stream.str();
}

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

