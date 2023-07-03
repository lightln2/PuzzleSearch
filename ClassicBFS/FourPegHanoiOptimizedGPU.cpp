#include "FourPegHanoiOptimizedGPU.h"
#include "HanoiTowersGPU.h"

std::string FourPegHanoiOptimizedGPU::Name() const {
    std::ostringstream stream;
    stream
        << "Four-Peg Hanoi Optimized (GPU), size=" << m_SimplePuzzle.Size()
        << "; symmetry=" << m_SimplePuzzle.UseSymmetry();
    return stream.str();
}

void FourPegHanoiOptimizedGPU::ExpandGpu(
    uint64_t* gpuIndexes,
    uint64_t* gpuExpanded,
    uint64_t count,
    CuStream stream)
{
    GpuHanoiTowersExpandOptimized(
        gpuIndexes,
        gpuExpanded,
        m_SimplePuzzle.Size(),
        m_SimplePuzzle.UseSymmetry(),
        count,
        stream);
}

void FourPegHanoiOptimizedGPU::Expand(
    std::vector<uint64_t>& indexes,
    std::vector<int>& usedOperatorBits,
    std::vector<uint64_t>& expandedIndexes,
    std::vector<int>& expandedOperators,
    ExpandHint hint)
{
    bool filterXSeg = (hint.SegmentBits == 32 && hint.CrossSegment && m_SimplePuzzle.Size() > 16);

    auto fbInSegOnly = [](uint64_t index) {
        uint32_t idx = index & 0xFFFFFFFF;
        uint32_t p0 = idx >> 1;
        uint32_t p1 = idx;
        uint32_t p0i = ~p0;
        uint32_t p1i = ~p1;
        bool z0 = ((p0 & p1) & 0x55555555ui32) != 0;
        bool z1 = ((p0 & p1i) & 0x55555555ui32) != 0;
        bool z2 = ((p0i & p1) & 0x55555555ui32) != 0;
        bool z3 = ((p0i & p1i) & 0x55555555ui32) != 0;
        return int(z0) + int(z1) + int(z2) + int(z3) >= 3;
    };

    if (filterXSeg) {
        size_t src = 0, dst = 0;
        while (dst < indexes.size()) {
            if (fbInSegOnly(indexes[dst])) {
                dst++;
                continue;
            }
            else if (src != dst) {
                indexes[src] = indexes[dst];
                usedOperatorBits[src] = usedOperatorBits[dst];
            }
            src++;
            dst++;
        }
        if (src != dst) {
            indexes.resize(src);
            usedOperatorBits.resize(src);
        }
    }

    PuzzleGpu::Expand(indexes, usedOperatorBits, expandedIndexes, expandedOperators, hint);
}
