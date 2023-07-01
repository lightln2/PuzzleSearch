#include "PancakeOptimizedGpu.h"
#include "PancakeGpu.h"

std::string PancakeOptimizedGPU::Name() const {
    std::ostringstream stream;
    stream
        << "Pancake (GPU, 29 bits optimized), size: " << m_SimplePuzzle.Size()
        << "; invIdx: " << m_SimplePuzzle.InvIndex();
    return stream.str();
}

void PancakeOptimizedGPU::ExpandGpu(
    uint64_t* gpuIndexes,
    uint64_t* gpuExpanded,
    uint64_t count,
    CuStream stream)
{
    ensure(false);
}

void PancakeOptimizedGPU::Expand(
    std::vector<uint64_t>& indexes,
    std::vector<int>& usedOperatorBits,
    std::vector<uint64_t>& expandedIndexes,
    std::vector<int>& expandedOperators,
    ExpandHint hint)
{
    SetupOutputBuffers(expandedIndexes, expandedOperators);

    if (indexes.size() == 0) return;

    auto br = BranchingFactor();

    CombineIndexAndOpBits(indexes, usedOperatorBits);

    auto* stream = AquireStream();

    CopyToGpu(&indexes[0], stream->gpuSrc, indexes.size(), stream->stream);

    if (hint.SegmentBits == 29 && m_SimplePuzzle.Size() > 12) {
        if (hint.CrossSegment) {
            //PancakeExpand(stream->gpuSrc, stream->gpuDst, m_SimplePuzzle.Size(), m_SimplePuzzle.InvIndex(), indexes.size(), stream->stream);
            //expandedIndexes.resize(indexes.size() * br);
            //expandedOperators.resize(indexes.size() * br);
            //CopyFromGpu(stream->gpuDst, &expandedIndexes[0], expandedIndexes.size(), stream->stream);
            PancakeExpandCrossSegment(stream->gpuSrc, stream->gpuDst, m_SimplePuzzle.Size(), m_SimplePuzzle.InvIndex(), indexes.size(), stream->stream);
            expandedIndexes.resize(indexes.size() * (br - 11));
            expandedOperators.resize(indexes.size() * (br - 11));
            CopyFromGpu(stream->gpuDst, &expandedIndexes[0], expandedIndexes.size(), stream->stream);
        }
        else {
            PancakeExpandInSegment(stream->gpuSrc, stream->gpuDst, m_SimplePuzzle.Size(), m_SimplePuzzle.InvIndex(), indexes.size(), stream->stream);
            expandedIndexes.resize(indexes.size() * 11);
            expandedOperators.resize(indexes.size() * 11);
            CopyFromGpu(stream->gpuDst, &expandedIndexes[0], expandedIndexes.size(), stream->stream);
        }
    }
    else {
        PancakeExpand(stream->gpuSrc, stream->gpuDst, m_SimplePuzzle.Size(), m_SimplePuzzle.InvIndex(), indexes.size(), stream->stream);
        expandedIndexes.resize(indexes.size() * br);
        expandedOperators.resize(indexes.size() * br);
        CopyFromGpu(stream->gpuDst, &expandedIndexes[0], expandedIndexes.size(), stream->stream);
    }

    ReleaseStream(stream);

    SplitIndexAndOps(expandedIndexes, expandedOperators);
}
