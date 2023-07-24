#pragma once

#include "PancakeOptimized.h"
#include "PuzzleGpu.h"

#include <string>

class PancakeOptimizedGPU : public PuzzleGpu {
public:
    PancakeOptimizedGPU(int size, bool invIndex)
        : m_SimplePuzzle(size, invIndex)
    {
        ensure(size <= 16);
    }

    virtual std::string Name() const;

    virtual int OperatorsCount() const { return m_SimplePuzzle.OperatorsCount(); }

    virtual int BranchingFactor() const { return m_SimplePuzzle.BranchingFactor(); }

    virtual uint64_t IndexesCount() const { return m_SimplePuzzle.IndexesCount(); }

    virtual bool HasOddLengthCycles() const { return m_SimplePuzzle.HasOddLengthCycles(); }

    virtual std::string ToString(uint64_t index) { return m_SimplePuzzle.ToString(index); }

    virtual uint64_t Parse(std::string state) { return m_SimplePuzzle.Parse(state); }

    virtual void Expand(
        std::vector<uint64_t>& indexes,
        std::vector<int>& usedOperatorBits,
        std::vector<uint64_t>& expandedIndexes,
        std::vector<int>& expandedOperators,
        ExpandHint hint);

    virtual void CrossSegmentPostProcess(
        int op,
        int segment,
        int segmentBits,
        Buffer<uint32_t>& expandedIndexes);

protected:
    virtual void ExpandGpu(
        uint64_t* gpuIndexes,
        uint64_t* gpuExpanded,
        uint64_t count,
        CuStream stream);
private:
    PancakeOptimized m_SimplePuzzle;
};
