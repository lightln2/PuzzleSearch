#pragma once

#include "FourPegHanoiSimple.h"
#include "PuzzleGpu.h"

#include <string>

class FourPegHanoiOptimizedGPU : public PuzzleGpu {
public:
    FourPegHanoiOptimizedGPU(int size, bool useSymmetry)
        : m_SimplePuzzle(size, useSymmetry)
    {
        ensure(size <= 24);
    }

    virtual std::string Name() const;

    virtual int OperatorsCount() const { return 4; }

    virtual int BranchingFactor() const { return 6; }

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

protected:
    virtual void ExpandGpu(
        uint64_t* gpuIndexes,
        uint64_t* gpuExpanded,
        uint64_t count,
        CuStream stream);
private:
    FourPegHanoiSimple m_SimplePuzzle;
};
