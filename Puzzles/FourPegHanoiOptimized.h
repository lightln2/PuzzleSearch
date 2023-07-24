#pragma once

#include "../Common/Puzzle.h"

#include <cstdint>
#include <string>

class FourPegHanoiOptimized : public Puzzle {
public:
    FourPegHanoiOptimized(int size, bool useSymmetry = false);

    virtual std::string Name() const;

    virtual int OperatorsCount() const { return 4; }

    virtual int BranchingFactor() const { return 6; }

    virtual uint64_t IndexesCount() const;

    virtual bool HasOddLengthCycles() const { return true; }

    virtual std::string ToString(uint64_t index);

    virtual uint64_t Parse(std::string stateStr);

    virtual void Expand(
        std::vector<uint64_t>& indexes,
        std::vector<int>& usedOperatorBits,
        std::vector<uint64_t>& expandedIndexes,
        std::vector<int>& expandedOperators,
        ExpandHint hint);

    int Size() const { return m_Size; }
    bool UseSymmetry() const { return m_UseSymmetry; }

private:
    void Expand(uint64_t index, int op, std::vector<uint64_t>& expandedIndexes, std::vector<int>& expandedOperators);

private:
    int m_Size;
    bool m_UseSymmetry;
};
