#pragma once

#include "../Common/Puzzle.h"

#include <cstdint>
#include <string>

class PancakeSimple : public Puzzle {
public:
    PancakeSimple(int size);

    virtual std::string Name() const;

    virtual int OperatorsCount() const { return m_Size - 1; }

    virtual int BranchingFactor() const { return m_Size - 1; }

    virtual uint64_t IndexesCount() const;

    virtual bool HasOddLengthCycles() const { return true; }

    virtual std::string ToString(uint64_t index);

    virtual uint64_t Parse(std::string state);

    virtual void Expand(
        std::vector<uint64_t>& indexes,
        std::vector<int>& usedOperatorBits,
        std::vector<uint64_t>& expandedIndexes,
        std::vector<int>& expandedOperators,
        ExpandHint hint);

private:
    void Expand(uint64_t index, int opBits, std::vector<uint64_t>& expandedIndexes, std::vector<int>& expandedOperators);

private:
    int m_Size;
};
