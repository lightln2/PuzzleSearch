#pragma once

#include "../Common/Puzzle.h"

#include <cstdint>
#include <string>

class FourPegHanoiSimple : public Puzzle {
public:
    FourPegHanoiSimple(int size, bool useSymmetry = false);

    virtual int OperatorsCount() const { return 4; }

    virtual uint64_t IndexesCount() const;

    virtual bool HasOddLengthCycles() const { return true; }

    virtual std::string ToString(uint64_t index);

    virtual uint64_t Parse(std::string stateStr);

    virtual void Expand(
        std::vector<uint64_t>& indexes,
        std::vector<int>& usedOperatorBits,
        std::vector<uint64_t>& expandedIndexes,
        std::vector<int>& expandedOperators);

    int Size() const { return m_Size; }
    bool UseSymmetry() const { return m_UseSymmetry; }

private:
    void Expand(uint64_t index, int op, uint64_t* children, int* operators);

private:
    int m_Size;
    bool m_UseSymmetry;
};
