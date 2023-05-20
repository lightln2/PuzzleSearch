#pragma once

#include "../Common/Puzzle.h"

#include <cstdint>
#include <string>

class SlidingTilePuzzleSimple : public Puzzle {
public:
    SlidingTilePuzzleSimple(int width, int height);

    virtual int OperatorsCount() const { return 4; }

    virtual uint64_t IndexesCount() const;

    virtual std::string ToString(uint64_t index);

    virtual uint64_t Parse(std::string state);

    virtual void Expand(
        const std::vector<uint64_t>& indexes,
        const std::vector<int>& usedOperatorBits,
        std::vector<uint64_t>& expandedIndexes,
        std::vector<int>& expandedOperators);

private:
    void Expand(uint64_t index, int op, uint64_t* children, int* operators);

private:
    int m_Width;
    int m_Height;
    int m_Size;
};
