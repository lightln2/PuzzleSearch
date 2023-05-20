#pragma once

#include "../Common/Puzzle.h"

#include <cstdint>
#include <string>

class SlidingTilePuzzleGpu : public Puzzle {
public:
    SlidingTilePuzzleGpu(int width, int height);
    ~SlidingTilePuzzleGpu();

    virtual int OperatorsCount() const { return 4; }

    virtual uint64_t IndexesCount() const;

    virtual std::string ToString(uint64_t index);

    virtual uint64_t Parse(std::string state);

    virtual void Expand(
        std::vector<uint64_t>& indexes,
        std::vector<int>& usedOperatorBits,
        std::vector<uint64_t>& expandedIndexes,
        std::vector<int>& expandedOperators);

private:
    int m_Width;
    int m_Height;
    int m_Size;
    uint64_t* gpuSrc;
    uint64_t* gpuDst;
};
