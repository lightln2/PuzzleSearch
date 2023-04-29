#pragma once

#include <cstdint>
#include <string>

class HashSlidingPuzzle {
public:
    static constexpr uint64_t INVALID_INDEX = uint64_t(-1);

    HashSlidingPuzzle(int width, int height);

    int MaxFanout();
    int OperatorBitsCount();
    int SegmentsCount();
    uint64_t MaxIndexesPerSegment();

    std::string ToString(uint32_t segment, uint32_t index);
    std::pair<uint32_t, uint32_t> Parse(std::string state);

    int Expand(uint64_t index, int usedOperatorBits, uint64_t* children, int* usedOperatorBit);

private:
    int m_Width;
    int m_Height;
    int m_Size;
};
