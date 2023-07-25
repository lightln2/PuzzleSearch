#pragma once

#include "../Common/Buffer.h"

#include <cstdint>
#include <string>
#include <vector>

template<int size>
class HanoiTowers {
public:
    static constexpr int Size = size;

    static uint64_t IndexesCount() { return 1ui64 << (size * 2); }

    static int MaxSegments() { return int((IndexesCount() + 0xFFFFFFFF) >> 32); }

    static std::string ToString(uint64_t index);

    static uint64_t Parse(std::string stateStr);

    static std::pair<int, uint32_t> SplitIndex(uint64_t index) {
        return { int(index >> 32), uint32_t(index) };
    }

    static void Expand(const std::vector<uint64_t>& indexes, std::vector<uint64_t>& children);

    static void ExpandInSegment(int segment, size_t count, const uint32_t* indexes, std::vector<uint32_t>& children);
    static void ExpandInSegmentWithoutSmallest(int segment, size_t count, const uint32_t* indexes, std::vector<uint32_t>& children);

    static void ExpandCrossSegment(int segment, const std::vector<uint32_t>& indexes, std::vector<uint64_t>& children);

public:

    static __forceinline bool InSegmentOnly(uint32_t index) {
        static constexpr uint32_t MASK = 0x55555555ui32;
        uint32_t p0 = index >> 1;
        uint32_t p1 = index;
        uint32_t p0i = ~p0;
        uint32_t p1i = ~p1;
        bool z0 = ((p0i & p1i) & MASK) != 0;
        bool z1 = ((p0i & p1) & MASK) != 0;
        bool z2 = ((p0 & p1i) & MASK) != 0;
        bool z3 = ((p0 & p1) & MASK) != 0;
        return int(z0) + int(z1) + int(z2) + int(z3) >= 3;
    }

    static __forceinline int PegsMult(uint64_t index) {
        static constexpr uint64_t BIT_MASK = 0x5555555555555555ui64;
        uint64_t p0 = index >> 1;
        uint64_t p1 = index;
        uint64_t pegs1 = (~p0 & p1) & BIT_MASK;
        uint64_t pegs2 = (p0 & ~p1) & BIT_MASK;
        return 6 - (int(pegs2 == 0) * 3) - int(pegs1 == 0);
    }

private:
    static void Expand(uint64_t index, std::vector<uint64_t>& children);
    static void ExpandCrossSegment(int segment, uint32_t index, std::vector<uint64_t>& children);

    static void ExpandInSegment(int segment, uint32_t index, std::vector<uint32_t>& children);
    static void ExpandInSegmentWithoutSmallest(int segment, uint32_t index, std::vector<uint32_t>& children);
    static void ExpandInSegmentNoSymmetry(int segment, uint32_t index, std::vector<uint32_t>& children);
};