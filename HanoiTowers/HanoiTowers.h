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

    static int MaxSegments() { return int(IndexesCount() >> 32); }

    static std::string ToString(uint64_t index);

    static uint64_t Parse(std::string stateStr);

    static std::pair<int, uint32_t> SplitIndex(uint64_t index) {
        return { int(index >> 32), uint32_t(index) };
    }

    static void Expand(const std::vector<uint64_t>& indexes, std::vector<uint64_t>& children);

    static void ExpandInSegment(int segment, size_t count, const uint32_t* indexes, std::vector<uint32_t>& children);

    static void ExpandCrossSegment(int segment, const std::vector<uint32_t>& indexes, std::vector<uint64_t>& children);

private:
    static void Expand(uint64_t index, std::vector<uint64_t>& children);

    static void ExpandInSegment(int segment, uint32_t index, std::vector<uint32_t>& children);
};