#pragma once

#include <cstdint>
#include <string>
#include <vector>

class Puzzle {
public:
    static constexpr size_t MAX_INDEXES_BUFFER = 128 * 1024;
    static constexpr uint64_t INVALID_INDEX = uint64_t(-1);

public:
    virtual int OperatorBitsCount() const = 0;

    virtual uint64_t IndexesCount() = 0;

    virtual std::string ToString(uint64_t index) = 0;

    virtual uint64_t Parse(std::string state) = 0;

    virtual void Expand(
        const std::vector<uint64_t>& indexes,
        const std::vector<int>& usedOperatorBits,
        std::vector<uint64_t>& expandedIndexes,
        std::vector<int>& expandedOperators) = 0;
};
