#pragma once

#include <cstdint>
#include <string>
#include <vector>

class Puzzle {
public:
    static constexpr size_t MAX_INDEXES_BUFFER = 128 * 1024;
    static constexpr uint64_t INVALID_INDEX = uint64_t(-1);

public:
    virtual int OperatorsCount() const = 0;
    virtual uint64_t IndexesCount() const = 0;

    virtual std::string ToString(uint64_t index) = 0;

    virtual uint64_t Parse(std::string state) = 0;

    virtual void Expand(
        const std::vector<uint64_t>& indexes,
        const std::vector<int>& usedOperatorBits,
        std::vector<uint64_t>& expandedIndexes,
        std::vector<int>& expandedOperators) = 0;
};

class ExpandBuffer {
public:
    ExpandBuffer(Puzzle& puzzle) 
        : puzzle(puzzle)
    {
        indexes.reserve(Puzzle::MAX_INDEXES_BUFFER);
        usedOperatorBits.reserve(Puzzle::MAX_INDEXES_BUFFER);
    }

    template<typename F>
    void Add(uint64_t index, int usedOperatorBits, F func) {
        indexes.push_back(index);
        usedOperatorBits.push_back(0);
        if (indexes.size() == Puzzle::MAX_INDEXES_BUFFER) {
            Expand(func);
        }
    }

    template<typename F>
    void Finish(F func) {
        if (indexes.size() > 0) {
            Expand(func);
        }
    }

private:
    template<typename F>
    uint64_t Expand(F func) {
        puzzle.Expand(indexes, usedOperatorBits, childIndexes, childOperators);
        for (size_t i = 0; i < childIndexes.size(); i++) {
            auto childIndex = childIndexes[i];
            auto op = childOperators[i];
            if (childIndex != puzzle.INVALID_INDEX) {
                func(childIndex, op);
            }
        }
        indexes.clear();
        usedOperatorBits.clear();
        childIndexes.clear();
        childOperators.clear();
    }

private:
    Puzzle& puzzle;
    std::vector<uint64_t> indexes;
    std::vector<int> usedOperatorBits;
    std::vector<uint64_t> childIndexes;
    std::vector<int> childOperators;
};