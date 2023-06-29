#pragma once

#include "Util.h"

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

struct ExpandHint {
    int SegmentBits = 0;
    bool CrossSegment = false;
};

class Puzzle {
public:
    static constexpr size_t MAX_INDEXES_BUFFER = 2 * 1024 * 1024;
    static constexpr uint64_t INVALID_INDEX = uint64_t(-1);

public:
    virtual std::string Name() const = 0;
    virtual int OperatorsCount() const = 0;
    virtual uint64_t IndexesCount() const = 0;
    virtual bool HasOddLengthCycles() const = 0;

    virtual std::string ToString(uint64_t index) = 0;

    virtual uint64_t Parse(std::string state) = 0;

    virtual void Expand(
        std::vector<uint64_t>& indexes,
        std::vector<int>& usedOperatorBits,
        std::vector<uint64_t>& expandedIndexes,
        std::vector<int>& expandedOperators,
        ExpandHint hint) = 0;

};

class ExpandBuffer {
public:
    ExpandBuffer(Puzzle& puzzle) 
        : puzzle(puzzle)
    {
        indexes.reserve(Puzzle::MAX_INDEXES_BUFFER);
        usedOperatorBits.reserve(Puzzle::MAX_INDEXES_BUFFER);
    }

    void SetExpandHint(int segmentBits, bool crossSegment) {
        hint.SegmentBits = segmentBits;
        hint.CrossSegment = crossSegment;
    }

    template<typename F>
    void Add(uint64_t index, int usedOperatorBits, F func) {
        indexes.push_back(index);
        this->usedOperatorBits.push_back(usedOperatorBits);
        if (indexes.size() == Puzzle::MAX_INDEXES_BUFFER) {
            Expand(func);
        }
    }

    template<typename F>
    void Finish(F func) {
        if (indexes.size() > 0) {
            Expand(func);
        }
        // reset hint
        SetExpandHint(0, false);
    }

    static void PrintStats();

private:
    template<typename F>
    void Expand(F func) {
        Timer expandTimer;

        puzzle.Expand(indexes, usedOperatorBits, childIndexes, childOperators, hint);

        m_StatExpandedTimes++;
        m_StatExpandedNodes += indexes.size();
        m_StatExpandedNanos += expandTimer.Elapsed();

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
    ExpandHint hint;
    std::vector<uint64_t> indexes;
    std::vector<int> usedOperatorBits;
    std::vector<uint64_t> childIndexes;
    std::vector<int> childOperators;

private:
    static std::atomic<uint64_t> m_StatExpandedNodes;
    static std::atomic<uint64_t> m_StatExpandedNanos;
    static std::atomic<uint64_t> m_StatExpandedTimes;
};