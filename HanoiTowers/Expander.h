#pragma once

#include "HanoiTowers.h"
#include "../Common/Util.h"

#include <iostream>
#include <vector>


template<int size>
class Expander {
public:
    static constexpr size_t MAX_INDEXES = 4 * 1024 * 1024;
public:
    Expander()
    {
        indexes.reserve(MAX_INDEXES);
        insegChildren.reserve(MAX_INDEXES * 6);
    }

    void AddCrossSegment(int segment, uint32_t index) {
        if (HanoiTowers<size>::InSegmentOnly(index)) return;
        indexes.push_back(index);
    }

    std::vector<uint64_t>& ExpandCrossSegment(int segment) {
        Timer expandTimer;

        crosssegChildren.clear();
        HanoiTowers<size>::ExpandCrossSegment(segment, indexes, crosssegChildren);

        m_StatXExpandedTimes++;
        m_StatXExpandedNodes += indexes.size();
        m_StatXExpandedNanos += expandTimer.Elapsed();

        indexes.clear();
        return crosssegChildren;
    }


    std::vector<uint32_t>& ExpandInSegment(int segment, size_t count, uint32_t* indexes) {
        Timer expandTimer;
        insegChildren.clear();
        HanoiTowers<size>::ExpandInSegment(segment, count, indexes, insegChildren);
        m_StatExpandedTimes++;
        m_StatExpandedNodes += count;
        m_StatExpandedNanos += expandTimer.Elapsed();
        return insegChildren;
    }

    std::vector<uint32_t>& ExpandInSegmentWithoutSmallest(int segment, size_t count, uint32_t* indexes) {
        Timer expandTimer;
        insegChildren.clear();
        HanoiTowers<size>::ExpandInSegmentWithoutSmallest(segment, count, indexes, insegChildren);
        m_StatExpandedTimes2++;
        m_StatExpandedNodes2 += count;
        m_StatExpandedNanos2 += expandTimer.Elapsed();
        return insegChildren;
    }

    static void PrintStats() {
        std::cerr
            << "Expand in-seg: " << WithDecSep(m_StatExpandedTimes) << " times, "
            << WithDecSep(m_StatExpandedNodes) << " nodes in "
            << WithTime(m_StatExpandedNanos)
            << std::endl;
        std::cerr
            << "Expand in-seg (3/6): " << WithDecSep(m_StatExpandedTimes2) << " times, "
            << WithDecSep(m_StatExpandedNodes2) << " nodes in "
            << WithTime(m_StatExpandedNanos2)
            << std::endl;
        std::cerr
            << "Expand x-seg: " << WithDecSep(m_StatXExpandedTimes) << " times, "
            << WithDecSep(m_StatXExpandedNodes) << " nodes in "
            << WithTime(m_StatXExpandedNanos)
            << std::endl;
    }

private:
    std::vector<uint32_t> indexes;
    std::vector<uint64_t> crosssegChildren;
    std::vector<uint32_t> insegChildren;

private:
    static std::atomic<uint64_t> m_StatExpandedNodes;
    static std::atomic<uint64_t> m_StatExpandedNanos;
    static std::atomic<uint64_t> m_StatExpandedTimes;
    static std::atomic<uint64_t> m_StatExpandedNodes2;
    static std::atomic<uint64_t> m_StatExpandedNanos2;
    static std::atomic<uint64_t> m_StatExpandedTimes2;
    static std::atomic<uint64_t> m_StatXExpandedNodes;
    static std::atomic<uint64_t> m_StatXExpandedNanos;
    static std::atomic<uint64_t> m_StatXExpandedTimes;
};
