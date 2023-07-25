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
        children.reserve(MAX_INDEXES * 6);
    }

    template<typename F>
    void Add(uint64_t index, F func) {
        indexes.push_back(index);
        if (indexes.size() >= MAX_INDEXES) {
            Expand(func);
        }
    }

    template<typename F>
    void Finish(F func) {
        if (indexes.size() > 0) {
            Expand(func);
        }
    }

    static void PrintStats() {
        std::cerr
            << "Expand: " << WithDecSep(m_StatExpandedTimes) << " times, "
            << WithDecSep(m_StatExpandedNodes) << " nodes in "
            << WithTime(m_StatExpandedNanos)
            << std::endl;
    }

private:
    template<typename F>
    void Expand(F func) {
        Timer expandTimer;

        HanoiTowers<size>::Expand(indexes, children);

        m_StatExpandedTimes++;
        m_StatExpandedNodes += indexes.size();
        m_StatExpandedNanos += expandTimer.Elapsed();

        for (auto child : children) {
            func(child);
        }

        indexes.clear();
        children.clear();
    }

private:
    std::vector<uint64_t> indexes;
    std::vector<uint64_t> children;

private:
    static std::atomic<uint64_t> m_StatExpandedNodes;
    static std::atomic<uint64_t> m_StatExpandedNanos;
    static std::atomic<uint64_t> m_StatExpandedTimes;
};