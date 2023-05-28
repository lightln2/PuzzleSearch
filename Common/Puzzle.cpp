#include "Puzzle.h"

std::atomic<uint64_t> ExpandBuffer::m_StatExpandedNodes{ 0 };
std::atomic<uint64_t> ExpandBuffer::m_StatExpandedNanos{ 0 };
std::atomic<uint64_t> ExpandBuffer::m_StatExpandedTimes{ 0 };

void ExpandBuffer::PrintStats() {
    std::cerr
        << "Expand: " << WithDecSep(m_StatExpandedTimes) << " times, "
        << WithDecSep(m_StatExpandedNodes) << " nodes in "
        << WithTime(m_StatExpandedNanos)
        << std::endl;
}
