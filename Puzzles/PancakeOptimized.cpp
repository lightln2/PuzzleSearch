#include "PancakeOptimized.h"
#include "PermutationMap.h"
#include "../Common/Util.h"

#include <sstream>

namespace {
    bool HasOp(int op, int dir) {
        return op & (1 << dir);
    }

    void MoveInternal(int* newarr, int size, int count) {
        int start = size - count, end = size - 1;
        while (start < end) {
            std::swap(newarr[start], newarr[end]);
            start++;
            end--;
        }
    }

    void Move(int* arr, int* newarr, int size, int count) {
        memcpy(newarr, arr, 16 * sizeof(int));
        MoveInternal(newarr, size, count);
    }

    uint64_t OptPermutationRank(int* arr, int size) {
        PermutationCompact(arr, size);
        if (size <= 12) {
            uint64_t index = 0;
            for (int i = 0; i < size; i++) {
                index *= (size - i);
                index += arr[i];
            }

            return index;
        }
        else {
            uint64_t index = 0;
            for (int i = size - 12; i < size; i++) {
                index *= (size - i);
                index += arr[i];
            }

            uint64_t segment = 0;
            for (int i = 0; i < size - 12; i++) {
                segment *= (size - i);
                segment += arr[i];
            }

            return (segment << 29) | index;
        }
    }

    void OptPermutationUnrank(uint64_t index, int* arr, int size) {
        if (size <= 12) {
            for (int i = size - 1; i >= 0; i--) {
                arr[i] = index % (size - i);
                index /= (size - i);
            }
        }
        else {
            uint64_t segment = index >> 29;
            index &= ((1ui64 << 29) - 1);

            for (int i = size - 13; i >= 0; i--) {
                arr[i] = segment % (size - i);
                segment /= (size - i);
            }

            for (int i = size - 1; i >= size - 12; i--) {
                arr[i] = index % (size - i);
                index /= (size - i);
            }
        }

        PermutationUncompact(arr, size);
    }

}

PancakeOptimized::PancakeOptimized(int size, bool inverseIndex)
    : m_Size(size)
    , m_InverseIndex(inverseIndex)
{
    ensure(size > 1 && size <= 16);
}

std::string PancakeOptimized::Name() const {
    std::ostringstream stream;
    stream << "Pancake optimized (29 bit segment), size: " << m_Size << "; invIdx: " << m_InverseIndex;
    return stream.str();
}

uint64_t PancakeOptimized::IndexesCount() const {
    if (m_Size < 12) {
        uint64_t result = 1;
        for (int i = 1; i <= m_Size; i++) result *= i;
        return result;
    }
    else {
        uint64_t result = 1;
        for (int i = 13; i <= m_Size; i++) result *= i;
        return result << 29;
    }
}

std::string PancakeOptimized::ToString(uint64_t index) {
    std::ostringstream stream;
    int arr[16];
    OptPermutationUnrank(index, arr, m_Size);
    for (int i = 0; i < m_Size; i++) {
        if (i > 0) stream << ' ';
        stream << arr[i];
    }
    return stream.str();
}

uint64_t PancakeOptimized::Parse(std::string state) {
    std::istringstream stream(state);
    int arr[16];
    for (int i = 0; i < m_Size; i++) {
        stream >> arr[i];
    }
    return OptPermutationRank(arr, m_Size);
}

void PancakeOptimized::CrossSegmentPostProcess(int op, int segment, int segmentBits, Buffer<uint32_t>& expandedIndexes) {
    bool invert = m_InverseIndex && segmentBits == 29 && m_Size > 12;
    if (!invert) return;
    const uint64_t SEG_MASK = (1ui64 << 29) - 1;
    for (size_t i = 0; i < expandedIndexes.Size(); i++) {
        uint64_t index = (uint64_t(segment) << segmentBits) | expandedIndexes[i];
        int arr[16];
        OptPermutationUnrank(index, arr, m_Size);
        // invert!
        MoveInternal(arr, m_Size, 12);
        index = OptPermutationRank(arr, m_Size);
        ensure(index >> segmentBits == segment);
        expandedIndexes[i] = uint32_t(index & SEG_MASK);
    }
}

void PancakeOptimized::Expand(uint64_t index, int opBits, std::vector<uint64_t>& expandedIndexes, std::vector<int>& expandedOperators) {
    int arr[16];
    OptPermutationUnrank(index, arr, m_Size);

    int newarr[16];
    for (int op = 0; op < m_Size - 1; op++) {
        if (HasOp(opBits, op)) continue;
        Move(arr, newarr, m_Size, op + 2);
        expandedIndexes.push_back(OptPermutationRank(newarr, m_Size));
        expandedOperators.push_back(op);
    }
}

void PancakeOptimized::ExpandInSegment(uint64_t index, int opBits, std::vector<uint64_t>& expandedIndexes, std::vector<int>& expandedOperators) {
    int arr[16];
    OptPermutationUnrank(index, arr, m_Size);

    int newarr[16];
    for (int op = 0; op < 11; op++) {
        if (HasOp(opBits, op)) continue;
        Move(arr, newarr, m_Size, op + 2);
        expandedIndexes.push_back(OptPermutationRank(newarr, m_Size));
        expandedOperators.push_back(op);
    }
}

void PancakeOptimized::ExpandCrossSegment(uint64_t index, int opBits, std::vector<uint64_t>& expandedIndexes, std::vector<int>& expandedOperators) {
    bool invert = m_InverseIndex && m_Size > 12;
    //std::cerr << "INVERT: " << m_InverseIndex<< "; s: " << m_Size << std::endl;
    int arr[16];
    OptPermutationUnrank(index, arr, m_Size);

    int newarr[16];
    for (int op = 11; op < m_Size - 1; op++) {
        if (HasOp(opBits, op)) continue;
        Move(arr, newarr, m_Size, op + 2);
        if (invert) MoveInternal(newarr, m_Size, 12);
        expandedIndexes.push_back(OptPermutationRank(newarr, m_Size));
        expandedOperators.push_back(op);
    }
}

void PancakeOptimized::Expand(
    std::vector<uint64_t>& indexes,
    std::vector<int>& usedOperatorBits,
    std::vector<uint64_t>& expandedIndexes,
    std::vector<int>& expandedOperators,
    ExpandHint hint)
{
    SetupOutputBuffers(expandedIndexes, expandedOperators);

    if (hint.SegmentBits == 29) {
        if (hint.CrossSegment) {
            for (int i = 0; i < indexes.size(); i++) {
                ExpandCrossSegment(indexes[i], usedOperatorBits[i], expandedIndexes, expandedOperators);
            }
        }
        else {
            for (int i = 0; i < indexes.size(); i++) {
                ExpandInSegment(indexes[i], usedOperatorBits[i], expandedIndexes, expandedOperators);
            }
        }
    }
    else {
        for (int i = 0; i < indexes.size(); i++) {
            Expand(indexes[i], usedOperatorBits[i], expandedIndexes, expandedOperators);
        }
    }
}
