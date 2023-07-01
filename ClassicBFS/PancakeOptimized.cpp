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

    void Move(int* arr, int* newarr, int size, int count, bool invIdx) {
        memcpy(newarr, arr, 16 * sizeof(int));
        if (invIdx && size > 12) {
            MoveInternal(newarr, size, 12);
            MoveInternal(newarr, size, count);
            MoveInternal(newarr, size, 12);
        }
        else {
            MoveInternal(newarr, size, count);
        }
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
            //index -= (segment << 29);
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
    if (m_InverseIndex && m_Size > 12) MoveInternal(arr, m_Size, 12);
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
    if (m_InverseIndex && m_Size > 12) MoveInternal(arr, m_Size, 12);
    return OptPermutationRank(arr, m_Size);
}

void PancakeOptimized::Expand(uint64_t index, int opBits, std::vector<uint64_t>& expandedIndexes, std::vector<int>& expandedOperators) {
    int arr[16];
    OptPermutationUnrank(index, arr, m_Size);

    int newarr[16];
    for (int op = 0; op < m_Size - 1; op++) {
        if (HasOp(opBits, op)) continue;
        Move(arr, newarr, m_Size, op + 2, m_InverseIndex);
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
        Move(arr, newarr, m_Size, op + 2, m_InverseIndex);
        expandedIndexes.push_back(OptPermutationRank(newarr, m_Size));
        expandedOperators.push_back(op);
    }
}

void PancakeOptimized::ExpandCrossSegment(uint64_t index, int opBits, std::vector<uint64_t>& expandedIndexes, std::vector<int>& expandedOperators) {
    int arr[16];
    OptPermutationUnrank(index, arr, m_Size);

    int newarr[16];
    for (int op = 11; op < m_Size - 1; op++) {
        if (HasOp(opBits, op)) continue;
        Move(arr, newarr, m_Size, op + 2, m_InverseIndex);
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
