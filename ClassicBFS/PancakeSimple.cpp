#include "PancakeSimple.h"
#include "PermutationMap.h"
#include "../Common/Util.h"

#include <sstream>

namespace {
    bool HasOp(int op, int dir) {
        return op & (1 << dir);
    }

    void Move(int* arr, int* newarr, int size, int count) {
        memcpy(newarr, arr, 16 * sizeof(int));
        int start = size - count, end = size - 1;
        while (start < end) {
            std::swap(newarr[start], newarr[end]);
            start++;
            end--;
        }
    }

}

PancakeSimple::PancakeSimple(int size)
    : m_Size(size)
{
    ensure(size > 1 && size <= 16);
}

std::string PancakeSimple::Name() const {
    std::ostringstream stream;
    stream << "Pancake simple, size: " << m_Size;
    return stream.str();
}

uint64_t PancakeSimple::IndexesCount() const {
    uint64_t result = 1;
    for (int i = 1; i <= m_Size; i++) result *= i;
    return result;
}

std::string PancakeSimple::ToString(uint64_t index) {
    std::ostringstream stream;
    int arr[16];
    PermutationUnrank(index, arr, m_Size);
    for (int i = 0; i < m_Size; i++) {
        if (i > 0) stream << ' ';
        stream << arr[i];
    }
    return stream.str();
}

uint64_t PancakeSimple::Parse(std::string state) {
    std::istringstream stream(state);
    int arr[16];
    for (int i = 0; i < m_Size; i++) {
        stream >> arr[i];
    }
    return PermutationRank(arr, m_Size);
}

void PancakeSimple::Expand(uint64_t index, int opBits, std::vector<uint64_t>& expandedIndexes, std::vector<int>& expandedOperators) {
    int arr[16];
    PermutationUnrank(index, arr, m_Size);

    int newarr[16];
    for (int op = 0; op < m_Size - 1; op++) {
        if (HasOp(opBits, op)) continue;
        Move(arr, newarr, m_Size, op + 2);
        expandedIndexes.push_back(PermutationRank(newarr, m_Size));
        expandedOperators.push_back(op);
    }
}

void PancakeSimple::Expand(
    std::vector<uint64_t>& indexes,
    std::vector<int>& usedOperatorBits,
    std::vector<uint64_t>& expandedIndexes,
    std::vector<int>& expandedOperators,
    ExpandHint hint)
{
    SetupOutputBuffers(expandedIndexes, expandedOperators);

    for (int i = 0; i < indexes.size(); i++) {
        Expand(indexes[i], usedOperatorBits[i], expandedIndexes, expandedOperators);
    }
}
