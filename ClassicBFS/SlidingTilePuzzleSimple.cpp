#include "SlidingTilePuzzleSimple.h"
#include "PermutationMap.h"
#include "../Common/Util.h"

#include <sstream>

namespace {
    const int OP_UP = 0, OP_LEFT = 1, OP_RIGHT = 2, OP_DOWN = 3;

    bool HasOp(int op, int dir) {
        return op & (1 << dir);
    }

    void Move(int* arr, int* newarr, int blank, int newblank) {
        memcpy(newarr, arr, 16 * sizeof(int));
        newarr[blank] = arr[newblank];
        newarr[newblank] = arr[blank];
    }

}

SlidingTilePuzzleSimple::SlidingTilePuzzleSimple(int width, int height)
    : m_Width(width)
    , m_Height(height)
    , m_Size(width* height)
{ }

uint64_t SlidingTilePuzzleSimple::IndexesCount() const {
    uint64_t result = 1;
    for (int i = 1; i <= m_Size; i++) result *= i;
    return result;
}

std::string SlidingTilePuzzleSimple::ToString(uint64_t index) {
    std::ostringstream stream;
    int arr[16];
    PermutationUnrank(index, arr, m_Size);
    for (int i = 0; i < m_Size; i++) {
        if (i > 0) stream << ' ';
        stream << arr[i];
    }
    return stream.str();
}

uint64_t SlidingTilePuzzleSimple::Parse(std::string state) {
    std::istringstream stream(state);
    int arr[16];
    for (int i = 0; i < m_Size; i++) {
        stream >> arr[i];
    }
    return PermutationRank(arr, m_Size);
}

void SlidingTilePuzzleSimple::Expand(uint64_t index, int op, uint64_t* children, int* operators) {
    int arr[16];
    int blank = -1;
    PermutationUnrank(index, arr, m_Size);
    for (int i = 0; i < m_Size; i++) {
        if (arr[i] == 0) {
            blank = i;
            break;
        }
    }
    ensure(blank >= 0);

    int newarr[16];

    if (blank >= m_Width && !HasOp(op, OP_UP)) {
        Move(arr, newarr, blank, blank - m_Width);
        children[0] = PermutationRank(newarr, m_Size);
        operators[0] = OP_DOWN;
    }

    if (blank < m_Size - m_Width && !HasOp(op, OP_DOWN)) {
        Move(arr, newarr, blank, blank + m_Width);
        children[1] = PermutationRank(newarr, m_Size);
        operators[1] = OP_UP;
    }

    if (blank % m_Width > 0 && !HasOp(op, OP_LEFT)) {
        Move(arr, newarr, blank, blank - 1);
        children[2] = PermutationRank(newarr, m_Size);
        operators[2] = OP_RIGHT;
    }

    if (blank % m_Width < m_Width - 1 && !HasOp(op, OP_RIGHT)) {
        Move(arr, newarr, blank, blank + 1);
        children[3] = PermutationRank(newarr, m_Size);
        operators[3] = OP_LEFT;
    }

}

void SlidingTilePuzzleSimple::Expand(
    std::vector<uint64_t>& indexes,
    std::vector<int>& usedOperatorBits,
    std::vector<uint64_t>& expandedIndexes,
    std::vector<int>& expandedOperators)
{
    if (expandedIndexes.capacity() < MAX_INDEXES_BUFFER * 4) {
        expandedIndexes.reserve(MAX_INDEXES_BUFFER * 4);
    }
    if (expandedOperators.capacity() < MAX_INDEXES_BUFFER * 4) {
        expandedOperators.reserve(MAX_INDEXES_BUFFER * 4);
    }
    expandedIndexes.clear();
    expandedOperators.clear();
    expandedIndexes.resize(indexes.size() * 4, INVALID_INDEX);
    expandedOperators.resize(indexes.size() * 4, -1);

    for (int i = 0; i < indexes.size(); i++) {
        Expand(indexes[i], usedOperatorBits[i], &expandedIndexes[i * 4], &expandedOperators[i * 4]);
    }
}
