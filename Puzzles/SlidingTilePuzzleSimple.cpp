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

std::string SlidingTilePuzzleSimple::Name() const {
    std::ostringstream stream;
    stream
        << "Sliding-Tile simple: " << m_Width << " x " << m_Height;
    return stream.str();
}

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

void SlidingTilePuzzleSimple::Expand(uint64_t index, int op, std::vector<uint64_t>& expandedIndexes, std::vector<int>& expandedOperators) {
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
        expandedIndexes.push_back(PermutationRank(newarr, m_Size));
        expandedOperators.push_back(OP_DOWN);
    }

    if (blank < m_Size - m_Width && !HasOp(op, OP_DOWN)) {
        Move(arr, newarr, blank, blank + m_Width);
        expandedIndexes.push_back(PermutationRank(newarr, m_Size));
        expandedOperators.push_back(OP_UP);
    }

    if (blank % m_Width > 0 && !HasOp(op, OP_LEFT)) {
        Move(arr, newarr, blank, blank - 1);
        expandedIndexes.push_back(PermutationRank(newarr, m_Size));
        expandedOperators.push_back(OP_RIGHT);
    }

    if (blank % m_Width < m_Width - 1 && !HasOp(op, OP_RIGHT)) {
        Move(arr, newarr, blank, blank + 1);
        expandedIndexes.push_back(PermutationRank(newarr, m_Size));
        expandedOperators.push_back(OP_LEFT);
    }

}

void SlidingTilePuzzleSimple::Expand(
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
