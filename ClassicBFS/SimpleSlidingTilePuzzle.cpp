#include "SimplePuzzle.h"
#include "PermutationMap.h"
#include "../SlidingTilePuzzle/Util.h"

#include <sstream>

static const int OP_UP = 0, OP_LEFT = 1, OP_RIGHT = 2, OP_DOWN = 3;

static bool HasOp(int op, int dir) {
    return op & (1 << dir);
}

SimpleSlidingPuzzle::SimpleSlidingPuzzle(int width, int height)
    : m_Width(width)
    , m_Height(height)
    , m_Size(width * height)
{ }

uint64_t SimpleSlidingPuzzle::MaxIndexes() {
    uint64_t result = 1;
    for (int i = 1; i <= m_Size; i++) result *= i;
    return result;
}

std::string SimpleSlidingPuzzle::ToString(uint64_t index) {
    std::ostringstream stream;
    int arr[16];
    PermutationUnrank(index, arr, m_Size);
    for (int i = 0; i < m_Size; i++) {
        if (i > 0) stream << ' ';
        stream << arr[i];
    }
    return stream.str();
}

uint64_t SimpleSlidingPuzzle::Parse(std::string state) {
    std::istringstream stream(state);
    int arr[16];
    for (int i = 0; i < m_Size; i++) {
        stream >> arr[i];
    }
    return PermutationRank(arr, m_Size);
}

static void Move(int* arr, int* newarr, int blank, int newblank) {
    memcpy(newarr, arr, 16 * sizeof(int));
    newarr[blank] = arr[newblank];
    newarr[newblank] = arr[blank];
}

void SimpleSlidingPuzzle::Expand(uint64_t index, int op, uint64_t* children, int* usedOperatorBits) {
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

    children[0] = children[1] = children[2] = children[3] = INVALID_INDEX;
    usedOperatorBits[0] = usedOperatorBits[1] = usedOperatorBits[2] = usedOperatorBits[3] = -1;

    if (blank >= m_Width && !HasOp(op, OP_UP)) {
        Move(arr, newarr, blank, blank - m_Width);
        children[0] = PermutationRank(newarr, m_Size);
        usedOperatorBits[0] = OP_DOWN;
    }

    if (blank < m_Size - m_Width && !HasOp(op, OP_DOWN)) {
        Move(arr, newarr, blank, blank + m_Width);
        children[1] = PermutationRank(newarr, m_Size);
        usedOperatorBits[1] = OP_UP;
    }

    if (blank % m_Width > 0 && !HasOp(op, OP_LEFT)) {
        Move(arr, newarr, blank, blank - 1);
        children[2] = PermutationRank(newarr, m_Size);
        usedOperatorBits[2] = OP_RIGHT;
    }

    if (blank % m_Width < m_Width - 1 && !HasOp(op, OP_RIGHT)) {
        Move(arr, newarr, blank, blank + 1);
        children[3] = PermutationRank(newarr, m_Size);
        usedOperatorBits[3] = OP_LEFT;
    }

}

void SimpleSlidingPuzzle::Expand(int count, uint64_t* indexes, int* ops, uint64_t* children, int* usedOperatorBits) {
    for (int i = 0; i < count; i++) {
        Expand(indexes[i], ops[i], children + i * count, usedOperatorBits + i * count);
    }
}
