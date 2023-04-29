#pragma once

#include <cstdint>
#include <string>

class SimpleSlidingPuzzle {
public:
    static constexpr uint64_t INVALID_INDEX = uint64_t(-1);

    SimpleSlidingPuzzle(int width, int height);

    int MaxBranching() const { return 4; }
    int OperatorBits() const { return 4; }

    uint64_t MaxIndexes();
    std::string ToString(uint64_t index);
    uint64_t Parse(std::string state);

    void Expand(uint64_t index, int op, uint64_t* children, int* usedOperatorBits);

    void Expand(int count, uint64_t* indexes, int* ops, uint64_t* children, int* usedOperatorBits);

private:
    int m_Width;
    int m_Height;
    int m_Size;
};
