#pragma once

#include "PuzzleGpu.h"

#include <string>

struct VPuzzleState {
    int tiles[16];
    VPuzzleState() = default;
    VPuzzleState(int tiles[16]) { memcpy(this->tiles, tiles, sizeof(this->tiles)); }
};

template<int width, int height>
class SlidingTilePuzzleOptimized : public PuzzleGpu {
public:
    static constexpr int size = width * height;
public:
    virtual std::string Name() const;

    virtual int OperatorsCount() const { return 4; }

    virtual uint64_t IndexesCount() const;

    virtual bool HasOddLengthCycles() const { return false; }

    virtual std::string ToString(uint64_t index);

    virtual uint64_t Parse(std::string state);

    virtual void Expand(
        std::vector<uint64_t>& indexes,
        std::vector<int>& usedOperatorBits,
        std::vector<uint64_t>& expandedIndexes,
        std::vector<int>& expandedOperators,
        ExpandHint hint);

private:
    bool CanMoveUp(uint32_t index) { return (index % 16) >= width; }
    static bool CanMoveDown(uint32_t index) { return (index % 16) < size - width; }
    static bool CanMoveLeft(uint32_t index) { return (index % 16) % width > 0; }
    static bool CanMoveRight(uint32_t index) { return (index % 16) % width < width - 1; }

    static uint32_t MoveLeft(uint32_t index) { return index - 1; }
    static uint32_t MoveRight(uint32_t index) { return index + 1; }

protected:
    virtual void ExpandGpu(
        uint64_t* gpuIndexes,
        uint64_t* gpuExpanded,
        uint64_t count,
        CuStream stream);

};
