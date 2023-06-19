#pragma once

#include "PuzzleGpu.h"

#include <string>

class SlidingTilePuzzleGpu : public PuzzleGpu {
public:
    SlidingTilePuzzleGpu(int width, int height);

    virtual int OperatorsCount() const { return 4; }

    virtual uint64_t IndexesCount() const;

    virtual bool HasOddLengthCycles() const { return false; }

    virtual std::string ToString(uint64_t index);

    virtual uint64_t Parse(std::string state);

protected:
    virtual void ExpandGpu(
        uint64_t* gpuIndexes,
        uint64_t* gpuExpanded,
        uint64_t count,
        CuStream stream);

private:
    int m_Width;
    int m_Height;
    int m_Size;
};
