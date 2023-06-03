#pragma once

#include "GPU.h"
#include "../Common/Puzzle.h"

#include <cstdint>
#include <mutex>
#include <string>

class SlidingTilePuzzleGpu : public Puzzle {
private:
    struct Exec {
        CuStream stream;
        uint64_t* gpuSrc;
        uint64_t* gpuDst;
        Exec();
        ~Exec();
    };
public:
    SlidingTilePuzzleGpu(int width, int height);

    virtual int OperatorsCount() const { return 4; }

    virtual uint64_t IndexesCount() const;

    virtual bool HasOddLengthCycles() const { return false; }

    virtual std::string ToString(uint64_t index);

    virtual uint64_t Parse(std::string state);

    virtual void Expand(
        std::vector<uint64_t>& indexes,
        std::vector<int>& usedOperatorBits,
        std::vector<uint64_t>& expandedIndexes,
        std::vector<int>& expandedOperators);

private:
    int m_Width;
    int m_Height;
    int m_Size;
    std::mutex m_Mutex;
    std::vector<std::unique_ptr<Exec>> m_Streams;
    std::vector<Exec*> m_FreeStreams;
};
