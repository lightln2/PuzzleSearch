#pragma once

#include "../Common/Puzzle.h"

#include <mutex>

using CuStream = void*;

class PuzzleGpu : public Puzzle {
private:
    struct Exec {
        CuStream stream;
        uint64_t* gpuSrc;
        uint64_t* gpuDst;
        Exec();
        ~Exec();
    };
public:
    virtual void Expand(
        std::vector<uint64_t>& indexes,
        std::vector<int>& usedOperatorBits,
        std::vector<uint64_t>& expandedIndexes,
        std::vector<int>& expandedOperators);

protected:
    virtual void ExpandGpu(
        uint64_t* gpuIndexes,
        uint64_t* gpuExpanded,
        uint64_t count,
        CuStream stream) = 0;

private:
    std::mutex m_Mutex;
    std::vector<std::unique_ptr<Exec>> m_Streams;
    std::vector<Exec*> m_FreeStreams;
};
