#pragma once

#include "FrontierFile.h"
#include "Puzzle.h"

#include <atomic>

template<int width, int height>
class Collector {
private:
    static constexpr size_t VALS_PER_BOUND_INDEX = 4096;
public:
    Collector(SegmentedFile& file);

    void SetSegment(uint32_t segment);

    void AddHorizontalMoves(uint32_t* indexes, uint8_t* bounds, size_t count);

    void Add(uint32_t index, uint8_t bounds);

    size_t SaveSegment();

    void PrintStats();

private:
    SegmentedFile& m_File;
    FrontierFileWriter m_FrontierWriter;
    Buffer<uint8_t> m_Bounds;
    Buffer<uint8_t> m_BoundIndex;

private:
    static std::atomic<uint64_t> m_NanosSaveSegment;
    static std::atomic <uint64_t> m_NanosHorizontalMoves;
    //static uint64_t m_NanosSaveSegment;
};

