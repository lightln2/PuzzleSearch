#pragma once

#include "FrontierFile.h"
#include "Puzzle.h"
#include "GpuSolver.h"

#include <atomic>

template<int width, int height>
class Collector {
private:
    static constexpr size_t VALS_PER_BOUND_INDEX = 4 * 1024;
public:
    Collector(SegmentedFile& file);

    void SetSegment(uint32_t segment);

    void AddHorizontalMoves(uint32_t* indexes, uint8_t* bounds, size_t count);
    // TODO: tests
    void AddSameSegmentVerticalMoves(uint32_t* indexes, uint8_t* bounds, size_t count);
    void AddUpMoves(uint32_t* indexes, size_t count);
    void AddDownMoves(uint32_t* indexes, size_t count);

    void Add(uint32_t index, uint8_t bounds) {
        m_Bounds[index / 16] |= (uint64_t(bounds) << ((index & 15) * 4));
        m_BoundsIndex[index / VALS_PER_BOUND_INDEX / 16] = 1;
    }

    size_t SaveSegment();

    static void PrintStats();

private:
    void FlushSameSegmentUpMoves();
    void FlushSameSegmentDownMoves();

private:
    SegmentedFile& m_File;
    FrontierFileWriter m_FrontierWriter;
    std::vector<uint64_t> m_Bounds;
    std::vector<uint8_t> m_BoundsIndex;
    GpuSolver<width, height> m_GpuSolver;
    HostBuffer m_UpBuffer;
    size_t m_UpBufferPosition = 0;
    HostBuffer m_DownBuffer;
    size_t m_DownBufferPosition = 0;
    HostBuffer m_UpDownSegments;

private:
    static std::atomic<uint64_t> m_NanosSaveSegment;
    static std::atomic <uint64_t> m_NanosHorizontalMoves;
    static std::atomic <uint64_t> m_NanosVerticalMoves;
    static std::atomic <uint64_t> m_NanosSameSegmentVerticalMoves;

    uint8_t m_DefaultBounds[16];
    uint64_t m_HorizontalMoves[16 * 16];
};

