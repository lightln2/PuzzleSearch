#pragma once

#include "GpuSolver.h"
#include "MTFrontierFile.h"
#include "MTVerticalMoves.h"
#include "MTVerticalMovesCollector.h"
#include "Puzzle.h"

#include <atomic>

template<int width, int height>
class MTCollector {
private:
    static constexpr size_t VALS_PER_BOUND_INDEX = 4 * 1024;
public:
    MTCollector(SegmentedFile& file, SegmentedFile& expandedUp, SegmentedFile& expandedDown);

    void SetSegment(uint32_t segment);

    void AddHorizontalMoves(uint32_t* indexes, size_t count);
    void AddSameSegmentVerticalMoves(uint32_t* indexes, size_t count);
    void AddCrossSegmentVerticalMoves(uint32_t* indexes, size_t count);
    void AddExclude(uint32_t* indexes, size_t count);

    void AddNoVert(uint32_t index) {
        m_BoundsVert[index / 64] |= (1ui64 << (index & 63));
        m_BoundsIndex[index / VALS_PER_BOUND_INDEX / 16] = 1;
    }

    void AddNoHoriz(uint32_t index) {
        m_BoundsHoriz[index / 64] |= (1ui64 << (index & 63));
        m_BoundsIndex[index / VALS_PER_BOUND_INDEX / 16] = 1;
    }

    void AddExclude(uint32_t index) {
        m_BoundsExclude[index / 64] |= (1ui64 << (index & 63));
        m_BoundsIndex[index / VALS_PER_BOUND_INDEX / 16] = 1;
    }

    size_t SaveSegment();

    void CloseAll() {
        m_VerticalMovesCollector.CloseAll();
    }

    static void PrintStats();

private:
    void FlushSameSegmentVerticalMoves();
    void FlushSameSegmentAllMoves();
 
private:
    SegmentedFile& m_FileVert;
    SegmentedFile& m_FileHoriz;
    SegmentedFile& m_Expanded;
    MTFrontierFileWriter m_FrontierWriterVert;
    MTFrontierFileWriter m_FrontierWriterHoriz;
    MTVerticalMoves<width, height> m_VerticalMoves;
    MTVerticalMovesCollector<width, height> m_VerticalMovesCollector;

    std::vector<uint64_t> m_BoundsVert;
    std::vector<uint64_t> m_BoundsHoriz;
    std::vector<uint64_t> m_BoundsExclude;
    std::vector<uint8_t> m_BoundsIndex;

private:
    static std::atomic<uint64_t> m_NanosSaveSegment;
    static std::atomic<uint64_t> m_NanosHorizontalMoves;
    static std::atomic<uint64_t> m_NanosVerticalMoves;
    static std::atomic<uint64_t> m_NanosExclude;
    static std::atomic<uint64_t> m_NanosSameSegmentVerticalMoves;

    uint64_t m_HorizontalMoves[64];
    bool m_VertChangesSegment[16];
};

