#include "MTCollector.h"
#include "Util.h"

#include <chrono>
#include <intrin.h>
#include <immintrin.h>

template<int width, int height>
std::atomic<uint64_t> MTCollector<width, height>::m_NanosHorizontalMoves(0);

template<int width, int height>
std::atomic<uint64_t> MTCollector<width, height>::m_NanosVerticalMoves(0);

template<int width, int height>
std::atomic<uint64_t> MTCollector<width, height>::m_NanosSaveSegment(0);

template<int width, int height>
std::atomic<uint64_t> MTCollector<width, height>::m_NanosExclude(0);

template<int width, int height>
std::atomic<uint64_t> MTCollector<width, height>::m_NanosSameSegmentVerticalMoves(0);

template<int width, int height>
MTCollector<width, height>::MTCollector(SegmentedFile& fileVert, SegmentedFile& fileHoriz, SegmentedFile& expanded)
    : m_FileVert(fileVert)
    , m_FileHoriz(fileHoriz)
    , m_Expanded(expanded)
    , m_FrontierWriterVert(fileVert)
    , m_FrontierWriterHoriz(fileHoriz)
    , m_VerticalMovesCollector(expanded, m_VerticalMoves)
    , m_HorizontalMoves{ 0 }
{
    Puzzle<width, height> puzzle;
    constexpr size_t size = puzzle.MaxIndexesPerSegment();
    m_BoundsVert.resize((size + 63) / 64, 0);
    m_BoundsHoriz.resize((size + 63) / 64, 0);
    m_BoundsExclude.resize((size + 63) / 64, 0);
    m_BoundsIndex.resize((m_BoundsVert.size() + VALS_PER_BOUND_INDEX - 1) / VALS_PER_BOUND_INDEX, 0);

    for (int blank = 0; blank < 64; blank++) {
        uint64_t horiz = 0;
        int index = blank;
        while (puzzle.CanMoveLeft(index)) {
            index = puzzle.MoveLeft(index);
            horiz |= (1ui64 << index);
        }
        index = blank;
        while (puzzle.CanMoveRight(index)) {
            index = puzzle.MoveRight(index);
            horiz |= (1ui64 << index);
        }
        m_HorizontalMoves[blank] = horiz;
    }

    for (int blank = 0; blank < puzzle.size; blank++) {
        m_VertChangesSegment[blank] = puzzle.MultiTileHasCrossSegment(blank);
    }
}

template<int width, int height>
void MTCollector<width, height>::SetSegment(uint32_t segment) {
    m_FrontierWriterVert.SetSegment(segment);
    m_FrontierWriterHoriz.SetSegment(segment);
    m_VerticalMoves.SetSegment(segment);
}

template<int width, int height>
void MTCollector<width, height>::AddHorizontalMoves(uint32_t* indexes, size_t count) {
    Timer timer;
    for (size_t i = 0; i < count; i++) {
        uint32_t index = indexes[i];
        uint32_t pos = index / 64;
        m_BoundsHoriz[pos] |= m_HorizontalMoves[index & 63];
        m_BoundsIndex[pos / VALS_PER_BOUND_INDEX] = 1;
    }
    m_NanosHorizontalMoves += timer.Elapsed();
}

template<int width, int height>
void MTCollector<width, height>::AddSameSegmentVerticalMoves(uint32_t* indexes, size_t count) {
    Timer timer;
    for (size_t i = 0; i < count; i++) {
        uint32_t index = indexes[i];
        if (!Puzzle<width, height>::UpChangesSegment(index & 16)) {
            if (m_VerticalMoves.AddSameSegment(index)) {
                FlushSameSegmentVerticalMoves();
            }
        }
    }
    m_NanosSameSegmentVerticalMoves += timer.Elapsed();
}

template<int width, int height>
void MTCollector<width, height>::FlushSameSegmentVerticalMoves() {
    auto size = m_VerticalMoves.VertSameSegment();
    auto* buf = m_VerticalMoves.GetBuffer();
    for (size_t i = 0; i < size; i++) {
        if (buf[i] != uint32_t(-1)) {
            AddNoVert(buf[i]);
        }
    }
    m_VerticalMoves.Clear();
}

template<int width, int height>
void MTCollector<width, height>::FlushSameSegmentAllMoves() {
    Timer timer;
    FlushSameSegmentVerticalMoves();
    m_NanosSameSegmentVerticalMoves += timer.Elapsed();
}

template<int width, int height>
void MTCollector<width, height>::AddCrossSegmentVerticalMoves(uint32_t* indexes, size_t count) {
    Timer timer;
    for (size_t i = 0; i < count; i++) {
        AddNoVert(indexes[i]);
        int blank = indexes[i] & 15;
        if (Puzzle<width, height>::CanMoveDown(blank)) {
            if (m_VerticalMoves.AddSameSegment(indexes[i])) {
                FlushSameSegmentVerticalMoves();
            }
        }
    }
    m_NanosVerticalMoves += timer.Elapsed();
}

template<int width, int height>
void MTCollector<width, height>::AddExclude(uint32_t* indexes, size_t count) {
    Timer timer;
    for (size_t i = 0; i < count; i++) {
        AddExclude(indexes[i]);
    }
    m_NanosExclude += timer.Elapsed();
}

template<int width, int height>
size_t MTCollector<width, height>::SaveSegment() {
    FlushSameSegmentAllMoves();
    int segment = m_FrontierWriterHoriz.GetSegment();
    m_VerticalMovesCollector.SetSegment(segment);

    Timer timer;

    size_t result = 0;

    for (size_t s = 0; s < m_BoundsIndex.size(); s++) {
        if (m_BoundsIndex[s] == 0) continue;
        m_BoundsIndex[s] = 0;
        size_t start = s * VALS_PER_BOUND_INDEX;
        size_t finish = std::min(start + VALS_PER_BOUND_INDEX, m_BoundsHoriz.size());
        for (size_t i = start; i < finish; i++) {
            uint64_t valHoriz = m_BoundsHoriz[i];
            uint64_t valVert = m_BoundsVert[i];
            uint64_t valExclude = m_BoundsExclude[i];
            if ((valHoriz | valVert | valExclude) == 0) continue;
            m_BoundsHoriz[i] = 0;
            m_BoundsVert[i] = 0;
            m_BoundsExclude[i] = 0;
            uint64_t val = (valHoriz | valVert) & ~valExclude;
            if (val == 0) continue;
            uint32_t i_base = i * 64;
            do {
                unsigned long bitIndex;
                _BitScanForward64(&bitIndex, val);
                //val &= ~(1ui64 << bitIndex);
                val = _blsr_u64(val);
                //bool bHoriz = (valHoriz >> bitIndex) & 1;
                //bool bVert = (valVert >> bitIndex) & 1;
                bool bHoriz = valHoriz & (1ui64 << bitIndex);
                bool bVert = valVert & (1ui64 << bitIndex);

                uint32_t index = i_base | bitIndex;
                result++;
                if (!bHoriz) {
                    m_FrontierWriterHoriz.Add(index);
                }
                if (!bVert) {
                    m_FrontierWriterVert.Add(index);
                    if (m_VertChangesSegment[index & 15]) {
                        m_VerticalMovesCollector.Add(index);
                    }
                }
            } while (val != 0);
        }
    }

    m_FrontierWriterHoriz.FinishSegment();
    m_FrontierWriterVert.FinishSegment();
    m_VerticalMovesCollector.Close();

    m_NanosSaveSegment += timer.Elapsed();

    return result;
}

template<int width, int height>
void MTCollector<width, height>::PrintStats() {
    std::cerr
        << "MTCollector: save segment=" << WithTime(m_NanosSaveSegment)
        << "; horiz moves=" << WithTime(m_NanosHorizontalMoves)
        << "; vert moves=" << WithTime(m_NanosVerticalMoves)
        << "; same seg moves=" << WithTime(m_NanosSameSegmentVerticalMoves)
        << "; exclude last =" << WithTime(m_NanosExclude)
        << std::endl;
}

template class MTCollector<2, 2>;
template class MTCollector<3, 2>;
template class MTCollector<4, 2>;
template class MTCollector<5, 2>;
template class MTCollector<6, 2>;
template class MTCollector<7, 2>;
template class MTCollector<8, 2>;

template class MTCollector<3, 3>;
template class MTCollector<4, 3>;
template class MTCollector<5, 3>;

template class MTCollector<4, 4>;
