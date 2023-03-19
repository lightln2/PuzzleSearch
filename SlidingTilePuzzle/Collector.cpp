#include "Collector.h"
#include "Util.h"

#include <chrono>
#include <immintrin.h>

template<int width, int height>
std::atomic<uint64_t> Collector<width, height>::m_NanosHorizontalMoves = 0;

template<int width, int height>
std::atomic<uint64_t> Collector<width, height>::m_NanosVerticalMoves = 0;

template<int width, int height>
std::atomic<uint64_t> Collector<width, height>::m_NanosSaveSegment = 0;

template<int width, int height>
Collector<width, height>::Collector(SegmentedFile& file)
    : m_File(file)
    , m_FrontierWriter(file)
    , m_DefaultBounds{ 0 }
    , m_HorizontalMoves{ 0 }
{
    Puzzle<width, height> puzzle;
    constexpr size_t size = puzzle.MaxIndexesPerSegment();
    m_Bounds.resize((size + 15) / 16, 0);
    m_BoundsIndex.resize((m_Bounds.size() + VALS_PER_BOUND_INDEX - 1) / VALS_PER_BOUND_INDEX, 0);

    for (int blank = 0; blank < puzzle.size; blank++) {
        m_DefaultBounds[blank] = puzzle.GetBounds(blank);
    }

    for (int blank = 0; blank < puzzle.size; blank++) {
        for (int bound = 0; bound < 16; bound++) {
            uint64_t horiz = 0;
            if (!(bound & puzzle.B_LEFT) && puzzle.CanMoveLeft(blank)) {
                int movedIndex = puzzle.MoveLeft(blank);
                uint64_t movedBound = puzzle.B_RIGHT | m_DefaultBounds[movedIndex];
                horiz |= (movedBound << (movedIndex * 4));
            }
            if (!(bound & puzzle.B_RIGHT) && puzzle.CanMoveRight(blank)) {
                int movedIndex = puzzle.MoveRight(blank);
                uint64_t movedBound = puzzle.B_LEFT | m_DefaultBounds[movedIndex];
                horiz |= (movedBound << (movedIndex * 4));
            }
            m_HorizontalMoves[(blank * 16) | bound] = horiz;
        }
    }
}

template<int width, int height>
void Collector<width, height>::SetSegment(uint32_t segment) {
    m_FrontierWriter.SetSegment(segment);
}

template<int width, int height>
void Collector<width, height>::AddHorizontalMoves(uint32_t* indexes, uint8_t* bounds, size_t count) {
    Timer timer;
    for (size_t i = 0; i < count; i++) {
        uint32_t index = indexes[i];
        uint8_t bound = bounds[i];
        uint32_t blank = index & 15;
        uint32_t pos = index / 16;
        m_Bounds[pos] |= m_HorizontalMoves[(blank * 16) | bound];
        m_BoundsIndex[pos / VALS_PER_BOUND_INDEX] = 1;
    }
    m_NanosHorizontalMoves += timer.Elapsed();
}

template<int width, int height>
void Collector<width, height>::AddUpMoves(uint32_t* indexes, size_t count) {
    Timer timer;
    for (size_t i = 0; i < count; i++) {
        Add(indexes[i], Puzzle<width, height>::B_DOWN);
    }
    m_NanosVerticalMoves += timer.Elapsed();
}

template<int width, int height>
void Collector<width, height>::AddDownMoves(uint32_t* indexes, size_t count) {
    Timer timer;
    for (size_t i = 0; i < count; i++) {
        Add(indexes[i], Puzzle<width, height>::B_UP);
    }
    m_NanosVerticalMoves += timer.Elapsed();
}

template<int width, int height>
void Collector<width, height>::Add(uint32_t index, uint8_t bounds) {
    m_Bounds[index / 16] |= (uint64_t(bounds) << ((index & 15) * 4));
    m_BoundsIndex[index / VALS_PER_BOUND_INDEX / 16] = 1;
}

template<int width, int height>
size_t Collector<width, height>::SaveSegment() {
    Timer timer;

    //__m256i ones = _mm256_set1_epi8(-1);

    size_t result = 0;

    for (size_t s = 0; s < m_BoundsIndex.size(); s++) {
        if (m_BoundsIndex[s] == 0) continue;
        m_BoundsIndex[s] = 0;
        size_t start = s * VALS_PER_BOUND_INDEX;
        size_t finish = std::min(start + VALS_PER_BOUND_INDEX, m_Bounds.size());
        for (size_t i = start; i < finish; i++) {
            /*if (i + 4 <= finish) {
                __m256i val = *(__m256i*) & m_Bounds[i];
                if (_mm256_testz_si256(val, ones)) {
                    i += 3;
                    continue;
                }
            }*/
            uint64_t val = m_Bounds[i];
            if (val == 0) continue;
            m_Bounds[i] = 0;
            unsigned long bitIndex = 0;

            do {
                _BitScanForward64(&bitIndex, val);
                int blank = bitIndex / 4;
                uint8_t bound = (val >> (blank * 4)) & 15;
                bound |= m_DefaultBounds[blank];
                result++;
                if (bound != 15) {
                    m_FrontierWriter.Add((uint32_t)(i * 16 + blank), bound);
                }

                val &= ~(15ui64 << (blank * 4));
            } while (val != 0);
        }
    }
    
    m_FrontierWriter.FinishSegment();

    m_NanosSaveSegment += timer.Elapsed();

    return result;
}

template<int width, int height>
void Collector<width, height>::PrintStats() {
    std::cerr 
        << "Collector: save segment=" << WithTime(m_NanosSaveSegment) 
        << "; horiz moves=" << WithTime(m_NanosHorizontalMoves)
        << "; vert moves=" << WithTime(m_NanosVerticalMoves)
        << std::endl;
}

template class Collector<2, 2>;
template class Collector<3, 2>;
template class Collector<4, 2>;
template class Collector<5, 2>;
template class Collector<6, 2>;
template class Collector<7, 2>;
template class Collector<8, 2>;

template class Collector<3, 3>;
template class Collector<4, 3>;
template class Collector<5, 3>;

template class Collector<4, 4>;
