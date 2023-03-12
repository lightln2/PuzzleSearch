#include "Collector.h"
#include "Util.h"

#include <chrono>

template<int width, int height>
std::atomic<uint64_t> Collector<width, height>::m_NanosHorizontalMoves = 0;

template<int width, int height>
std::atomic<uint64_t> Collector<width, height>::m_NanosSaveSegment = 0;

template<int width, int height>
Collector<width, height>::Collector(SegmentedFile& file)
    : m_File(file)
    , m_FrontierWriter(file)
    , m_Bounds(Puzzle<width, height>::MaxIndexesPerSegment())
    , m_BoundIndex(m_Bounds.Capacity() / VALS_PER_BOUND_INDEX / 8 + 1)
{
    m_Bounds.SetSize(m_Bounds.Capacity());
    memset(m_Bounds.Buf(), 0, m_Bounds.Size());

    m_BoundIndex.SetSize(m_BoundIndex.Capacity());
    memset(m_BoundIndex.Buf(), 0, m_BoundIndex.Size());

}

template<int width, int height>
void Collector<width, height>::SetSegment(uint32_t segment) {
    m_FrontierWriter.SetSegment(segment);
}

template<int width, int height>
void Collector<width, height>::AddHorizontalMoves(uint32_t* indexes, uint8_t* bounds, size_t count) {
    Timer timer;
    Puzzle<width, height> puzzle;
    for (size_t i = 0; i < count; i++) {
        uint32_t index = indexes[i];
        uint8_t bound = bounds[i];
        if (!(bound & puzzle.B_LEFT)) {
            auto leftMove = puzzle.MoveLeft(index);
            Add(leftMove, puzzle.GetBounds(leftMove) | puzzle.B_RIGHT);
        }
        if (!(bound & puzzle.B_RIGHT)) {
            auto rightMove = puzzle.MoveRight(index);
            Add(rightMove, puzzle.GetBounds(rightMove) | puzzle.B_LEFT);
        }
    }
    m_NanosHorizontalMoves += timer.Elapsed();
}

template<int width, int height>
void Collector<width, height>::Add(uint32_t index, uint8_t bounds) {
    m_Bounds[index] |= bounds;
    m_BoundIndex[index / VALS_PER_BOUND_INDEX / 8] = 1;
}

template<int width, int height>
size_t Collector<width, height>::SaveSegment() {
    Timer timer;

    size_t result = 0;

    uint64_t* buffer = (uint64_t*)m_Bounds.Buf();
    size_t buf_len = m_Bounds.Size() / 8;

    for (size_t s = 0; s < m_BoundIndex.Size(); s++) {
        if (m_BoundIndex[s] == 0) continue;
        m_BoundIndex[s] = 0;
        size_t start = s * VALS_PER_BOUND_INDEX;
        size_t finish = std::min(start + VALS_PER_BOUND_INDEX, buf_len);
        for (size_t i = start; i < finish; i++) {
            auto val = buffer[i];
            if (val == 0) continue;
            unsigned long bitIndex = 0;

            while (_BitScanForward64(&bitIndex, val)) {
                int byteIndex = bitIndex / 8;
                uint8_t bound = (val >> (byteIndex * 8)) & 15;
                result++;
                if (bound != 15) {
                    m_FrontierWriter.Add((uint32_t)(i * 8 + byteIndex), bound);
                }

                val &= ~(15ui64 << (byteIndex * 8));
            }
            buffer[i] = 0;
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
