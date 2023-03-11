#include "Collector.h"
#include "Util.h"

#include <chrono>

//std::atomic<uint64_t> Collector::m_NanosSaveSegment = 0;
uint64_t Collector::m_NanosSaveSegment = 0;

Collector::Collector(size_t count, SegmentedFile& file) 
    : m_File(file)
    , m_FrontierWriter(file)
    , m_Bounds(count)
{
    m_Bounds.SetSize(count);
    memset(m_Bounds.Buf(), 0, m_Bounds.Size());
}

void Collector::SetSegment(uint32_t segment) {
    m_FrontierWriter.SetSegment(segment);
}

void Collector::Add(uint32_t index, uint8_t bounds) {
    m_Bounds[index] |= bounds;
}

size_t Collector::SaveSegment() {
    auto timerStart = std::chrono::high_resolution_clock::now();

    size_t result = 0;

    uint64_t* buffer = (uint64_t*)m_Bounds.Buf();
    size_t buf_len = m_Bounds.Size() / 8;

    for (size_t i = 0; i < buf_len; i++) {
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
    /*
    for (size_t i = 0; i < m_Bounds.Size(); i++) {
        auto bound = m_Bounds[i];
        if (bound == 0) continue;
        result++;
        if (bound != 15) {
            m_FrontierWriter.Add((uint32_t)i, bound);
        }
        m_Bounds[i] = 0;
    }
    */

    m_FrontierWriter.FinishSegment();

    auto timerEnd = std::chrono::high_resolution_clock::now();
    m_NanosSaveSegment += (timerEnd - timerStart).count();

    return result;
}

void Collector::PrintStats() {
    std::cerr << "Collector: save segment=" << WithTime(m_NanosSaveSegment) << std::endl;
}