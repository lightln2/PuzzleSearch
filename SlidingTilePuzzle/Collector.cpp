#include "Collector.h"

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
    size_t result = 0;
    for (size_t i = 0; i < m_Bounds.Size(); i++) {
        auto bound = m_Bounds[i];
        if (bound == 0) continue;
        result++;
        if (bound != 15) {
            m_FrontierWriter.Add((uint32_t)i, bound);
        }
        m_Bounds[i] = 0;
    }
    m_FrontierWriter.FinishSegment();
    return result;
}