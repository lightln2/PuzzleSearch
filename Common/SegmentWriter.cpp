#include "SegmentWriter.h"
#include "Util.h"

SegmentWriter::SegmentWriter(Store& store) 
    : m_Store(store)
{
    m_Buffer.reserve(BUFSIZE);
}

void SegmentWriter::SetSegment(int segment) {
    ensure(m_Buffer.empty());
    m_Segment = segment;
}

void SegmentWriter::Add(uint32_t value) {
    m_Buffer.push_back(value);
    if (m_Buffer.size() == BUFSIZE) {
        Flush();
    }
}

void SegmentWriter::Flush() {
    m_Store.Write(m_Segment, m_Buffer);
    m_Buffer.clear();
}
