#include "SegmentReader.h"
#include "Util.h"

SegmentReader::SegmentReader(Store& store) 
    : m_Store(store)
{
    m_Buffer.reserve(BUFSIZE);
}

void SegmentReader::SetSegment(int segment) {
    m_Segment = segment;
    m_Store.Rewind(segment);
}

std::vector<uint32_t>& SegmentReader::Read() {
    m_Buffer.clear();
    m_Store.Read(m_Segment, m_Buffer);
    return m_Buffer;
}
