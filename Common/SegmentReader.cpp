#include "SegmentReader.h"
#include "Util.h"

SegmentReader::SegmentReader(Store& store)
    : m_Store(store)
    , m_Buffer(BUFSIZE)
{}

void SegmentReader::SetSegment(int segment) {
    m_Segment = segment;
    m_Store.Rewind(segment);
}

Buffer<uint32_t>& SegmentReader::Read() {
    m_Store.Read(m_Segment, m_Buffer);
    return m_Buffer;
}
