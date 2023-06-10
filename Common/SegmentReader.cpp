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

void SegmentReader::Delete(int segment) {
    m_Store.Delete(segment);
}

Buffer<uint32_t>& SegmentReader::Read() {
    m_Store.Read(m_Segment, m_Buffer);
    return m_Buffer;
}

OpBitsReader::OpBitsReader(Store& store)
    : m_Store(store)
    , m_Buffer(BUFSIZE)
{}

void OpBitsReader::SetSegment(int segment) {
    m_Segment = segment;
    m_Store.Rewind(segment);
}

void OpBitsReader::Delete(int segment) {
    m_Store.Delete(segment);
}

Buffer<uint8_t>& OpBitsReader::Read() {
    m_Store.Read(m_Segment, m_Buffer);
    return m_Buffer;
}


CrossSegmentReader::CrossSegmentReader(StoreSet& storeSet)
    : m_StoreSet(storeSet)
    , m_Buffer(BUFSIZE)
{}

void CrossSegmentReader::SetSegment(int segment) {
    m_Segment = segment;
    m_StoreSet.Rewind(segment);
}

Buffer<uint32_t>& CrossSegmentReader::Read(int op) {
    m_StoreSet.Stores[op].Read(m_Segment, m_Buffer);
    return m_Buffer;
}

void CrossSegmentReader::Delete(int segment) {
    m_StoreSet.Delete(segment);
}

FrontierReader::FrontierReader(Store& store)
    : m_Store(store)
    , m_IndexBuffer(BUFSIZE)
    , m_OpsBuffer(BUFSIZE)
{}

void FrontierReader::SetSegment(int segment) {
    m_Segment = segment;
    m_Store.Rewind(segment);
}

void FrontierReader::Delete(int segment) {
    m_Store.Delete(segment);
}

FrontierReader::FrontierBuffer FrontierReader::Read() {
    m_Store.Read(m_Segment, m_IndexBuffer);
    m_Store.Read(m_Segment, m_OpsBuffer);
    ensure(m_IndexBuffer.Size() == m_OpsBuffer.Size());
    return { m_IndexBuffer, m_OpsBuffer };
}

