#include "SegmentWriter.h"
#include "Util.h"

SegmentWriter::SegmentWriter(Store& store) 
    : m_Store(store)
    , m_Buffer(BUFSIZE)
{}

void SegmentWriter::SetSegment(int segment) {
    ensure(m_Buffer.IsEmpty());
    m_Segment = segment;
}

void SegmentWriter::Add(uint32_t value) {
    m_Buffer.Add(value);
    if (m_Buffer.IsFull()) Flush();
}

void SegmentWriter::Flush() {
    m_Store.Write(m_Segment, m_Buffer);
    m_Buffer.Clear();
}

OpBitsWriter::OpBitsWriter(Store& store)
    : m_Store(store)
    , m_Buffer(BUFSIZE)
{}

void OpBitsWriter::SetSegment(int segment) {
    ensure(m_Buffer.IsEmpty());
    m_Segment = segment;
}

void OpBitsWriter::Add(uint8_t value) {
    m_Buffer.Add(value);
    if (m_Buffer.IsFull()) Flush();
}

void OpBitsWriter::Flush() {
    m_Store.Write(m_Segment, m_Buffer);
    m_Buffer.Clear();
}

FrontierWriter::FrontierWriter(Store& store)
    : m_Store(store)
    , m_IndexBuffer(BUFSIZE)
    , m_OpsBuffer(BUFSIZE)
{}

void FrontierWriter::SetSegment(int segment) {
    ensure(m_IndexBuffer.IsEmpty());
    m_Segment = segment;
}

void FrontierWriter::Add(uint32_t value, int opBits) {
    m_IndexBuffer.Add(value);
    m_OpsBuffer.Add(uint8_t(opBits));
    if (m_IndexBuffer.IsFull()) Flush();
}

void FrontierWriter::Flush() {
    m_Store.Write(m_Segment, m_IndexBuffer);
    m_Store.Write(m_Segment, m_OpsBuffer);
    m_IndexBuffer.Clear();
    m_OpsBuffer.Clear();
}
