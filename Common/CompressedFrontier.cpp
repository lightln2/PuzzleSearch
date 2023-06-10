#include "CompressedFrontier.h"
#include "Util.h"


CompressedFrontierReader::CompressedFrontierReader(Store& store)
    : m_Store(store)
    , m_InputBuffer(BUFSIZE)
    , m_IndexBuffer(StreamVInt::MAX_INDEXES_COUNT)
    , m_OpsBuffer(StreamVInt::MAX_INDEXES_COUNT)
    , m_InputPos(0)
{}

void CompressedFrontierReader::SetSegment(int segment) {
    m_Segment = segment;
    m_Store.Rewind(segment);
}

void CompressedFrontierReader::Delete(int segment) {
    m_Store.Delete(segment);
}

CompressedFrontierReader::FrontierBuffer CompressedFrontierReader::Read() {
    if (m_InputPos == m_InputBuffer.Size()) {
        m_Store.Read(m_Segment, m_InputBuffer);
        m_InputPos = 0;
    }
    m_IndexBuffer.Clear();
    m_OpsBuffer.Clear();
    m_InputPos = StreamVInt::Decode(m_InputPos, m_InputBuffer, m_IndexBuffer, m_OpsBuffer);
    return { m_IndexBuffer, m_OpsBuffer };
}

CompressedFrontierWriter::CompressedFrontierWriter(Store& store)
    : m_Store(store)
    , m_IndexBuffer(StreamVInt::MAX_INDEXES_COUNT)
    , m_OpsBuffer(StreamVInt::MAX_INDEXES_COUNT)
    , m_CompressedBuffer(StreamVInt::MAX_BUFFER_SIZE)
    , m_OutputBuffer(BUFSIZE)
{}

void CompressedFrontierWriter::SetSegment(int segment) {
    m_Segment = segment;
    ensure(m_IndexBuffer.IsEmpty());
    ensure(m_OpsBuffer.IsEmpty());
    ensure(m_CompressedBuffer.IsEmpty());
}

void CompressedFrontierWriter::Flush() {
    if (!m_IndexBuffer.IsEmpty()) FlushData();
    if (!m_OutputBuffer.IsEmpty()) FlushBuffer();
}

void CompressedFrontierWriter::Add(uint32_t index, int opBits) {
    m_IndexBuffer.Add(index);
    m_OpsBuffer.Add(uint8_t(opBits));
    if (m_IndexBuffer.IsFull()) {
        FlushData();
    }
}

void CompressedFrontierWriter::FlushData() {
    StreamVInt::Encode(m_IndexBuffer, m_OpsBuffer, m_CompressedBuffer);
    m_IndexBuffer.Clear();
    m_OpsBuffer.Clear();
    if (!m_OutputBuffer.CanAppend(m_CompressedBuffer)) FlushBuffer();
    m_OutputBuffer.Append(m_CompressedBuffer);
    m_CompressedBuffer.Clear();
}

void CompressedFrontierWriter::FlushBuffer() {
    m_Store.Write(m_Segment, m_OutputBuffer);
    m_OutputBuffer.Clear();
}

CompressedCrossSegmentReader::CompressedCrossSegmentReader(StoreSet& storeSet)
    : m_StoreSet(storeSet)
    , m_InputBuffer(BUFSIZE)
    , m_IndexBuffer(StreamVInt::MAX_INDEXES_COUNT)
    , m_InputPos(0)
{}

void CompressedCrossSegmentReader::SetSegment(int segment) {
    m_Segment = segment;
    m_StoreSet.Rewind(segment);
}

Buffer<uint32_t>& CompressedCrossSegmentReader::Read(int op) {
    if (m_InputPos == m_InputBuffer.Size()) {
        m_StoreSet.Stores[op].Read(m_Segment, m_InputBuffer);
        m_InputPos = 0;
        m_LastOp = op;
    }
    ensure(m_LastOp == op);
    m_IndexBuffer.Clear();
    m_InputPos = StreamVInt::Decode(m_InputPos, m_InputBuffer, m_IndexBuffer);
    return m_IndexBuffer;
}

void CompressedCrossSegmentReader::Delete(int segment) {
    m_StoreSet.Delete(segment);
}
