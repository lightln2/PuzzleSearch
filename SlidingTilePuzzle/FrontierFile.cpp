#include "FrontierFile.h"
#include "StreamVInt.h"

FrontierBuffer FrontierFileReader::Read() {
    if (m_Position == m_Buffer.Size()) {
        m_File.Read(m_Segment, m_Buffer);
        m_Position = 0;
    }
    m_Indexes.Clear();
    m_Bounds.Clear();
    m_Position = StreamVInt::Decode(m_Position, m_Buffer, m_Indexes, m_Bounds);
    return { m_Indexes.Size(), m_Indexes.Buf(), m_Bounds.Buf() };
}

SmallSegmentWriter::SmallSegmentWriter(SegmentedFile& file, int maxSegments)
    : m_File(file)
    , m_Buffer(new uint8_t[maxSegments * SMALL_BUFFER_SIZE])
    , m_Lengths(maxSegments, 0)
{
    ensure(m_Buffer != nullptr);
}

SmallSegmentWriter::~SmallSegmentWriter() {
    delete[] m_Buffer;
}

void SmallSegmentWriter::Consume(int segment, uint8_t* buf, int size) {
    if (size >= SMALL_BUFFER_SIZE) {
        m_File.Write(segment, buf, size);
    }
    else if (m_Lengths[segment] + size <= SMALL_BUFFER_SIZE) {
        memcpy(m_Buffer + segment * SMALL_BUFFER_SIZE + m_Lengths[segment], buf, size);
        m_Lengths[segment] += size;
    }
    else if (size > m_Lengths[segment]) {
        m_File.Write(segment, buf, size);
    }
    else {
        m_File.Write(segment, m_Buffer + segment * SMALL_BUFFER_SIZE, m_Lengths[segment]);
        memcpy(m_Buffer + segment * SMALL_BUFFER_SIZE, buf, size);
        m_Lengths[segment] = size;
    }
}

void SmallSegmentWriter::FlushAll() {
    /*
    Buffer<uint32_t>& out_segs = GetIndexArray();
    Buffer<uint32_t>& out_lens = GetIndexArray();
    Buffer<uint8_t>& out_buff = GetBufferArray();
    out_segs.Clear();
    out_buff.Clear();
    out_segs.Clear();
    for (int i = 0; i < m_Lengths.size(); i++) {
        auto len = m_Lengths[i];
        if (len == 0) continue;
        auto seg = i;
        if (out_buff.Size() + len > out_buff.Capacity()) {
            // write
            //
            //
            out_segs.Clear();
            out_lens.Clear();
            out_buff.Clear();
        }
        memcpy(out_buff.Buf() + out_buff.Size(), m_Buffer + seg * SMALL_BUFFER_SIZE, len);
        out_segs.Add(seg);
        out_lens.Add(len);
        out_buff.SetSize(out_buff.Size() + len);
    }
    if (!out_buff.IsEmpty()) {
        // write
        //
        //
        out_segs.Clear();
        out_lens.Clear();
        out_buff.Clear();
    }
    ResetPools();
    */
    for (int i = 0; i < m_Lengths.size(); i++) {
        if (m_Lengths[i] > 0) {
            m_File.Write(i, m_Buffer + i * SMALL_BUFFER_SIZE, m_Lengths[i]);
            m_Lengths[i] = 0;
        }
    }
}

Buffer<uint32_t>& SmallSegmentWriter::GetIndexArray() {
    if (m_IndexesPoolPos == m_IndexesPool.size()) {
        m_IndexesPool.emplace_back(std::make_unique<Buffer<uint32_t>>(StreamVInt::MAX_INDEXES_COUNT));
    }
    return *m_IndexesPool[m_IndexesPoolPos++];
}

Buffer<uint8_t>& SmallSegmentWriter::GetBufferArray() {
    if (m_BuffersPoolPos == m_BuffersPool.size()) {
        m_BuffersPool.emplace_back(std::make_unique<Buffer<uint8_t>>(LARGE_BUFFER_SIZE));
    }
    return *m_BuffersPool[m_BuffersPoolPos++];
}
