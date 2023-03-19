#include "FrontierFile.h"
#include "StreamVInt.h"

void FrontierFileWriter::FlushData() {
    int pos = 0;
    pos = FrontierEncode(pos, m_Indexes, m_Bounds, m_Buffer);
    if (pos < m_Indexes.Size()) {
        FlushBuffer();
        pos = FrontierEncode(pos, m_Indexes, m_Bounds, m_Buffer);
    }
    ensure(pos == m_Indexes.Size());
    m_Indexes.Clear();
    m_Bounds.Clear();
}

void FrontierFileWriter::FlushBuffer() {
    m_File.Write(m_Segment, m_Buffer);
    m_Buffer.Clear();
}

FrontierBuffer FrontierFileReader::Read() {
    m_Indexes.Clear();
    m_Bounds.Clear();
    if (m_BufferPosition == m_Buffer.Size()) ReadBuffer();

    m_BufferPosition = FrontierDecode(m_BufferPosition, m_Buffer, m_Indexes, m_Bounds);
    assert(m_BufferPosition <= size);

    return { m_Indexes.Size(), m_Indexes.Buf(), m_Bounds.Buf() };
}

void FrontierFileReader::ReadBuffer() {
    m_File.Read(m_Segment, m_Buffer);
    m_BufferPosition = 0;
}

