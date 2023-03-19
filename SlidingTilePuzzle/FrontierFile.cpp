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

