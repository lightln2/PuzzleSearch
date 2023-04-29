#include "DiskBasedSearch.h"

SimpleFileRW::SimpleFileRW(const std::string& fileName)
    : m_FileName(fileName)
{ }

void SimpleFileRW::Write(uint8_t* buffer, size_t count) {
    m_Mutex.lock();
    if (!m_File) m_File = std::make_unique<RWFile>(m_FileName);
    m_File->Write(buffer, m_TotalLength, count);
    m_Offsets.push_back(m_TotalLength);
    m_Lengths.push_back(count);
    m_TotalLength += count;
    m_Mutex.unlock();
}

size_t SimpleFileRW::Read(uint8_t* buffer, size_t count) {
    if (!m_File) return 0;
    if (m_ReadHeader >= m_Offsets.size()) return 0;
    size_t read = m_File->Read(buffer, m_Offsets[m_ReadHeader], m_Lengths[m_ReadHeader]);
    m_ReadHeader++;
    return read;
}

void SimpleFileRW::Delete() {
    if (m_File) m_File.reset();
    m_Offsets.clear();
    m_Lengths.clear();
    m_TotalLength = 0;
    m_ReadHeader = 0;
}

SimpleFileRW& SimpleSegmentedFileRW::GetFile(int segment) {
    while (segment >= m_Files.size()) {
        m_Files.resize(m_Files.size() * 2);
    }
    if (!m_Files[segment]) {
        m_Files[segment] = std::make_unique<SimpleFileRW>(GetFileName(segment));
    }
    return *m_Files[segment];
}
