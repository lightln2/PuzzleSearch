#include "SegmentedFile.h"
#include "Util.h"

#include <cassert>
#include <iomanip>
#include <sstream>

std::atomic<uint64_t> SegmentedFile::m_StatReadsCount = 0;
std::atomic<uint64_t> SegmentedFile::m_StatReadNanos = 0;
std::atomic<uint64_t> SegmentedFile::m_StatReadBytes = 0;
std::atomic<uint64_t> SegmentedFile::m_StatWritesCount = 0;
std::atomic<uint64_t> SegmentedFile::m_StatWriteNanos = 0;
std::atomic<uint64_t> SegmentedFile::m_StatWriteBytes = 0;

SegmentedFile::SegmentedFile(int maxSegments, const std::string& filePath)
    : m_FilePath(filePath)
    , m_File(std::make_unique<RWFile>(filePath))
    , m_TotalLength(0)
    , m_Chunks(0)
    , m_Heads(maxSegments, -1)
    , m_Tails(maxSegments, -1)
    , m_ReadPointers(maxSegments, -1)
    , m_Mutex(std::make_unique<std::mutex>())
{
}

uint64_t SegmentedFile::Length(int segment) const {
    uint64_t totalLength = 0;
    int pos = m_Heads[segment];
    while (pos >= 0) {
        totalLength += m_Chunks[pos].length;
        pos = m_Chunks[pos].next;
    }
    return totalLength;
}

void SegmentedFile::Rewind(int segment) {
    m_ReadPointers[segment] = m_Heads[segment];
}

void SegmentedFile::RewindAll() {
    for (int i = 0; i < m_ReadPointers.size(); i++) m_ReadPointers[i] = m_Heads[i];
}

void SegmentedFile::Write(int segment, void* buffer, size_t size) {
    if (size == 0) return;
    Timer timer;
    assert(segment >= 0 && segment < m_Heads.size());

    std::lock_guard<std::mutex> g(*m_Mutex);
    m_File->Write(buffer, m_TotalLength, size);

    int pos = (int)m_Chunks.size();
    m_Chunks.push_back(Chunk{ m_TotalLength, (uint32_t)size, -1 });
    m_TotalLength += size;

    if (m_Heads[segment] == -1) {
        m_Heads[segment] = m_Tails[segment] = m_ReadPointers[segment] = pos;
    } else {
        m_Chunks[m_Tails[segment]].next = pos;
        m_Tails[segment] = pos;
    }

    m_StatWritesCount++;
    m_StatWriteBytes += size;
    m_StatWriteNanos += timer.Elapsed();
}

size_t SegmentedFile::Read(int segment, void* buffer, size_t size) {
    Timer timer;
    assert(segment >= 0 && segment < m_Heads.size());
    if (m_ReadPointers[segment] == -1) return 0;
    auto& chunk = m_Chunks[m_ReadPointers[segment]];
    ensure(chunk.length <= size);

    std::lock_guard<std::mutex> g(*m_Mutex);
    auto read = m_File->Read(buffer, chunk.offset, chunk.length);

    m_ReadPointers[segment] = chunk.next;

    m_StatReadsCount++;
    m_StatReadBytes += read;
    m_StatReadNanos += timer.Elapsed();

    return read;
}

void SegmentedFile::DeleteAll() {
    m_File.reset();
    m_File = std::make_unique<RWFile>(m_FilePath);
    m_TotalLength = 0;
    m_Chunks.clear();
    for (int i = 0; i < m_Heads.size(); i++) {
        m_Heads[i] = m_Tails[i] = m_ReadPointers[i] = -1;
    }
}

void SegmentedFile::PrintStats() {
    std::cerr << "SegmentedFile: "
        << WithDecSep(m_StatReadsCount) << " reads: "
        << WithSize(m_StatReadBytes) << " in " << WithTime(m_StatReadNanos)
        << "; " << WithDecSep(m_StatWritesCount) << " writes: "
        << WithSize(m_StatWriteBytes) << " in " << WithTime(m_StatWriteNanos)
        << std::endl;
}

