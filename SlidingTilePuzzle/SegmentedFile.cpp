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

SegmentedFile::SegmentedFile(int maxSegments, std::initializer_list<std::string> filePaths)
    : m_TotalLength(0)
    , m_Chunks(0)
    , m_Heads(maxSegments, -1)
    , m_Tails(maxSegments, -1)
    , m_ReadPointers(maxSegments, -1)
    , m_Mutex(std::make_unique<std::mutex>())
{
    for (const auto& file : filePaths) {
        m_FilePaths.push_back(file);
        m_Files.emplace_back(std::make_unique<RWFile>(file));
        m_Mutexes.emplace_back(std::make_unique<std::mutex>());
    }
}

SegmentedFile::SegmentedFile(int maxSegments, const std::vector<std::string>& filePaths)
    : m_TotalLength(0)
    , m_Chunks(0)
    , m_Heads(maxSegments, -1)
    , m_Tails(maxSegments, -1)
    , m_ReadPointers(maxSegments, -1)
    , m_Mutex(std::make_unique<std::mutex>())
{
    for (const auto& file : filePaths) {
        m_FilePaths.push_back(file);
        m_Files.emplace_back(std::make_unique<RWFile>(file));
        m_Mutexes.emplace_back(std::make_unique<std::mutex>());
    }
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
    size_t fileIdx = segment % m_Files.size();

    m_Mutex->lock();
    auto offset = m_TotalLength;
    m_TotalLength += size;
    m_Mutex->unlock();

    m_Mutexes[fileIdx]->lock();
    m_Files[fileIdx]->Write(buffer, offset, size);
    m_Mutexes[fileIdx]->unlock();

    m_Mutex->lock();
    int pos = (int)m_Chunks.size();
    m_Chunks.push_back(Chunk{ offset, (uint32_t)size, -1 });
    m_TotalLength += size;
    m_Mutex->unlock();

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
    size_t fileIdx = segment % m_Files.size();

    //m_Mutex->lock();
    m_Mutexes[fileIdx]->lock();
    auto read = m_Files[fileIdx]->Read(buffer, chunk.offset, chunk.length);
    m_ReadPointers[segment] = chunk.next;
    m_Mutexes[fileIdx]->unlock();
    //m_Mutex->unlock();

    m_StatReadsCount++;
    m_StatReadBytes += read;
    m_StatReadNanos += timer.Elapsed();

    return read;
}

void SegmentedFile::DeleteAll() {
    m_Files.clear();
    for (const auto& file : m_FilePaths) {
        m_Files.emplace_back(std::make_unique<RWFile>(file));
    }
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

