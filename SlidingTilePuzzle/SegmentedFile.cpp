#include "SegmentedFile.h"
#include "Util.h"

#include <cassert>
#include <iomanip>
#include <sstream>

std::atomic<uint64_t> SegmentedFile::m_StatReadNanos = 0;
std::atomic<uint64_t> SegmentedFile::m_StatReadBytes = 0;
std::atomic<uint64_t> SegmentedFile::m_StatWriteNanos = 0;
std::atomic<uint64_t> SegmentedFile::m_StatWriteBytes = 0;
std::atomic<uint64_t> SegmentedFile::m_StatDeleteNanos = 0;
std::atomic<uint64_t> SegmentedFile::m_StatCreateNanos = 0;

SegmentedFile::SegmentedFile(int maxSegments, const std::string& directory)
    : m_Directory(directory)
    , m_Files(maxSegments)
    , m_Sizes(maxSegments)
{
    file::CreateDirectory(directory);
}

void SegmentedFile::RewindAll() {
    for (int segment = 0; segment < m_Files.size(); segment++) {
        Rewind(segment);
    }
}

void SegmentedFile::Rewind(int segment) {
    assert(segment >= 0 && segment < m_Files.size());
    if (!HasData(segment)) return;
    m_Files[segment]->Rewind();
}

void SegmentedFile::Write(int segment, void* buffer, size_t size) {
    Timer timer;

    assert(segment >= 0 && segment < m_Files.size());
    auto& file = m_Files[segment];
    if (!file.has_value()) {
        Timer timer;
        auto fileName = SegmentFileName(segment);
        file.emplace(fileName);
        m_StatCreateNanos += timer.Elapsed();
    }
    file->Write(buffer, size);
    m_Sizes[segment] += size;

    m_StatWriteBytes += size;
    m_StatWriteNanos += timer.Elapsed();
}

size_t SegmentedFile::Read(int segment, void* buffer, size_t size) {
    Timer timer;

    assert(segment >= 0 && segment < m_Files.size());
    if (!HasData(segment)) return 0;
    auto read =  m_Files[segment]->Read(buffer, size);

    m_StatReadBytes += read;
    m_StatWriteNanos += timer.Elapsed();

    return read;
}

void SegmentedFile::Delete(int segment) {
    Timer timer;
    m_Files[segment] = std::nullopt;
    m_Sizes[segment] = 0;
    m_StatDeleteNanos += timer.Elapsed();
}

void SegmentedFile::DeleteAll() {
    for (int i = 0; i < m_Files.size(); i++) Delete(i);
}

void SegmentedFile::PrintStats() {
    std::cerr << "SegmentedFile:"
        << " read: " << WithDecSep(m_StatReadBytes) << " in " << WithTime(m_StatReadNanos)
        << " write: " << WithDecSep(m_StatWriteBytes) << " in " << WithTime(m_StatWriteNanos)
        << " create: " << WithTime(m_StatCreateNanos)
        << " delete: " << WithTime(m_StatDeleteNanos)
        << std::endl;
}

std::string SegmentedFile::SegmentFileName(int segment) {
    std::ostringstream stream;
    stream << m_Directory << "/" << std::hex << std::setfill('0') << std::setw(3) << segment;
    return stream.str();
}

